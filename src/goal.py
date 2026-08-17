"""The one goal-directed visual action: find a named object and light it.

Hour 4 gave the character a memory. This module gives it a single *job*, and
deliberately only one:

    "find my mug and shine your light on it"

        --> Goal(kind="find_and_light", target="mug")     (parsed here)
        --> fresh frame  -> CharacterBrain.locate(jpeg, "mug")
        --> found + left/center/right
        --> ORIENTATIONS[location](lamp)                  (a named gesture)
        --> A SECOND fresh frame -> locate(jpeg, "mug")   (same object, same side?)
        --> lamp.light_on()

The camera is the laptop's and does not move with the simulated head, so the
second look is a *consistency* check, not an aiming check: the target must
still be there, still on the side the robot aimed at, and still certain. See
:meth:`TargetSighting.matches_expected_location`.

Three things are the point of the file.

**The goal vocabulary is two words wide.** ``kind`` is ``none`` or
``find_and_light`` and there is no third option, no argument list and no
sequence. A model cannot describe an action here because there is no field
that could carry one -- exactly the reason the conversational contract in
``character.py`` is three words wide.

**The target is text, not a parameter.** ``normalize_target`` reduces whatever
the model said to lowercase words, at most :data:`MAX_TARGET_CHARS` of them,
and rejects everything else. The result is used as a *noun in a prompt*; it
never reaches PyBullet, a joint or a pose.

**Orientation is a literal dict of three named behaviours.**
:data:`ORIENTATIONS` maps the coarse location enum onto argument-free
``LampController`` calls. There is no ``getattr``, no ``eval``, no angle and no
pixel arithmetic anywhere in this module -- ``unknown`` is simply absent from
the table, which is how "I am not sure where it is" becomes "do not move".

This module holds no PyBullet, no camera and no network: :class:`GoalSession`
runs the two vision calls and the closing sentence on a worker thread and hands
plain data back to the control thread, which owns every motion.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from scene_memory import LOCATIONS

# --------------------------------------------------------------------------
# The goal contract
# --------------------------------------------------------------------------

#: Every goal the character may hold. "none" is an ordinary conversational
#: turn; there is exactly one action, and adding a second means editing this
#: tuple, the schema in ``character.py`` and the flow in ``engagement_demo.py``
#: by hand. That is the intended cost.
GOAL_KINDS = ("none", "find_and_light")

#: Hard ceiling on the object noun. "water bottle" fits; a sentence does not.
MAX_TARGET_CHARS = 24

#: ...and neither does a phrase. Two words is enough for every everyday desk
#: object worth naming, and it is what stops flattened multi-line text from
#: arriving as a plausible-looking noun.
MAX_TARGET_WORDS = 2

#: Lowercase words separated by single spaces, and nothing else. Digits,
#: punctuation, newlines and quotes are all rejected rather than stripped,
#: because the target is pasted into a prompt as a noun and anything that is
#: not a noun has no business being there.
_TARGET_PATTERN = re.compile(r"^[a-z]+(?: [a-z]+)*$")


class GoalError(RuntimeError):
    """A goal or a target sighting was unusable, so nothing is attempted.

    Always fails closed: no motion, no light.
    """


@dataclass(frozen=True)
class Goal:
    """What the character has been asked to do, if anything."""

    #: One of :data:`GOAL_KINDS`.
    kind: str = "none"
    #: A bounded lowercase noun, or "" when there is no goal.
    target: str = ""

    @property
    def is_action(self) -> bool:
        """True only for the one goal that moves the robot."""
        return self.kind == "find_and_light"

    def summary(self) -> str:
        return "none" if not self.is_action else f"find_and_light({self.target})"


#: The default for every ordinary turn.
NO_GOAL = Goal()


def normalize_target(value: object) -> str:
    """Reduce *value* to a bounded lowercase noun, or refuse it.

    This is the whole of the target's validation, and it runs twice: once when
    the spoken turn's response is parsed, and again inside
    ``CharacterBrain.locate`` immediately before the noun is put in a prompt.
    """
    if not isinstance(value, str):
        raise GoalError("goal target is not a string")
    # split() with no argument splits on any whitespace and drops empties, so
    # newlines and tabs cannot survive into the prompt as line breaks.
    text = " ".join(value.split()).lower()
    text = "".join(character for character in text if character.isprintable())
    if not text:
        raise GoalError("goal target is empty")
    if len(text) > MAX_TARGET_CHARS:
        raise GoalError(
            f"goal target is {len(text)} characters, over the {MAX_TARGET_CHARS} limit"
        )
    if len(text.split(" ")) > MAX_TARGET_WORDS:
        raise GoalError(
            f"goal target {text!r} is a phrase, not a noun of at most "
            f"{MAX_TARGET_WORDS} words"
        )
    if not _TARGET_PATTERN.match(text):
        raise GoalError(f"goal target {text!r} is not a plain lowercase noun")
    return text


def parse_goal(payload: object) -> Goal:
    """Validate the ``goal`` block of a character response.

    *payload* is the already-decoded value of the optional ``goal`` field, so
    ``None`` (an older or goal-free response) is the ordinary case and means
    :data:`NO_GOAL`. Anything present but malformed raises: a turn that cannot
    be understood does not become a turn with a guessed goal.
    """
    if payload is None:
        return NO_GOAL
    if not isinstance(payload, dict):
        raise GoalError("goal is not a JSON object")

    unexpected = set(payload) - {"kind", "target"}
    if unexpected:
        raise GoalError(f"goal has unexpected fields: {sorted(unexpected)}")

    kind = payload.get("kind")
    if kind not in GOAL_KINDS:
        raise GoalError(f"unknown goal kind {kind!r}; allowed: {list(GOAL_KINDS)}")
    if kind == "none":
        # The target is ignored rather than checked: "no goal" is no goal
        # whatever noun came with it.
        return NO_GOAL
    return Goal(kind="find_and_light", target=normalize_target(payload.get("target")))


# --------------------------------------------------------------------------
# The target-specific sighting contract
# --------------------------------------------------------------------------
#
# Three fields, deliberately fewer than the Hour 4 observation: this call is
# not describing the desk, it is answering one yes/no question about one named
# thing, plus which third of the frame it is in.

SIGHTING_FIELDS = ("found", "location", "confident")


@dataclass(frozen=True)
class TargetSighting:
    """One bounded answer to "is the *target* in this frame, and where?"."""

    found: bool
    #: One of :data:`scene_memory.LOCATIONS` -- the same coarse vocabulary the
    #: Hour 4 record uses, reused rather than reinvented.
    location: str
    confident: bool

    @property
    def actionable(self) -> bool:
        """True only if this sighting may cause the robot to turn.

        ``unknown`` is excluded because it is absent from :data:`ORIENTATIONS`;
        the property states that explicitly so the caller cannot get there by
        a lookup that happens to miss.
        """
        return self.found and self.confident and self.location in ORIENTATIONS

    def matches_expected_location(self, expected: Optional[str]) -> bool:
        """True only if this sighting may cause the light to come on.

        Deliberately not a context-free property: what counts as a good second
        look depends on the first one. *expected* is the coarse location that
        caused the orientation -- :attr:`GoalSession.expected_location`.

        The camera is the laptop's, and it is bolted to the laptop. Turning the
        simulated lamp head does not turn it, so a bottle that was on the right
        is *still* on the right in the frame after ``look_right`` -- that is the
        correct answer, not a failure. What the second look proves is that the
        desk has not changed under the aim: same object, same side, still sure.

        So a bottle picked up mid-gesture and put down on the other side comes
        back as ``left`` against an expected ``right``, and the goal fails
        rather than lighting an empty patch of desk. One turn, then this check.
        Nothing here re-aims: a corrective gesture would need the tracking this
        milestone deliberately does not have.
        """
        return (
            self.found
            and self.confident
            # `expected` is one of the three we can actually turn towards, so
            # an absent or 'unknown' expectation can never be matched -- and
            # neither can an 'unknown' sighting, since it is not a key here.
            and expected in ORIENTATIONS
            and self.location == expected
        )

    def summary(self) -> str:
        if not self.found:
            return "not found"
        return f"found ({self.location}{'' if self.confident else ', unsure'})"


def parse_sighting(raw: str) -> TargetSighting:
    """Validate a target-lookup response. Raises on anything else.

    Applied even though the vision call requests a strict schema, for the same
    reason ``parse_observation`` is: this is the check that gates a motion and
    a light, so it does not delegate to the service.
    """
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise GoalError(f"target lookup was not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise GoalError("target lookup was not a JSON object")

    unexpected = set(payload) - set(SIGHTING_FIELDS)
    if unexpected:
        raise GoalError(f"target lookup has unexpected fields: {sorted(unexpected)}")
    missing = [field for field in SIGHTING_FIELDS if field not in payload]
    if missing:
        raise GoalError(f"target lookup is missing fields: {missing}")

    for field in ("found", "confident"):
        if not isinstance(payload[field], bool):
            raise GoalError(f"target lookup field '{field}' is not a boolean")

    location = payload["location"]
    if location not in LOCATIONS:
        raise GoalError(
            f"target lookup gave unknown location {location!r}; allowed: {list(LOCATIONS)}"
        )
    return TargetSighting(
        found=payload["found"], location=location, confident=payload["confident"]
    )


# --------------------------------------------------------------------------
# The orientation allowlist
# --------------------------------------------------------------------------
#
# Keys are the three *usable* members of scene_memory.LOCATIONS; values are
# bound, argument-free LampController behaviours. "center" reuses the existing
# front-facing engage pose rather than inventing a fourth gesture. "unknown" is
# deliberately absent: a location we could not read is a location we do not
# turn towards.

ORIENTATIONS: Dict[str, Callable[[object], None]] = {
    "left": lambda lamp: lamp.look_left(),
    "center": lambda lamp: lamp.engage(),
    "right": lambda lamp: lamp.look_right(),
}

if set(ORIENTATIONS) | {"unknown"} != set(LOCATIONS):
    # Not an assert: this invariant is what keeps the vision vocabulary and the
    # robot's vocabulary identical, and must survive `python -O`.
    raise RuntimeError(
        f"orientation table {sorted(ORIENTATIONS)} does not cover the location "
        f"contract {sorted(LOCATIONS)}"
    )


# --------------------------------------------------------------------------
# What the character says about it
# --------------------------------------------------------------------------
#
# Locally authored, not model-generated. The closing line reports a physical
# fact (the light is on, or it is not), so it is written here where it cannot
# disagree with what actually happened.


def found_line(target: str) -> str:
    return f"There it is. I've put my light on your {target}."


def lost_line(target: str) -> str:
    # Covers both ways the second look can fail: the target is gone, or it is
    # no longer where it was when I aimed at it.
    return f"Your {target} has moved since I aimed at it. I'll leave my light off."


def missing_line(target: str) -> str:
    return f"I've had a look, but I can't see a {target} from here."


# --------------------------------------------------------------------------
# Running the goal, one bounded step at a time
# --------------------------------------------------------------------------
#
# The flow alternates between the two threads, which is the whole reason it is
# a state machine rather than a function:
#
#   PENDING -> [control thread grabs a fresh frame]
#   LOCATE  -> [worker: vision call]        -> control thread plays a gesture
#   MOVED   -> [control thread grabs ANOTHER fresh frame, a tick later]
#   VERIFY  -> [worker: vision call]        -> control thread turns the light on
#   HOLD    -> [control thread waits for the lamp's speaker to be free]
#   SPEAK   -> [worker: text-to-speech]     -> control thread plays the audio
#
# MOVED exists so the second frame cannot be the first one: the session refuses
# to verify until the control thread hands it a frame, and the control thread
# only has a post-motion frame on the tick *after* the gesture played.
#
# HOLD exists for the same kind of reason one step later: the closing sentence
# must not be rendered while the turn's own reply is still coming out of the
# speaker, because both use the same player and the newer preparation would
# discard the staged audio the older playback is still reading.

STAGE_PENDING = "pending"
STAGE_LOCATE = "locate"
STAGE_MOVED = "moved"
STAGE_VERIFY = "verify"
STAGE_HOLD = "hold"
STAGE_SPEAK = "speak"

#: The stages that are waiting on the control thread for a camera frame.
FRAME_STAGES = {STAGE_PENDING: STAGE_LOCATE, STAGE_MOVED: STAGE_VERIFY}


@dataclass(frozen=True)
class GoalOutcome:
    """What one worker step finished with, tagged with its stage and epoch."""

    stage: str
    #: The engagement generation the goal was started under; the control thread
    #: compares it with the current one to spot work that outlived its context.
    epoch: int
    target: str
    sighting: Optional[TargetSighting] = None
    #: A ``PreparedPlayback`` from the SPEAK stage.
    playback: object = None
    error: Optional[BaseException] = None


class GoalSession:
    """Runs the network half of one find-and-light goal off the control thread.

    Same shape and the same reasons as ``voice_turn.VoiceSession`` and
    ``scene_memory.ObservationSession``: the worker receives already-encoded
    JPEG bytes or a plain string and touches nothing else, and the only channel
    back is a one-slot queue of :class:`GoalOutcome`.

    The session never moves the robot and never turns the light on. It only
    reports; the control thread reads the report, plays the named behaviour and
    switches the light, which is why every physical consequence of a goal
    happens in one place.
    """

    def __init__(self, brain, player, spawn: Optional[Callable] = None):
        self._brain = brain
        self._player = player
        self._spawn = spawn if spawn is not None else _spawn_thread
        self._results: "queue.Queue[GoalOutcome]" = queue.Queue(maxsize=1)
        self._running = threading.Event()
        self.goal: Goal = NO_GOAL
        self.epoch = -1
        #: None when idle; otherwise one of the STAGE_* constants.
        self.stage: Optional[str] = None
        #: How many frames this run actually sent to the vision call. The light
        #: may only come on after the second, and tests assert on it.
        self.looks = 0
        #: The closing sentence, written as soon as the outcome is known and
        #: rendered once the lamp's speaker is free.
        self.line = ""
        #: The coarse location the first look reported and the head turned
        #: towards, or None. The whole of what this goal remembers between its
        #: two looks: one word out of the location enum. No image, no
        #: coordinates, no angle. Cleared when a goal starts and when one ends,
        #: so a later goal can never inherit it.
        self.expected_location: Optional[str] = None

    # -- state -------------------------------------------------------------

    @property
    def busy(self) -> bool:
        """True while a worker step is in flight."""
        return self._running.is_set()

    @property
    def active(self) -> bool:
        """True from the moment a goal is accepted until it is finished."""
        return self.stage is not None

    @property
    def occupied(self) -> bool:
        """True while this session still owns *anything* -- the guard to use.

        ``active`` answers "is a goal still being pursued?", which goes false
        the instant a goal is abandoned. That is not the same question as "may
        something else start now?": an abandoned goal's worker is still holding
        the shared brain and the shared player, and its result is still on its
        way to a queue that must be drained before the next goal can use it.

        Every guard that exists to stop two things racing uses this one.
        """
        return self.active or self.busy or not self._results.empty()

    @property
    def awaiting_frame(self) -> bool:
        """True when the control thread owes this goal a fresh camera frame."""
        return self.stage in FRAME_STAGES

    @property
    def held(self) -> bool:
        """True when the closing line is written and waiting for the speaker."""
        return self.stage == STAGE_HOLD

    # -- control-thread API ------------------------------------------------

    def request(self, goal: Goal, epoch: int) -> bool:
        """Accept *goal*. False if it is not an action or one is already running."""
        if not isinstance(goal, Goal) or not goal.is_action:
            return False
        if self.occupied:
            return False
        self.goal = goal
        self.epoch = epoch
        self.stage = STAGE_PENDING
        self.looks = 0
        self.line = ""
        self.expected_location = None
        return True

    def supply_frame(self, jpeg: bytes) -> bool:
        """Hand the next vision call a frame the control thread just captured.

        False unless the goal is actually waiting for one, so a frame can never
        skip the movement step or start a third look.
        """
        next_stage = FRAME_STAGES.get(self.stage)
        if next_stage is None or not jpeg or self.busy:
            return False
        target = self.goal.target
        self.stage = next_stage
        self.looks += 1
        self._start(next_stage, lambda: self._brain.locate(jpeg, target))
        return True

    def oriented(self, location: str) -> bool:
        """Record that the named orientation behaviour for *location* has played.

        Only legal straight after the first look, and it is the *only* way to
        reach the verification stage -- which is what makes the second frame
        mandatory rather than merely usual.

        *location* is the coarse word the first look reported and that chose the
        gesture; it is kept so the second look can be compared against it. Only
        the three we can actually turn towards are accepted, so the expectation
        can never be something the robot did not act on.
        """
        if self.stage != STAGE_LOCATE or self.busy:
            return False
        if location not in ORIENTATIONS:
            return False
        self.expected_location = location
        self.stage = STAGE_MOVED
        return True

    def hold(self, line: str) -> bool:
        """Write the closing sentence and wait for the lamp's speaker.

        Deliberately separate from :meth:`say`: the outcome is known now, but
        rendering it means preparing audio on the shared player, and that must
        not happen while the turn's own reply is still being played.
        """
        if not self.active or self.busy or not line:
            return False
        self.line = line
        self.stage = STAGE_HOLD
        return True

    def say(self) -> bool:
        """Render the held line. Legal only from HOLD, and only when idle."""
        if self.stage != STAGE_HOLD or self.busy:
            return False
        line = self.line
        self.stage = STAGE_SPEAK
        self._start(STAGE_SPEAK, lambda: self._player.prepare(self._brain.synthesize(line)))
        return True

    def finish(self) -> None:
        """Drop the goal, whatever stage it was in. Idempotent.

        This ends the goal *logically*. It does not cancel a worker that is
        already running, and it deliberately does not discard a queued result:
        that result still has to come back through :meth:`poll` so the control
        thread can see it, ignore it and release the session. Until then
        :attr:`occupied` stays true.
        """
        self.stage = None
        self.goal = NO_GOAL
        self.line = ""
        self.expected_location = None

    def poll(self) -> Optional[GoalOutcome]:
        """Non-blocking: the finished step, or None. Control thread only.

        A result is withheld while a worker is still marked busy. The worker
        publishes its outcome *before* clearing ``_running`` -- deliberately,
        so the session is never externally "idle with a result in flight" --
        and without this check the control thread could dequeue that outcome
        during the window in between, then be told ``False`` by the very
        transition it is trying to make. The goal would sit active forever,
        possibly lit, refusing every later turn.

        Checking ``_running`` first closes the window from this side: if the
        flag is clear then the ``put`` that precedes it has already happened,
        so nothing is lost by waiting.
        """
        if self._running.is_set():
            return None
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    # -- the worker --------------------------------------------------------

    def _start(self, stage: str, work: Callable[[], object]) -> None:
        self._running.set()
        epoch, target = self.epoch, self.goal.target

        def run() -> None:
            try:
                value = work()
                if isinstance(value, TargetSighting):
                    outcome = GoalOutcome(
                        stage=stage, epoch=epoch, target=target, sighting=value
                    )
                else:
                    outcome = GoalOutcome(
                        stage=stage, epoch=epoch, target=target, playback=value
                    )
            except BaseException as exc:  # noqa: BLE001 - a worker must never die silently
                outcome = GoalOutcome(stage=stage, epoch=epoch, target=target, error=exc)
            try:
                # Publish before clearing `busy`, for the same reason the other
                # two sessions do: the other order leaves a window where the
                # session looks idle with a result still in flight.
                self._results.put(outcome)
            finally:
                self._running.clear()

        self._spawn(run)


def _spawn_thread(work: Callable[[], None]) -> None:
    """One daemon thread, so Ctrl-C still exits the demo."""
    threading.Thread(target=work, name="goal", daemon=True).start()
