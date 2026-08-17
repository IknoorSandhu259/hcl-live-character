"""Scene memory: what the character looked at once, and what it kept.

This module owns the character's memory of the desk. It never talks to a
network, never touches PyBullet, and never sees a joint -- it holds a small
validated record and hands it back as text when a later turn needs it.

    camera frame --> [vision call, in character.py] --> raw JSON
        --> parse_observation()  (local validation)
        --> SceneMemory.remember()  (one slot, locally owned)

    ...later, with no camera involved...

    SceneMemory.prompt_note() --> appended to the spoken turn's input

Two properties are the point of the whole file.

**The memory is ours, not the model's.** The retained facts live in a
dataclass we can print, assert on and clear. Nothing is remembered by leaving
it in a conversation history, so "what does the character actually know?" has
a definite answer at any moment.

**Recall never looks again.** ``prompt_note`` returns text. There is no code
path from a question to a camera, so answering "what colour was the mug?"
cannot silently re-observe the scene -- if the object has been moved or taken
away, the character still answers from what it noted.

The record is deliberately one object wide. A second observation replaces the
first rather than accumulating, because a list would immediately raise
questions (which one did they mean? is this the same mug?) that this milestone
has no machinery to answer.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional

# --------------------------------------------------------------------------
# The observation contract
# --------------------------------------------------------------------------

#: Rough relative position, from the character's own point of view. An enum
#: rather than free text so the stored location is comparable and printable,
#: and so "vaguely over there somewhere" cannot enter the record.
LOCATIONS = ("left", "center", "right", "unknown")

#: How each location reads when the character speaks about it later.
LOCATION_PHRASES = {
    "left": "on your left",
    "center": "straight ahead of you",
    "right": "on your right",
    "unknown": "somewhere in front of you (you were not sure where)",
}

#: Field caps. These are small because the record is meant to be one glance's
#: worth of fact, not a paragraph, and because everything here is eventually
#: pasted into a later prompt.
MAX_NAME_CHARS = 40
MAX_COLOR_CHARS = 40
MAX_DETAIL_CHARS = 120

#: Exactly the keys the vision response may contain. Anything else is a
#: contract violation, not a field to ignore.
OBSERVATION_FIELDS = ("name", "color", "location", "detail", "confident")


class ObservationError(RuntimeError):
    """A vision response was unusable, so nothing was remembered.

    Always fails closed: the previous memory (usually empty) is left alone.
    """


@dataclass(frozen=True)
class SceneObservation:
    """One validated look at one desk object.

    Immutable, tiny, and printable in a single line -- which is what makes
    "confirm what was stored" a real step in the demo rather than an act of
    faith.
    """

    name: str
    color: str
    #: One of :data:`LOCATIONS`.
    location: str
    #: One extra short fact, or "" when the model had nothing reliable to add.
    detail: str
    #: ``time.time()`` when the *frame was captured*; for the log line.
    observed_at: float = 0.0
    #: ``time.monotonic()`` when the frame was captured; for computing age.
    #: Capture time, not response time, so "you looked at it 30 seconds ago"
    #: counts the seconds the vision call itself took rather than pretending
    #: the character saw the desk at the moment the answer came back.
    observed_monotonic: float = 0.0

    def at(self, observed_at: float, observed_monotonic: float) -> "SceneObservation":
        """A copy stamped with when the frame was actually taken."""
        return replace(
            self, observed_at=observed_at, observed_monotonic=observed_monotonic
        )

    def summary(self) -> str:
        """One line, for the terminal. This is the "what was stored" evidence."""
        parts = [f"{self.name} ({self.color}, {self.location})"]
        if self.detail:
            parts.append(f'"{self.detail}"')
        return " -- ".join(parts)

    def to_dict(self) -> Dict[str, object]:
        """Plain data, for logging or a future on-disk record."""
        return {
            "name": self.name,
            "color": self.color,
            "location": self.location,
            "detail": self.detail,
            "observed_at": self.observed_at,
        }


# --------------------------------------------------------------------------
# Local validation
# --------------------------------------------------------------------------


def _clean(value: object, field: str, limit: int, allow_empty: bool = False) -> str:
    """Return *value* as a single-line, printable, length-bounded string.

    Newlines and control characters are collapsed away rather than rejected.
    That is not cosmetic: the stored text is later pasted into a prompt as a
    block of labelled lines, so a description containing its own newlines could
    otherwise forge an extra line -- a fake fact, or a fake header. Flattening
    to one line removes that possibility structurally instead of relying on the
    model to behave.
    """
    if not isinstance(value, str):
        raise ObservationError(f"observation field '{field}' is not a string")
    # str.split() with no argument splits on any whitespace, including the
    # newlines and tabs we are trying to remove, and drops empty pieces.
    text = " ".join(value.split())
    text = "".join(character for character in text if character.isprintable())
    text = text.strip()
    if not text and not allow_empty:
        raise ObservationError(f"observation field '{field}' is empty")
    if len(text) > limit:
        raise ObservationError(
            f"observation field '{field}' is {len(text)} characters, over the {limit} limit"
        )
    return text


def parse_observation(raw: str, now: Optional[float] = None) -> SceneObservation:
    """Validate a vision response into a :class:`SceneObservation`.

    Applied even though the vision call requests a strict schema: this is the
    check that decides what the character will later claim to have seen, so it
    does not delegate that to the service.

    Raises :class:`ObservationError` for anything malformed *and* for an honest
    "I could not make out a single clear object" -- both mean the same thing to
    the caller, which is that the memory must not be written.

    The record is stamped with *now* (default: this moment), which is when the
    response arrived. :class:`ObservationSession` restamps it with the capture
    time it recorded before the call went out; this default only applies when
    the parser is used on its own.
    """
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ObservationError(f"observation response was not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ObservationError("observation response was not a JSON object")

    unexpected = set(payload) - set(OBSERVATION_FIELDS)
    if unexpected:
        raise ObservationError(
            f"observation response has unexpected fields: {sorted(unexpected)}"
        )
    missing = [field for field in OBSERVATION_FIELDS if field not in payload]
    if missing:
        raise ObservationError(f"observation response is missing fields: {missing}")

    confident = payload["confident"]
    if not isinstance(confident, bool):
        raise ObservationError("observation field 'confident' is not a boolean")
    if not confident:
        # A refusal, not a failure. Nothing is stored, and the demo says so.
        raise ObservationError(
            "no single clear object was identified in the frame; nothing was remembered"
        )

    location = payload["location"]
    if location not in LOCATIONS:
        raise ObservationError(
            f"observation response gave unknown location {location!r}; "
            f"allowed: {list(LOCATIONS)}"
        )

    stamp = time.time() if now is None else now
    return SceneObservation(
        name=_clean(payload["name"], "name", MAX_NAME_CHARS),
        color=_clean(payload["color"], "color", MAX_COLOR_CHARS),
        location=location,
        detail=_clean(payload["detail"], "detail", MAX_DETAIL_CHARS, allow_empty=True),
        observed_at=stamp,
        observed_monotonic=time.monotonic(),
    )


# --------------------------------------------------------------------------
# The memory itself
# --------------------------------------------------------------------------

#: Header of the block appended to a later spoken turn. Named here so the test
#: that proves the facts reached the model can assert on the real string.
NOTE_HEADER = (
    "SCENE MEMORY -- your own note about something you looked at earlier through "
    "your camera. You are NOT looking at it now."
)

#: What is appended when the character has never looked at anything.
EMPTY_NOTE = ""


class SceneMemory:
    """One observation, held locally, readable from either thread.

    Written on the control thread (when an observation comes back) and read on
    the voice worker thread (when a question is being answered), so the single
    slot is guarded by a lock. The lock is uncontended in practice -- both
    operations are a field copy -- and is cheaper than reasoning about whether
    a torn read is possible.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[SceneObservation] = None

    # -- writing -----------------------------------------------------------

    def remember(self, observation: SceneObservation) -> Optional[SceneObservation]:
        """Store *observation*, replacing any previous one. Returns the old one."""
        if not isinstance(observation, SceneObservation):
            raise TypeError("SceneMemory stores SceneObservation values only")
        with self._lock:
            previous, self._latest = self._latest, observation
        return previous

    def forget(self) -> None:
        """Drop what is remembered. Not wired to a key; used by tests."""
        with self._lock:
            self._latest = None

    # -- reading -----------------------------------------------------------

    @property
    def latest(self) -> Optional[SceneObservation]:
        with self._lock:
            return self._latest

    def __bool__(self) -> bool:
        return self.latest is not None

    def prompt_note(self, now: Optional[float] = None) -> str:
        """Render the retained facts for a later turn. Never touches a camera.

        Returns :data:`EMPTY_NOTE` when nothing has been observed, so the
        caller can send the question on its own rather than an empty block the
        model would have to interpret.
        """
        observation = self.latest
        if observation is None:
            return EMPTY_NOTE

        elapsed = (time.monotonic() if now is None else now) - observation.observed_monotonic
        lines = [
            NOTE_HEADER,
            f"- object: {observation.name}",
            f"- colour: {observation.color}",
            f"- where it was: {LOCATION_PHRASES[observation.location]}",
        ]
        if observation.detail:
            lines.append(f"- you also noted: {observation.detail}")
        lines.append(f"- you looked at it about {max(0, int(round(elapsed)))} seconds ago")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Running the bounded observation off the control thread
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationOutcome:
    """What one observation worker finished with."""

    observation: Optional[SceneObservation] = None
    error: Optional[Exception] = None


class ObservationSession:
    """Runs one bounded vision call on a background worker at a time.

    Same shape and the same reasons as ``voice_turn.VoiceSession``: the call is
    network-bound and takes seconds, and the control thread owns PyBullet, the
    camera and the engagement state machine, so it must not block. The worker
    receives an already-encoded JPEG -- a plain ``bytes`` -- and touches
    nothing else, so there is no shared mutable state beyond the one-slot
    result queue.

    Deliberately *not* a loop: an observation happens when the person asks for
    one. Streaming frames to a vision API continuously would cost far more,
    would give the character a memory that silently rewrites itself, and would
    make "what did it store, and when?" unanswerable.

    The session does not own the :class:`SceneMemory`. The control thread polls
    the outcome and writes the memory itself, so every write happens in one
    place that also logs it.
    """

    def __init__(self, brain, spawn: Optional[Callable] = None):
        self._brain = brain
        self._spawn = spawn if spawn is not None else _spawn_thread
        self._results: "queue.Queue[ObservationOutcome]" = queue.Queue(maxsize=1)
        self._running = threading.Event()

    @property
    def busy(self) -> bool:
        return self._running.is_set()

    def ready(self) -> bool:
        return not self.busy and self._results.empty()

    def start(self, jpeg: bytes) -> bool:
        """Send one frame to the vision call. False if one is already in flight.

        The capture time is taken here, on the control thread, immediately
        after the frame was grabbed and encoded -- the record then dates from
        when the character actually looked, not from when the answer came back
        a second or two later.
        """
        if not self.ready():
            return False
        self._running.set()
        captured_at, captured_monotonic = time.time(), time.monotonic()
        self._spawn(lambda: self._work(jpeg, captured_at, captured_monotonic))
        return True

    def _work(self, jpeg: bytes, captured_at: float, captured_monotonic: float) -> None:
        try:
            observation = self._brain.observe(jpeg).at(captured_at, captured_monotonic)
            outcome = ObservationOutcome(observation=observation)
        except BaseException as exc:  # noqa: BLE001 - a worker must never die silently
            outcome = ObservationOutcome(error=exc)
        try:
            # Publish before clearing `busy`, for the same reason VoiceSession
            # does: the other order leaves a window where the session looks
            # idle with a result still in flight.
            self._results.put(outcome)
        finally:
            self._running.clear()

    def poll(self) -> Optional[ObservationOutcome]:
        """Non-blocking: the finished observation, or None. Control thread only."""
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None


def _spawn_thread(work: Callable[[], None]) -> None:
    """One daemon thread, so Ctrl-C still exits the demo."""
    threading.Thread(target=work, name="observation", daemon=True).start()
