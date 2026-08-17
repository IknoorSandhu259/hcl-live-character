"""Orchestrator for one spoken turn, and the model-to-action boundary.

    microphone -> STT -> character -> {reply, behavior} -> LampController -> speaker

This file is where a word chosen by a language model becomes robot motion, so
it is deliberately the dullest possible piece of code: a literal dict from the
three allowed words to three zero-argument calls on the already-safe
``LampController``. There is no ``eval``, no ``getattr(lamp, name)``, no tool
registry and no way to reach a joint -- adding a behaviour means editing
:data:`BEHAVIORS` by hand, which is the point.

Two-phase turn
--------------
A turn is split at the moment the robot commits to moving:

    prepare()   record -> transcribe -> respond -> validate -> synthesize
                -> decode + validate audio -> confirm the output device
    ------------------------- commit point -------------------------
    commit()    play one named gesture -> hand the audio to PortAudio

Everything fallible and slow lives in ``prepare``. By the time ``commit`` runs,
the reply has been written, spoken audio exists, it has been decoded into
samples, and the sound card has said it will accept them. If any of that fails
the turn is reported as failed and *the robot does not move at all*.

This is a strong ordering guarantee, not an absolute one: a device unplugged
between the two phases, or an audio server that dies mid-buffer, can still
leave the character having gestured without speaking. That residual window is
one PortAudio call wide and is not worth a rollback mechanism.

``prepare`` touches no PyBullet and no LampController, so it is safe to run on
a worker thread; ``commit`` must run on the thread that owns the simulator.
:class:`VoiceSession` enforces exactly that split.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from audio_io import AudioError, PreparedPlayback, Player, Recorder
from character import ALLOWED_BEHAVIORS, CharacterBrain, CharacterError, redact


class TurnError(RuntimeError):
    """A spoken turn failed.

    Raised from ``prepare`` it guarantees no robot motion happened. Raised from
    ``commit`` it means the gesture may already have played; the message says
    which. Redacts its own message, so no credential can reach a terminal
    through this type.
    """

    def __init__(self, message: str = ""):
        super().__init__(redact(str(message)))


# --------------------------------------------------------------------------
# The allowlist
# --------------------------------------------------------------------------
#
# Keys are exactly character.ALLOWED_BEHAVIORS; values are bound, argument-free
# LampController gestures. A turn dispatches at most one entry, once.

BEHAVIORS: Dict[str, Callable[[object], None]] = {
    "none": lambda _lamp: None,
    "nod": lambda lamp: lamp.nod(),
    "engage": lambda lamp: lamp.engage(),
}

if set(BEHAVIORS) != set(ALLOWED_BEHAVIORS):
    # Not an assert: this invariant is what keeps the model's vocabulary and
    # the robot's vocabulary identical, and must survive `python -O`.
    raise RuntimeError(
        f"dispatch table {sorted(BEHAVIORS)} does not match the "
        f"contract {sorted(ALLOWED_BEHAVIORS)}"
    )


@dataclass(frozen=True)
class PreparedTurn:
    """A validated turn waiting for the control thread to commit it."""

    transcript: str
    reply: str
    behavior: str
    playback: PreparedPlayback


@dataclass(frozen=True)
class TurnResult:
    """What happened in one committed turn, for logging and tests."""

    transcript: str
    reply: str
    behavior: str
    #: False for "none"; True when exactly one gesture was played.
    moved: bool
    #: How long the character will be talking, so the caller can stay quiet.
    speech_seconds: float


class VoiceTurn:
    """One push-to-talk exchange, split into prepare and commit.

    All four collaborators are injected, so a unit test can drive the whole
    path with fakes and no hardware, no key and no network.
    """

    def __init__(self, brain: CharacterBrain, recorder: Recorder, player: Player, lamp):
        self.brain = brain
        self.recorder = recorder
        self.player = player
        self.lamp = lamp

    # -- phase 1: everything that may fail, before anything that may move ---

    def prepare(self) -> PreparedTurn:
        """Listen, think and render speech. Never touches the robot.

        Raises :class:`TurnError` for every recoverable failure; the caller can
        report it and carry on. Runs on a worker thread in the live demo, so it
        deliberately avoids PyBullet, LampController and the OpenCV GUI.
        """
        wav = _stage("recording", self.recorder.record)
        if not wav:
            raise TurnError("recording failed: the microphone returned no audio")

        transcript = _stage("speech-to-text", self.brain.transcribe, wav)
        reply = _stage("character response", self.brain.respond, transcript)
        speech = _stage("text-to-speech", self.brain.synthesize, reply.reply)

        # Decoding, format validation and the device's own "will you accept
        # this rate?" check all happen here, before the commit point -- this is
        # the part that used to sit after the gesture.
        playback = _stage("audio preparation", self.player.prepare, speech)

        # Belt and braces: parse_character_reply already rejected anything
        # outside the enum, so a miss here would mean the two lists drifted.
        if reply.behavior not in BEHAVIORS:
            raise TurnError(f"no dispatch entry for behavior {reply.behavior!r}")

        return PreparedTurn(
            transcript=transcript,
            reply=reply.reply,
            behavior=reply.behavior,
            playback=playback,
        )

    # -- phase 2: commit ---------------------------------------------------

    def commit(self, prepared: PreparedTurn) -> TurnResult:
        """Play at most one gesture, then start the audio. Control thread only."""
        action = BEHAVIORS.get(prepared.behavior)
        if action is None:
            raise TurnError(f"no dispatch entry for behavior {prepared.behavior!r}")

        # A behaviour is a LampController call and can in principle raise (a
        # disconnected simulator, a joint that vanished). That is a failed turn,
        # not a dead demo.
        _stage("behavior", action, self.lamp)

        try:
            self.player.play(prepared.playback)
        except Exception as exc:
            # Past the commit point: the gesture already played. Say so rather
            # than implying nothing happened.
            raise TurnError(
                f"playback failed after the gesture had already played: {_describe(exc)}"
            ) from exc

        return TurnResult(
            transcript=prepared.transcript,
            reply=prepared.reply,
            behavior=prepared.behavior,
            moved=prepared.behavior != "none",
            speech_seconds=prepared.playback.seconds,
        )

    def run(self) -> TurnResult:
        """prepare + commit, synchronously. Convenient for tests and scripts."""
        return self.commit(self.prepare())


def _stage(label: str, call, *args):
    """Run one stage of a turn, funnelling every failure into TurnError.

    The catch is deliberately broad. The stages call into the OpenAI SDK, the
    ``wave`` module, NumPy, PortAudio and PyBullet, and between them those can
    raise ``OSError``, ``EOFError``, ``ValueError``, ``struct.error``,
    ``ImportError``, ``KeyError`` and library-private exception types. A
    spoken turn is a recoverable unit of work: any of those means "this turn
    failed", never "tear the demo down".
    """
    try:
        return call(*args)
    except TurnError:
        raise
    except CharacterError as exc:
        # Already names its own stage ("speech-to-text failed: ...").
        raise TurnError(str(exc)) from exc
    except AudioError as exc:
        raise TurnError(f"{label} failed: {exc}") from exc
    except Exception as exc:
        raise TurnError(f"{label} failed: {_describe(exc)}") from exc


def _describe(exc: BaseException) -> str:
    """First line of an exception plus its type; never ``repr``."""
    try:
        message = str(exc).splitlines()[0] if str(exc) else ""
    except Exception:
        message = ""
    name = type(exc).__name__
    return f"{name}: {message}" if message else name


# --------------------------------------------------------------------------
# Running the slow half off the control thread
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnOutcome:
    """What a worker finished with, tagged with the engagement it belongs to."""

    #: The engagement generation the turn was started under. The control thread
    #: compares this with the current one to spot a stale result.
    epoch: int
    prepared: Optional[PreparedTurn] = None
    error: Optional[TurnError] = None


class VoiceSession:
    """Runs ``VoiceTurn.prepare`` on one background worker at a time.

    The smallest thing that keeps the engagement loop responsive: recording,
    three API calls and audio decoding -- together several seconds -- move off
    the control thread, and the *only* channel back is a one-slot queue holding
    a :class:`TurnOutcome`. The worker never touches PyBullet, LampController,
    OpenCV or the tracker, so there is no shared mutable state to lock.

    At most one turn is in flight. The control thread stays the sole owner of
    robot motion: it calls ``commit`` itself, after checking the result is
    still relevant.
    """

    def __init__(self, turn: VoiceTurn, spawn: Optional[Callable] = None):
        self._turn = turn
        #: Injectable so tests can run the worker inline instead of threading.
        self._spawn = spawn if spawn is not None else _spawn_thread
        self._results: "queue.Queue[TurnOutcome]" = queue.Queue(maxsize=1)
        self._running = threading.Event()
        #: Wall-clock time before which a new turn would talk over the lamp.
        self._quiet_until = 0.0

    @property
    def busy(self) -> bool:
        """True while a worker is preparing a turn."""
        return self._running.is_set()

    def ready(self, now: float) -> bool:
        """True when a new turn may start: idle, drained, and not mid-sentence."""
        return not self.busy and self._results.empty() and now >= self._quiet_until

    def start(self, epoch: int, now: float) -> bool:
        """Kick off a turn for engagement generation *epoch*. False if not ready."""
        if not self.ready(now):
            return False
        self._running.set()
        self._spawn(lambda: self._work(epoch))
        return True

    def _work(self, epoch: int) -> None:
        try:
            outcome = TurnOutcome(epoch=epoch, prepared=self._turn.prepare())
        except TurnError as exc:
            outcome = TurnOutcome(epoch=epoch, error=exc)
        except BaseException as exc:  # noqa: BLE001 - a worker must never die silently
            outcome = TurnOutcome(
                epoch=epoch, error=TurnError(f"voice worker crashed: {_describe(exc)}")
            )
        try:
            # Publish before clearing `busy`, never after: in the other order
            # there is a window where the session looks idle with a result
            # still in flight, and a second turn could start behind it.
            self._results.put(outcome)
        finally:
            self._running.clear()

    def poll(self) -> Optional[TurnOutcome]:
        """Non-blocking: the finished turn, or None. Control thread only."""
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def mark_speaking(self, seconds: float, now: float) -> None:
        """Refuse new turns until the character has finished this sentence.

        Without this, a second push-to-talk would record the lamp's own voice.
        """
        self._quiet_until = max(self._quiet_until, now + max(0.0, seconds))


def _spawn_thread(work: Callable[[], None]) -> None:
    """Default worker: one daemon thread, so Ctrl-C still exits the demo."""
    threading.Thread(target=work, name="voice-turn", daemon=True).start()
