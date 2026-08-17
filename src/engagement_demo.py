"""Camera-driven engagement demo for the dummy five-DOF character lamp.

Run:  python src/engagement_demo.py
      python src/engagement_demo.py --no-preview      (no camera window)
      python src/engagement_demo.py --no-voice        (camera only, no API key)
      python src/engagement_demo.py --camera-index 1  (pick another webcam)

The character starts neutral. Look at the laptop camera and it sits up
(``LampController.engage``); look away or leave for a few seconds and it
settles back down (``LampController.neutral``).

While it is ENGAGED, press Enter (or 't' in the preview window) to talk: the
demo records a short bounded utterance, sends it through speech-to-text, the
character model and text-to-speech, plays the reply through the speaker and
performs at most one named gesture.

Also while ENGAGED, press 'o' (in the lamp window or the camera preview) and
the character takes one look at the desk: a single frame goes to the vision
call, and the five validated fields that come back are stored in local scene
memory and printed. A later spoken question is answered from that record --
the recall path has no camera in it at all, so the answer survives the object
being moved or taken away.

This file is the *wiring* only, and is deliberately thin:

    attention.AttentionSensor      camera -> "is a face facing us?"  (perception)
    attention.EngagementTracker    noisy booleans -> IDLE / ENGAGED  (policy)
    attention.encode_frame_jpeg    one frame -> bounded JPEG bytes
    audio_io                       microphone / speaker              (audio)
    character.CharacterBrain       transcript + note -> {reply, behavior}  (language)
                                   one frame -> a scene observation
    scene_memory.SceneMemory       the retained facts, locally owned
    scene_memory.ObservationSession  the vision call, off this thread
    voice_turn.VoiceTurn           one turn, and the action allowlist
    voice_turn.VoiceSession        the slow half of a turn, off this thread
    robot_controller.LampController   IDLE / ENGAGED / gesture -> motion (body)

Perception and language never issue a joint command, and the body layer never
sees a pixel or a token. Every joint command still goes through
LampController, which owns the joint limits.

The loop below owns PyBullet, the tracker and the OpenCV window, and never
blocks on the network. A spoken turn's recording, API calls and audio decoding
run on a single worker thread; this loop only polls for the finished result
and commits it. So the camera keeps being read, engagement keeps updating and
the character can still disengage while it is thinking about an answer.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention import (  # noqa: E402
    DETECT_HZ,
    AttentionSensor,
    CameraError,
    EngagementState,
    EngagementTracker,
    encode_frame_jpeg,
)
from audio_io import MicrophoneRecorder, SpeakerPlayer  # noqa: E402
from character import (  # noqa: E402
    CharacterBrain,
    CharacterError,
    MissingCredentialsError,
    require_api_key,
)
from robot_controller import LampController, load_lamp  # noqa: E402
from scene_memory import ObservationSession, SceneMemory  # noqa: E402
from voice_turn import TurnError, VoiceSession, VoiceTurn  # noqa: E402

#: Simulation time advanced per perception tick while no behaviour is playing.
#: The lamp is holding a pose here, so this only has to keep the position
#: controller ticking over; the frame grab dominates the loop period anyway.
IDLE_SIM_STEP = 1.0 / 60.0

PREVIEW_WINDOW = "lamp attention"

#: Push-to-talk keys accepted by the preview window: 't', Enter, keypad Enter.
TALK_KEYS = (ord("t"), 13, 10)

#: "Have a look at the desk" in the preview window.
OBSERVE_KEYS = (ord("o"),)


def _draw_preview(reading, state: EngagementState) -> None:
    """Show what the perception layer actually sees. Debug aid, not a UI.

    Annotates a *copy*. ``cv2.rectangle`` and ``cv2.putText`` draw in place, and
    the same ``reading.frame`` is what an observation uploads later in the same
    tick -- so drawing on the original would send the vision model our own face
    boxes and status text instead of the camera image. The copy costs about a
    megabyte per tick at 640x480 and removes the coupling entirely.
    """
    if reading.frame is None:
        return
    frame = reading.frame.copy()
    colour = (0, 200, 0) if state is EngagementState.ENGAGED else (0, 165, 255)
    for (x, y, w, h) in reading.faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    cv2.putText(
        frame,
        f"{state.value}   attending={reading.attending}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        colour,
        2,
    )
    cv2.imshow(PREVIEW_WINDOW, frame)


class TerminalTalkKey:
    """Non-blocking "did the user press Enter?" check for the terminal.

    The perception loop must keep running at DETECT_HZ, so it cannot sit in
    ``input()``. This peeks at stdin instead and returns immediately. Kept
    behind a tiny interface (``pressed()``) so tests inject a scripted stand-in
    and never touch a real terminal.

    POSIX reads the raw descriptor with ``os.read`` rather than
    ``sys.stdin.readline()``. ``sys.stdin`` is a buffered ``TextIOWrapper``:
    ``select`` reports on the *file descriptor* while ``readline`` reads
    through the buffer, so the two can disagree about whether input is
    pending, and a ``readline`` that guesses wrong blocks the whole loop.
    Reading the descriptor directly keeps the two in step.
    """

    def __init__(self) -> None:
        self.fileno = None
        try:
            if sys.stdin is not None and sys.stdin.isatty():
                self.fileno = sys.stdin.fileno()
        except (OSError, ValueError):
            self.fileno = None  # detached or closed stdin: silently inert

    @property
    def available(self) -> bool:
        return self.fileno is not None

    def pressed(self) -> bool:
        if not self.available:
            return False
        if sys.platform == "win32":
            import msvcrt  # noqa: PLC0415  (platform specific)

            hit = False
            while msvcrt.kbhit():
                # Drain everything buffered so a burst counts as one request.
                hit = msvcrt.getwch() in ("\r", "\n", "t") or hit
            return hit

        import select  # noqa: PLC0415  (platform specific)

        hit = False
        while True:
            try:
                ready, _, _ = select.select([self.fileno], [], [], 0)
                if not ready:
                    return hit
                chunk = os.read(self.fileno, 1024)
            except (OSError, ValueError):
                return hit
            if not chunk:
                return hit  # stdin closed
            # Anything the user typed and committed with Enter counts; the
            # whole buffer is drained so a burst is one request, not several.
            hit = hit or b"\n" in chunk or b"\r" in chunk or b"t" in chunk


class SimulatorKeyboard:
    """The PyBullet window's keyboard, read exactly once per control tick.

    ``p.getKeyboardEvents`` is *draining*: it reports what happened since the
    last call, so a KEY_WAS_TRIGGERED flag is delivered to whoever asks first
    and to nobody else. With push-to-talk and the observation key each asking
    on their own, the talk poll ran first every tick and silently ate the 'o'
    press before the observation handler could see it.

    So the loop polls here once, and both keys read the same event set.
    """

    def __init__(self, client_id: int):
        self.client_id = client_id
        #: Keys that were triggered during the tick just polled.
        self.triggered: frozenset = frozenset()
        #: How many times the window was actually asked. Tests assert on it.
        self.polls = 0

    def poll(self) -> None:
        """Drain the window's events into :attr:`triggered`. Once per tick."""
        self.polls += 1
        try:
            events = p.getKeyboardEvents(physicsClientId=self.client_id)
        except Exception:
            events = {}  # no GUI, or the client went away mid-shutdown
        self.triggered = frozenset(
            key for key, state in events.items() if state & p.KEY_WAS_TRIGGERED
        )


class SimulatorKey:
    """Reports whether one set of keys is in the current tick's events.

    Given a shared :class:`SimulatorKeyboard` it is a pure reader and polls
    nothing. Given only a client id it owns a keyboard and polls it itself,
    which is how it behaves outside the demo loop.

    Inert (and harmless) in DIRECT/headless mode, where there is no window.
    """

    KEYS: tuple = ()

    def __init__(
        self,
        client_id: Optional[int] = None,
        keyboard: Optional[SimulatorKeyboard] = None,
        keys: Optional[tuple] = None,
    ):
        if keyboard is None and client_id is None:
            raise ValueError("a SimulatorKey needs either a keyboard or a client id")
        self.keyboard = keyboard if keyboard is not None else SimulatorKeyboard(client_id)
        #: True when nobody else is polling for us.
        self.self_polling = keyboard is None
        self.keys = frozenset(keys if keys is not None else self.KEYS)

    @property
    def client_id(self) -> int:
        return self.keyboard.client_id

    def pressed(self) -> bool:
        if self.self_polling:
            self.keyboard.poll()
        return bool(self.keyboard.triggered & self.keys)


class SimulatorTalkKey(SimulatorKey):
    """Push-to-talk from inside the PyBullet GUI window."""

    #: Enter, space, or 't'. Enter and space are what people try first.
    KEYS = (p.B3G_RETURN, ord(" "), ord("t"))


class SimulatorObserveKey(SimulatorKey):
    """"Have a look at the desk" from inside the PyBullet GUI window.

    Deliberately its own key rather than a spoken command: an observation is
    the one event that sends a picture off this machine, so it happens when a
    person presses a key for it and at no other time.
    """

    KEYS = (ord("o"),)


def build_simulator_keys(client_id: int):
    """One keyboard, two readers. Returns ``(keyboard, talk, observe)``.

    Constructing them together is what guarantees the single poll: there is no
    arrangement of these three objects in which two ``getKeyboardEvents`` calls
    can happen in one tick.
    """
    keyboard = SimulatorKeyboard(client_id)
    return keyboard, SimulatorTalkKey(keyboard=keyboard), SimulatorObserveKey(keyboard=keyboard)


class AnyTalkKey:
    """True if any source saw a keypress this tick.

    Polls every source rather than short-circuiting, so an unread terminal
    buffer still gets drained when the GUI answered first.
    """

    def __init__(self, *sources):
        self.sources = [source for source in sources if source is not None]

    def pressed(self) -> bool:
        return any([source.pressed() for source in self.sources])


def build_voice_turn(
    lamp: LampController,
    memory: Optional[SceneMemory] = None,
    brain: Optional[CharacterBrain] = None,
) -> VoiceTurn:
    """Default wiring of the spoken path onto a live lamp.

    *brain* is shared with the observation path when there is one, so the
    process holds a single OpenAI client. *memory* is the same object the
    observation path writes to; the turn only reads it.
    """
    return VoiceTurn(
        brain=brain if brain is not None else CharacterBrain(),
        recorder=MicrophoneRecorder(),
        player=SpeakerPlayer(),
        lamp=lamp,
        memory=memory,
    )


def _log(message: str, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=stream, flush=True)


def _handle_outcome(outcome, session, voice, sensor, tracker, epoch: int) -> None:
    """Apply one finished turn on the control thread. Reports, never raises.

    This is the only place robot motion can result from something the model
    said, and it happens on the thread that owns PyBullet.
    """
    if outcome.error is not None:
        # Failed inside prepare(): voice_turn guarantees the robot was never
        # touched, so there is nothing to undo.
        _log(f"turn failed: {outcome.error}", error=True)
        return

    prepared = outcome.prepared
    if outcome.epoch != epoch or tracker.state is not EngagementState.ENGAGED:
        # The person left (or left and came back) while we were thinking.
        # Answering now would be the character talking to an empty chair, so
        # the result is dropped whole -- gesture and audio together.
        _log(f'discarded a stale turn: "{prepared.transcript}" (no longer engaged)')
        return

    try:
        result = voice.commit(prepared)
    except TurnError as exc:
        _log(f"turn failed: {exc}", error=True)
        return

    session.mark_speaking(result.speech_seconds, time.monotonic())
    _log(f'heard: "{result.transcript}"')
    _log(f'said:  "{result.reply}"  behavior={result.behavior}')
    if result.moved:
        # The gesture blocked the loop for a beat; drop the frames the driver
        # queued behind it so the tracker reasons about now, not then.
        sensor.flush()


def _handle_observation(outcome, memory: SceneMemory) -> None:
    """Apply one finished observation on the control thread. Never raises.

    The single place the scene memory is written, so "what does the character
    know, and when did it learn it?" is answered by these two log lines.
    """
    if outcome.error is not None:
        # Includes the honest "I could not make out one clear object": nothing
        # is stored, and the previous note (if any) is left untouched.
        _log(f"looked, but remembered nothing: {outcome.error}", error=True)
        return

    previous = memory.remember(outcome.observation)
    if previous is not None:
        _log(f"replacing what I had noted: {previous.summary()}")
    _log(f"remembered: {outcome.observation.summary()}")
    _log(f"scene memory now holds: {outcome.observation.to_dict()}")


def run(
    camera_index: int = 0,
    preview: bool = True,
    headless: bool = False,
    sensor=None,
    voice_factory=None,
    talk_key=None,
    memory: Optional[SceneMemory] = None,
    observer: Optional[ObservationSession] = None,
    observe_key=None,
) -> None:
    """Drive the lamp from the camera until interrupted.

    *sensor* exists so the loop can be exercised headlessly against a scripted
    fake reader; leave it None to use the real webcam. *voice_factory* is
    called with the LampController to build the spoken path; leave it None for
    the camera-only demo. *talk_key* answers "does the user want to talk now?".

    *observer* is the bounded observation path; when it is present, *memory* is
    the record it fills and the spoken path reads. *observe_key* answers "does
    the user want me to look at the desk now?".
    """
    memory = memory if memory is not None else SceneMemory()
    # Open the camera before spending time on the simulator, so a permission
    # or device problem surfaces immediately instead of behind a GUI window.
    if sensor is None:
        sensor = AttentionSensor(camera_index=camera_index).open()

    client = None
    try:
        client = p.connect(p.DIRECT if headless else p.GUI)
        if client < 0:
            raise RuntimeError("Could not connect to PyBullet")

        p.setGravity(0, 0, -9.81, physicsClientId=client)
        if not headless:
            # Same face-on viewpoint as demo.py.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.25,
                cameraYaw=272,
                cameraPitch=-10,
                cameraTargetPosition=[-0.10, 0.0, 0.50],
                physicsClientId=client,
            )

        lamp = LampController(load_lamp(client), client)
        tracker = EngagementTracker()

        voice = voice_factory(lamp) if voice_factory is not None else None
        session = VoiceSession(voice) if voice is not None else None

        # One keyboard for the lamp window, shared by both keys and polled once
        # per tick by the loop below -- see SimulatorKeyboard for why.
        keyboard = sim_talk = sim_observe = None
        needs_window_keys = (voice is not None and talk_key is None) or (
            observer is not None and observe_key is None
        )
        if needs_window_keys and not headless:
            keyboard, sim_talk, sim_observe = build_simulator_keys(client)

        if voice is not None and talk_key is None:
            # Two sources, because which window has focus is not ours to
            # decide: the terminal, and the PyBullet window when there is one.
            talk_key = AnyTalkKey(TerminalTalkKey(), sim_talk)
        if observer is not None and observe_key is None:
            # 'o' comes from a window, not the terminal: the terminal watcher
            # drains stdin wholesale for push-to-talk, so a second reader on the
            # same descriptor would steal its bytes.
            observe_key = sim_observe
        #: Bumped on every IDLE -> ENGAGED transition. A turn carries the value
        #: it started under, which is how a result that outlived its
        #: conversation is recognised and dropped.
        epoch = 0

        lamp.neutral()
        print(f"[{time.strftime('%H:%M:%S')}] ready in {tracker.state.value}; "
              "look at the camera to engage. Ctrl-C (or 'q' in the preview) to quit.")
        if voice is not None:
            print("      while ENGAGED, press Enter to talk -- in this terminal, "
                  "in the lamp window, or 't' in the camera preview.")
        if observer is not None:
            print("      while ENGAGED, press 'o' -- in the lamp window or the camera "
                  "preview -- to look at the desk and remember one object.")

        period = 1.0 / DETECT_HZ
        while True:
            tick = time.monotonic()
            try:
                reading = sensor.read()
            except StopIteration:
                break  # a scripted fake sensor ran out of frames
            transition = tracker.update(reading.attending, tick)

            # Exactly one getKeyboardEvents call per tick, before either key
            # reads it. Asking twice would let the first reader drain the 'o'
            # press the second one was waiting for.
            if keyboard is not None:
                keyboard.poll()

            wants_to_talk = talk_key.pressed() if talk_key is not None else False
            wants_to_look = observe_key.pressed() if observe_key is not None else False
            if preview:
                _draw_preview(reading, tracker.state)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                wants_to_talk = wants_to_talk or key in TALK_KEYS
                wants_to_look = wants_to_look or key in OBSERVE_KEYS

            if transition is EngagementState.ENGAGED:
                epoch += 1
                print(f"[{time.strftime('%H:%M:%S')}] IDLE -> ENGAGED")
                lamp.engage()
                sensor.flush()
            elif transition is EngagementState.IDLE:
                print(f"[{time.strftime('%H:%M:%S')}] ENGAGED -> IDLE")
                lamp.neutral()
                sensor.flush()
            else:
                # No transition: hold the pose and keep the physics ticking.
                lamp.step(IDLE_SIM_STEP)

            # Finished observations are banked *first*, before anything this
            # tick can consult or extend the memory. Two things fall out of
            # that ordering: a second 'o' press is judged against a session
            # that has already been drained rather than being refused for a
            # result nobody collected yet, and a spoken turn started later in
            # this same tick reads a memory that is up to date.
            #
            # An observation is about the desk, not about the person, so unlike
            # a spoken turn it is still worth keeping if they looked away while
            # the call was in flight.
            if observer is not None:
                seen = observer.poll()
                if seen is not None:
                    _handle_observation(seen, memory)

            # One frame, once, when a person asks for it. The frame is encoded
            # here -- on the thread that owns the camera -- and only the JPEG
            # bytes cross to the worker, so the vision call never blocks the
            # loop and never reaches back into OpenCV.
            if observer is not None and wants_to_look:
                if tracker.state is not EngagementState.ENGAGED:
                    _log("not engaged yet; look at the camera first.")
                elif observer.busy:
                    _log("still looking at the last thing you showed me.")
                elif session is not None and session.busy:
                    # A turn is mid-prepare and has already read the memory.
                    # Observing now would leave "did that answer use the new
                    # record?" up to which network call returned first.
                    _log("hold on -- I'm still thinking about what you said.")
                else:
                    try:
                        jpeg = encode_frame_jpeg(reading.frame)
                    except CameraError as exc:
                        _log(f"could not use that frame: {exc}", error=True)
                    else:
                        if observer.start(jpeg):
                            _log(f"looking at the desk ({len(jpeg)} byte frame)...")

            # Talking is only offered once the character has actually noticed
            # someone, so the spoken path is downstream of engagement rather
            # than a second, parallel way in. Starting a turn only hands work
            # to a worker thread -- this loop keeps running at DETECT_HZ
            # throughout the recording and the API calls.
            if session is not None and wants_to_talk:
                if tracker.state is not EngagementState.ENGAGED:
                    _log("not engaged yet; look at the camera first.")
                elif session.busy:
                    _log("still working on the last thing you said.")
                elif observer is not None and not observer.ready():
                    # The other half of the guard above: a look is in flight, so
                    # the answer would depend on whether the vision call beat
                    # the transcription. Wait a beat and the question is
                    # answered against a settled memory.
                    _log("hold on -- I'm still looking at the desk.")
                elif not session.start(epoch, time.monotonic()):
                    _log("still talking; wait for me to finish.")
                else:
                    _log("listening...")

            if session is not None:
                outcome = session.poll()
                if outcome is not None:
                    _handle_outcome(outcome, session, voice, sensor, tracker, epoch)

            # Pace the perception loop to DETECT_HZ so it does not spin a core
            # for information the camera cannot supply any faster. A behaviour
            # that just played already overran this budget, so the sleep
            # collapses to zero.
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    finally:
        sensor.close()
        if preview:
            cv2.destroyAllWindows()
        if client is not None and p.isConnected(client):
            p.disconnect(client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--no-preview", action="store_true", help="hide the camera preview window")
    parser.add_argument("--headless", action="store_true", help="run PyBullet without a window")
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="disable the spoken turn (camera-only demo; no OPENAI_API_KEY needed)",
    )
    args = parser.parse_args()

    voice_factory = None
    memory = SceneMemory()
    observer = None
    if not args.no_voice:
        # Check the credential before opening a camera or a GUI, so a missing
        # key is a one-line startup error rather than a surprise mid-demo.
        require_api_key()
        # One client for both the spoken turn and the observation, and one
        # memory that the observation writes and the spoken turn reads.
        brain = CharacterBrain()
        observer = ObservationSession(brain)
        voice_factory = lambda lamp: build_voice_turn(lamp, memory, brain)  # noqa: E731

    run(
        camera_index=args.camera_index,
        preview=not args.no_preview,
        headless=args.headless,
        voice_factory=voice_factory,
        memory=memory,
        observer=observer,
    )


if __name__ == "__main__":
    try:
        main()
    except MissingCredentialsError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
    except CharacterError as exc:
        # Building the shared client can fail before the loop starts (no SDK
        # installed, malformed key). Subclass order matters: the credential
        # case above is more specific and is caught first.
        print(f"language layer unavailable: {exc}", file=sys.stderr)
        sys.exit(1)
    except CameraError as exc:
        print(f"camera error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
