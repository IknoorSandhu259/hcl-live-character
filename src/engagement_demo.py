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

This file is the *wiring* only, and is deliberately thin:

    attention.AttentionSensor      camera -> "is a face facing us?"  (perception)
    attention.EngagementTracker    noisy booleans -> IDLE / ENGAGED  (policy)
    audio_io                       microphone / speaker              (audio)
    character.CharacterBrain       transcript -> {reply, behavior}   (language)
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
import sys
import time
from pathlib import Path

import cv2
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention import (  # noqa: E402
    DETECT_HZ,
    AttentionSensor,
    CameraError,
    EngagementState,
    EngagementTracker,
)
from audio_io import MicrophoneRecorder, SpeakerPlayer  # noqa: E402
from character import CharacterBrain, MissingCredentialsError, require_api_key  # noqa: E402
from robot_controller import LampController, load_lamp  # noqa: E402
from voice_turn import TurnError, VoiceSession, VoiceTurn  # noqa: E402

#: Simulation time advanced per perception tick while no behaviour is playing.
#: The lamp is holding a pose here, so this only has to keep the position
#: controller ticking over; the frame grab dominates the loop period anyway.
IDLE_SIM_STEP = 1.0 / 60.0

PREVIEW_WINDOW = "lamp attention"

#: Push-to-talk keys accepted by the preview window: 't', Enter, keypad Enter.
TALK_KEYS = (ord("t"), 13, 10)


def _draw_preview(reading, state: EngagementState) -> None:
    """Show what the perception layer actually sees. Debug aid, not a UI."""
    frame = reading.frame
    if frame is None:
        return
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
    """

    def __init__(self) -> None:
        self.available = sys.stdin is not None and sys.stdin.isatty()

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
        while select.select([sys.stdin], [], [], 0)[0]:
            if not sys.stdin.readline():
                return hit  # stdin closed
            hit = True
        return hit


def build_voice_turn(lamp: LampController) -> VoiceTurn:
    """Default wiring of the spoken path onto a live lamp."""
    return VoiceTurn(
        brain=CharacterBrain(),
        recorder=MicrophoneRecorder(),
        player=SpeakerPlayer(),
        lamp=lamp,
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


def run(
    camera_index: int = 0,
    preview: bool = True,
    headless: bool = False,
    sensor=None,
    voice_factory=None,
    talk_key=None,
) -> None:
    """Drive the lamp from the camera until interrupted.

    *sensor* exists so the loop can be exercised headlessly against a scripted
    fake reader; leave it None to use the real webcam. *voice_factory* is
    called with the LampController to build the spoken path; leave it None for
    the camera-only demo. *talk_key* answers "does the user want to talk now?".
    """
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
        if voice is not None and talk_key is None:
            talk_key = TerminalTalkKey()
        #: Bumped on every IDLE -> ENGAGED transition. A turn carries the value
        #: it started under, which is how a result that outlived its
        #: conversation is recognised and dropped.
        epoch = 0

        lamp.neutral()
        print(f"[{time.strftime('%H:%M:%S')}] ready in {tracker.state.value}; "
              "look at the camera to engage. Ctrl-C (or 'q' in the preview) to quit.")
        if voice is not None:
            print("      while ENGAGED, press Enter (or 't' in the preview) to talk.")

        period = 1.0 / DETECT_HZ
        while True:
            tick = time.monotonic()
            try:
                reading = sensor.read()
            except StopIteration:
                break  # a scripted fake sensor ran out of frames
            transition = tracker.update(reading.attending, tick)

            wants_to_talk = talk_key.pressed() if talk_key is not None else False
            if preview:
                _draw_preview(reading, tracker.state)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                wants_to_talk = wants_to_talk or key in TALK_KEYS

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
    if not args.no_voice:
        # Check the credential before opening a camera or a GUI, so a missing
        # key is a one-line startup error rather than a surprise mid-demo.
        require_api_key()
        voice_factory = build_voice_turn

    run(
        camera_index=args.camera_index,
        preview=not args.no_preview,
        headless=args.headless,
        voice_factory=voice_factory,
    )


if __name__ == "__main__":
    try:
        main()
    except MissingCredentialsError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
    except CameraError as exc:
        print(f"camera error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
