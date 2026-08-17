"""Continuous engagement + spoken recall demo for Hour 4.

Run:  python src/scene_memory_demo.py

Flow:
    camera attention -> engagement -> optional spoken turns
    current camera frame --press o--> visual observation -> local SceneMemory
    later microphone question -> STT -> character + retained facts -> TTS

The observation is an explicit demo step so it is obvious when vision happened.
Later recall never captures or sends another frame. Press ``o`` while ENGAGED
with one desk object clearly visible, then move or remove it and ask about it
with the existing push-to-talk control.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
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
from character import CharacterBrain, CharacterError, MissingCredentialsError, require_api_key  # noqa: E402
from engagement_demo import (  # noqa: E402
    AnyTalkKey,
    IDLE_SIM_STEP,
    PREVIEW_WINDOW,
    SimulatorTalkKey,
    TALK_KEYS,
    TerminalTalkKey,
    _draw_preview,
    _handle_outcome,
    _log,
)
from robot_controller import LampController, load_lamp  # noqa: E402
from scene_memory import MemoryAwareBrain, SceneMemory  # noqa: E402
from voice_turn import VoiceSession, VoiceTurn  # noqa: E402

OBSERVE_KEY = ord("o")


class SimulatorObserveKey:
    """Observation trigger from the PyBullet window."""

    def __init__(self, client_id: int):
        self.client_id = client_id

    def pressed(self) -> bool:
        try:
            events = p.getKeyboardEvents(physicsClientId=self.client_id)
        except Exception:
            return False
        return bool(events.get(OBSERVE_KEY, 0) & p.KEY_WAS_TRIGGERED)


class ObservationSession:
    """Run JPEG encoding + cloud vision off the PyBullet/camera control thread."""

    def __init__(self, brain: CharacterBrain, memory: SceneMemory):
        self.brain = brain
        self.memory = memory
        self._running = threading.Event()
        self._results: "queue.Queue[tuple[object | None, Exception | None]]" = queue.Queue(maxsize=1)

    @property
    def busy(self) -> bool:
        return self._running.is_set()

    def start(self, frame) -> bool:
        if self.busy or not self._results.empty():
            return False
        # Snapshot now. The camera loop may mutate/reuse its next frame, and
        # preview drawing adds overlays after this point.
        snapshot = frame.copy()
        self._running.set()
        threading.Thread(
            target=self._work,
            args=(snapshot,),
            name="scene-observation",
            daemon=True,
        ).start()
        return True

    def _work(self, frame) -> None:
        observation = None
        error = None
        try:
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise CharacterError("could not encode camera frame as JPEG")
            observation = self.brain.observe(encoded.tobytes())
        except Exception as exc:  # one failed observation must not kill the demo
            error = exc
        try:
            self._results.put((observation, error))
        finally:
            self._running.clear()

    def poll(self):
        try:
            observation, error = self._results.get_nowait()
        except queue.Empty:
            return None
        if error is None:
            self.memory.remember(observation)
        return observation, error


def build_system(lamp: LampController):
    """Share one CharacterBrain between vision and the existing voice pipeline."""
    brain = CharacterBrain()
    memory = SceneMemory()
    voice = VoiceTurn(
        brain=MemoryAwareBrain(brain, memory),
        recorder=MicrophoneRecorder(),
        player=SpeakerPlayer(),
        lamp=lamp,
    )
    return voice, ObservationSession(brain, memory), memory


def run(camera_index: int = 0, preview: bool = True, headless: bool = False) -> None:
    sensor = AttentionSensor(camera_index=camera_index).open()
    client = None
    try:
        client = p.connect(p.DIRECT if headless else p.GUI)
        if client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        if not headless:
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
        voice, observer, memory = build_system(lamp)
        voice_session = VoiceSession(voice)
        talk_key = AnyTalkKey(
            TerminalTalkKey(),
            None if headless else SimulatorTalkKey(client),
        )
        observe_key = None if headless else SimulatorObserveKey(client)
        epoch = 0

        lamp.neutral()
        _log(f"ready in {tracker.state.value}; look at the camera to engage.")
        print("      ENGAGED controls: Enter/space/t = talk; o = observe one visible desk object.")
        print("      For the clearest memory demo, press o, wait for 'remembered', then move the object")
        print("      and ask a spoken question about its earlier color/location/detail.")

        period = 1.0 / DETECT_HZ
        while True:
            tick = time.monotonic()
            reading = sensor.read()
            transition = tracker.update(reading.attending, tick)

            wants_to_talk = talk_key.pressed()
            wants_to_observe = observe_key.pressed() if observe_key is not None else False

            if preview:
                _draw_preview(reading, tracker.state)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                wants_to_talk = wants_to_talk or key in TALK_KEYS
                wants_to_observe = wants_to_observe or key == OBSERVE_KEY

            if transition is EngagementState.ENGAGED:
                epoch += 1
                _log("IDLE -> ENGAGED")
                lamp.engage()
                sensor.flush()
            elif transition is EngagementState.IDLE:
                _log("ENGAGED -> IDLE")
                lamp.neutral()
                sensor.flush()
            else:
                lamp.step(IDLE_SIM_STEP)

            if wants_to_observe:
                if tracker.state is not EngagementState.ENGAGED:
                    _log("not engaged yet; look at the camera before observing.")
                elif observer.busy:
                    _log("still observing the previous frame.")
                elif observer.start(reading.frame):
                    _log("observing one desk object from this frame...")
                else:
                    _log("previous observation result has not been collected yet.")

            observed = observer.poll()
            if observed is not None:
                observation, error = observed
                if error is not None:
                    _log(f"observation failed: {error}", error=True)
                else:
                    _log(f"remembered: {observation.summary()}")
                    _log("the image is no longer needed; later recall uses only those stored facts.")

            if wants_to_talk:
                if tracker.state is not EngagementState.ENGAGED:
                    _log("not engaged yet; look at the camera first.")
                elif voice_session.busy:
                    _log("still working on the last thing you said.")
                elif not voice_session.start(epoch, time.monotonic()):
                    _log("still talking; wait for me to finish.")
                else:
                    if memory.observation is None:
                        _log("listening... (no scene object remembered yet)")
                    else:
                        _log("listening... (retained scene memory available)")

            outcome = voice_session.poll()
            if outcome is not None:
                _handle_outcome(outcome, voice_session, voice, sensor, tracker, epoch)

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
    parser.add_argument("--no-preview", action="store_true", help="hide camera preview")
    parser.add_argument("--headless", action="store_true", help="run PyBullet without a window")
    args = parser.parse_args()
    require_api_key()
    run(camera_index=args.camera_index, preview=not args.no_preview, headless=args.headless)


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
