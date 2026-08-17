"""Scene-memory tests: fake camera frame, fake OpenAI client, no hardware.

Nothing here opens a device, reads a credential or makes a paid API call. The
whole path

    frame -> encode -> [vision call] -> validate -> local memory
    ...later...
    question -> [character call] -> reply

runs in-process against stand-ins.

The load-bearing property is the *last* one in this file: the recall path must
not send an image. Everything else can be re-derived from the code; that one is
the difference between remembering and quietly looking again.

    python -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import attention  # noqa: E402
import character  # noqa: E402
import engagement_demo  # noqa: E402
import scene_memory  # noqa: E402
from character import CharacterBrain, CharacterError  # noqa: E402
from scene_memory import (  # noqa: E402
    LOCATIONS,
    ObservationError,
    ObservationSession,
    SceneMemory,
    SceneObservation,
    parse_observation,
)
from voice_turn import BEHAVIORS, TurnError  # noqa: E402

from test_voice_turn import (  # noqa: E402
    FakeLamp,
    ScriptedSensor,
    _Endpoint,
    _Response,
    build_turn,
    fake_client,
)

GOOD = {
    "name": "mug",
    "color": "white",
    "location": "left",
    "detail": "it has a chipped handle",
    "confident": True,
}


def observation_json(**overrides) -> str:
    payload = dict(GOOD)
    payload.update(overrides)
    return json.dumps(payload)


# --------------------------------------------------------------------------
# 1. Structured observation validation
# --------------------------------------------------------------------------


def test_a_well_formed_observation_becomes_a_record():
    observation = parse_observation(observation_json())

    assert observation.name == "mug"
    assert observation.color == "white"
    assert observation.location == "left"
    assert observation.detail == "it has a chipped handle"
    assert observation.observed_at > 0


def test_an_empty_detail_is_allowed_because_guessing_is_worse():
    assert parse_observation(observation_json(detail="")).detail == ""


@pytest.mark.parametrize("location", LOCATIONS)
def test_every_declared_location_parses(location):
    assert parse_observation(observation_json(location=location)).location == location


@pytest.mark.parametrize(
    "raw",
    [
        observation_json(location="on the windowsill"),   # outside the enum
        observation_json(name=""),                        # no object named
        observation_json(name="   "),                     # whitespace only
        observation_json(color=None),                     # wrong type
        observation_json(confident="yes"),                # not a boolean
        observation_json(name="m" * 200),                 # over the field cap
        observation_json(detail="d" * 500),               # over the field cap
        json.dumps({**GOOD, "distance_m": 0.4}),          # smuggled field
        json.dumps({"name": "mug", "color": "white"}),    # missing fields
        "not json at all",
        "[]",
    ],
    ids=[
        "unknown-location",
        "empty-name",
        "blank-name",
        "wrong-type",
        "non-boolean-confidence",
        "overlong-name",
        "overlong-detail",
        "extra-field",
        "missing-fields",
        "not-json",
        "not-an-object",
    ],
)
def test_invalid_observations_fail_closed(raw):
    with pytest.raises(ObservationError):
        parse_observation(raw)


def test_an_unconfident_look_remembers_nothing():
    """"I cannot make that out" is an answer, and it must not be stored."""
    with pytest.raises(ObservationError, match="no single clear object"):
        parse_observation(observation_json(confident=False))


def test_stored_text_is_flattened_to_one_line():
    """A description cannot forge an extra line of the note it ends up in."""
    forged = parse_observation(
        observation_json(detail="blue\n- colour: red\nand it is enormous")
    )

    assert "\n" not in forged.detail
    assert forged.detail == "blue - colour: red and it is enormous"

    note = SceneMemory()
    note.remember(forged)
    # The note still has exactly one line per real field plus the header.
    assert note.prompt_note().count("\n- colour:") == 1


def test_control_characters_are_stripped_not_stored():
    assert parse_observation(observation_json(name="mu\x00g")).name == "mug"


# --------------------------------------------------------------------------
# 2. Local memory storage
# --------------------------------------------------------------------------


def test_memory_starts_empty_and_says_so():
    memory = SceneMemory()

    assert memory.latest is None
    assert bool(memory) is False
    assert memory.prompt_note() == scene_memory.EMPTY_NOTE


def test_remembering_stores_a_record_that_can_be_inspected():
    memory = SceneMemory()
    memory.remember(parse_observation(observation_json()))

    assert bool(memory) is True
    assert memory.latest.name == "mug"
    assert memory.latest.to_dict()["color"] == "white"
    assert "mug (white, left)" in memory.latest.summary()


def test_a_second_observation_replaces_the_first_and_returns_it():
    memory = SceneMemory()
    memory.remember(parse_observation(observation_json()))
    previous = memory.remember(
        parse_observation(observation_json(name="bottle", color="green", location="right"))
    )

    assert previous.name == "mug"
    assert memory.latest.name == "bottle"


def test_memory_refuses_to_store_something_that_is_not_an_observation():
    with pytest.raises(TypeError):
        SceneMemory().remember({"name": "mug"})


def test_the_note_carries_every_retained_fact():
    memory = SceneMemory()
    memory.remember(parse_observation(observation_json()))
    note = memory.prompt_note()

    assert scene_memory.NOTE_HEADER in note
    assert "mug" in note
    assert "white" in note
    assert "on your left" in note
    assert "chipped handle" in note
    assert "seconds ago" in note


def test_the_note_reports_how_long_ago_the_look_happened():
    memory = SceneMemory()
    observation = parse_observation(observation_json())
    memory.remember(observation)

    note = memory.prompt_note(now=observation.observed_monotonic + 42.0)

    assert "about 42 seconds ago" in note


def test_forgetting_empties_the_record():
    memory = SceneMemory()
    memory.remember(parse_observation(observation_json()))
    memory.forget()

    assert memory.latest is None
    assert memory.prompt_note() == ""


# --------------------------------------------------------------------------
# 3. The observation call itself
# --------------------------------------------------------------------------


def vision_client(raw: str = None, error: Exception = None):
    """A client whose responses endpoint answers the observation request."""
    client = fake_client()
    client.responses = _Endpoint(
        result=_Response(raw if raw is not None else observation_json()), error=error
    )
    return client


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_observe_sends_one_image_and_returns_a_validated_record():
    client = vision_client()
    brain = CharacterBrain(client=client)

    observation = brain.observe(b"\xff\xd8\xff-pretend-jpeg")

    assert observation.name == "mug"
    request = client.responses.calls[0]
    content = request["input"][0]["content"]
    kinds = [part["type"] for part in content]
    assert kinds == ["input_text", "input_image"]
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert request["model"] == character.VISION_MODEL
    assert 0 < request["timeout"] <= 30


def test_observe_requests_the_schema_but_still_validates_locally():
    client = vision_client()
    CharacterBrain(client=client).observe(b"jpeg")
    fmt = client.responses.calls[0]["text"]["format"]

    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["properties"]["location"]["enum"] == list(LOCATIONS)
    # ...and the local validator rejects what the schema would have caught.
    with pytest.raises(ObservationError):
        parse_observation(observation_json(location="behind you"))


def test_observe_with_no_frame_fails_before_any_call():
    client = vision_client()

    with pytest.raises(CharacterError, match="no camera frame"):
        CharacterBrain(client=client).observe(b"")

    assert client.responses.calls == []


def test_a_failed_vision_call_is_a_character_error():
    brain = CharacterBrain(client=vision_client(error=RuntimeError("gateway timeout")))

    with pytest.raises(CharacterError, match="looking at the scene failed"):
        brain.observe(b"jpeg")


def test_an_empty_vision_response_is_a_character_error():
    brain = CharacterBrain(client=vision_client(raw="   "))

    with pytest.raises(CharacterError, match="observation response was empty"):
        brain.observe(b"jpeg")


SECRET = "sk-test-DO-NOT-LEAK-abcdef123456"


def test_the_key_never_appears_in_an_observation_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    leaky = RuntimeError(f"401 invalid_api_key: incorrect API key provided: {SECRET}")
    brain = CharacterBrain(client=vision_client(error=leaky))

    with pytest.raises(CharacterError) as excinfo:
        brain.observe(b"jpeg")

    assert SECRET not in str(excinfo.value)
    assert "<redacted>" in str(excinfo.value)


# --------------------------------------------------------------------------
# 4. The frame that gets uploaded is bounded
# --------------------------------------------------------------------------


def test_a_frame_encodes_to_bounded_jpeg_bytes():
    payload = attention.encode_frame_jpeg(FRAME)

    assert payload[:2] == b"\xff\xd8"  # JPEG SOI
    assert 0 < len(payload) <= attention.OBSERVE_MAX_BYTES


def test_an_oversized_frame_is_scaled_down_before_upload():
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)

    small = attention.encode_frame_jpeg(big, max_width=320)

    assert len(small) < len(attention.encode_frame_jpeg(big))


def test_encoding_refuses_a_missing_frame():
    with pytest.raises(attention.CameraError, match="no camera frame"):
        attention.encode_frame_jpeg(None)


def test_an_implausibly_large_encoding_is_refused(monkeypatch):
    monkeypatch.setattr(attention, "OBSERVE_MAX_BYTES", 8)

    with pytest.raises(attention.CameraError, match="upload limit"):
        attention.encode_frame_jpeg(FRAME)


# --------------------------------------------------------------------------
# 5. Later reasoning receives the retained facts
# --------------------------------------------------------------------------


def _voice_turn_with_memory(memory, **client_kwargs):
    turn, lamp, recorder, player = build_turn(**client_kwargs)
    turn.memory = memory
    return turn, lamp, player


def test_a_later_question_is_answered_with_the_retained_facts():
    memory = SceneMemory()
    memory.remember(parse_observation(observation_json()))
    turn, _lamp, _player = _voice_turn_with_memory(
        memory,
        transcript="what colour was the mug?",
        reply="It was white, over on my left.",
    )

    result = turn.run()
    sent = turn.brain._client.responses.calls[0]["input"]

    assert result.reply == "It was white, over on my left."
    # The note reached the model, labelled, alongside the question.
    assert scene_memory.NOTE_HEADER in sent
    assert "mug" in sent and "white" in sent and "on your left" in sent
    assert character.SPOKEN_HEADER in sent
    assert "what colour was the mug?" in sent


def test_a_turn_with_an_empty_memory_sends_the_transcript_unchanged():
    """No note, no block: an ordinary turn looks exactly as it did in Hour 3."""
    turn, *_ = build_turn(transcript="hello there")

    turn.run()

    assert turn.brain._client.responses.calls[0]["input"] == "hello there"


def test_the_prompt_tells_the_character_the_note_is_a_memory_not_a_view():
    prompt = character.SYSTEM_PROMPT.lower()

    assert "scene memory" in prompt
    assert "past tense" in prompt
    # It must decline rather than invent when the note does not cover the ask.
    assert "never guess" in prompt


# --------------------------------------------------------------------------
# 6. Recall does NOT send a new image
# --------------------------------------------------------------------------


class _NoLookingBrain(CharacterBrain):
    """A brain that fails loudly if a spoken turn tries to observe."""

    def observe(self, jpeg_bytes):  # pragma: no cover - the assertion is the point
        raise AssertionError("the recall path must never take a new picture")


def test_answering_a_question_never_submits_a_frame():
    memory = SceneMemory()
    memory.remember(parse_observation(observation_json()))
    turn, _lamp, _recorder, _player = build_turn(transcript="where was it?")
    turn.memory = memory
    turn.brain = _NoLookingBrain(client=turn.brain._client)

    turn.run()

    request = turn.brain._client.responses.calls[0]
    # The whole request body is text: no image part, and no base64 payload
    # smuggled into the input string.
    assert isinstance(request["input"], str)
    assert "input_image" not in request["input"]
    assert "data:image" not in request["input"]


def test_the_voice_turn_has_no_route_to_the_camera():
    """A structural check: the recall path holds no camera and no frame."""
    turn, *_ = build_turn()

    assert not hasattr(turn, "sensor")
    assert not hasattr(turn, "camera")
    # Its only new collaborator is the memory, which is plain local data.
    assert isinstance(turn.memory, SceneMemory)


def test_the_memory_survives_the_object_being_taken_away():
    """The point of the demonstration, in one test.

    The object is observed once; the world then changes (the camera would now
    show an empty desk); the question is still answered from the record.
    """
    memory = SceneMemory()
    memory.remember(parse_observation(observation_json()))

    # ...the mug is removed. Nothing in the recall path consults the camera,
    # so there is nothing to update.
    note = memory.prompt_note()

    assert "mug" in note and "white" in note


# --------------------------------------------------------------------------
# 7. The observation runs off the control thread
# --------------------------------------------------------------------------


class _RecordingBrain:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.frames = []

    def observe(self, jpeg):
        self.frames.append(jpeg)
        if self.error is not None:
            raise self.error
        return self.result


def _inline_session(brain):
    return ObservationSession(brain, spawn=lambda work: work())


OBSERVATION = SceneObservation(
    name="mug", color="white", location="left", detail="", observed_at=1.0
)


def test_session_reports_a_finished_observation():
    session = _inline_session(_RecordingBrain(result=OBSERVATION))

    assert session.start(b"jpeg") is True
    outcome = session.poll()

    assert outcome.error is None
    # Restamped with the capture time, so it is a copy of the same facts.
    assert outcome.observation.summary() == OBSERVATION.summary()
    assert session.busy is False


def test_session_turns_a_failure_into_an_outcome_not_an_exception():
    session = _inline_session(_RecordingBrain(error=ObservationError("no clear object")))
    session.start(b"jpeg")
    outcome = session.poll()

    assert outcome.observation is None
    assert isinstance(outcome.error, ObservationError)


def test_session_survives_a_worker_that_raises_something_unexpected():
    session = _inline_session(_RecordingBrain(error=KeyboardInterrupt("interrupted")))
    session.start(b"jpeg")

    assert isinstance(session.poll().error, KeyboardInterrupt)
    assert session.busy is False


def test_session_refuses_a_second_look_while_one_is_undrained():
    brain = _RecordingBrain(result=OBSERVATION)
    session = _inline_session(brain)

    assert session.start(b"one") is True
    assert session.start(b"two") is False  # result not yet polled
    session.poll()
    assert session.start(b"three") is True
    assert brain.frames == [b"one", b"three"]


def test_the_worker_receives_bytes_and_nothing_else():
    """Everything OpenCV-shaped stays on the control thread."""
    brain = _RecordingBrain(result=OBSERVATION)
    _inline_session(brain).start(attention.encode_frame_jpeg(FRAME))

    assert isinstance(brain.frames[0], bytes)


def _await_observation(session, tries: int = 500):
    idle = threading.Event()
    for _ in range(tries):
        outcome = session.poll()
        if outcome is not None:
            return outcome
        idle.wait(0.01)
    raise AssertionError("the observation worker never published a result")


def test_a_real_worker_thread_publishes_its_result():
    brain = _RecordingBrain(result=OBSERVATION)
    session = ObservationSession(brain)
    session.start(b"jpeg")

    outcome = _await_observation(session)

    assert outcome.observation.name == OBSERVATION.name


# --------------------------------------------------------------------------
# 8. Wiring into the engagement loop
# --------------------------------------------------------------------------


class ScriptedKey:
    """Reports a press on the given loop iterations."""

    def __init__(self, *frames: int):
        self.frames = set(frames)
        self.index = -1

    def pressed(self) -> bool:
        self.index += 1
        return self.index in self.frames


class _FrameSensor(ScriptedSensor):
    """A scripted sensor that also hands back a camera frame.

    *faces* is non-empty in the preview test, because face boxes are one of the
    things the preview overlay draws onto the frame.
    """

    def __init__(self, script, frame=None, faces=()):
        super().__init__(script)
        self.frame = FRAME if frame is None else frame
        self.faces = faces

    def read(self):
        reading = super().read()
        return type(reading)(
            attending=reading.attending, faces=self.faces, frame=self.frame
        )


def _run_observing_demo(script, observe_key, brain, memory=None):
    memory = memory if memory is not None else SceneMemory()
    engagement_demo.run(
        preview=False,
        headless=True,
        sensor=_FrameSensor(script),
        memory=memory,
        observer=_inline_session(brain),
        observe_key=observe_key,
    )
    return memory


def test_an_observation_stores_and_logs_what_it_remembered(capsys):
    brain = _RecordingBrain(result=OBSERVATION)
    memory = _run_observing_demo([(True, 4.0)], ScriptedKey(40), brain)
    out = capsys.readouterr().out

    assert memory.latest.name == "mug"
    assert "looking at the desk" in out
    assert "remembered: mug (white, left)" in out
    assert "scene memory now holds:" in out


def test_looking_is_refused_until_the_character_is_engaged(capsys):
    brain = _RecordingBrain(result=OBSERVATION)
    memory = _run_observing_demo([(True, 4.0)], ScriptedKey(0), brain)
    out = capsys.readouterr().out

    assert brain.frames == []
    assert memory.latest is None
    assert "not engaged yet" in out


def test_a_refused_observation_leaves_the_previous_memory_alone(capsys):
    memory = SceneMemory()
    memory.remember(OBSERVATION)
    brain = _RecordingBrain(error=ObservationError("no single clear object"))

    _run_observing_demo([(True, 4.0)], ScriptedKey(40), brain, memory=memory)
    captured = capsys.readouterr()

    assert memory.latest is OBSERVATION
    assert "remembered nothing" in captured.err


def test_the_camera_only_demo_never_looks_at_the_desk(capsys):
    engagement_demo.run(preview=False, headless=True, sensor=ScriptedSensor([(True, 2.0)]))

    assert "press 'o'" not in capsys.readouterr().out


def test_the_simulator_observe_key_is_its_own_key(monkeypatch):
    events = {}
    monkeypatch.setattr(
        engagement_demo.p, "getKeyboardEvents", lambda physicsClientId: events
    )
    observe = engagement_demo.SimulatorObserveKey(client_id=0)
    talk = engagement_demo.SimulatorTalkKey(client_id=0)

    events[ord("o")] = engagement_demo.p.KEY_WAS_TRIGGERED
    assert observe.pressed() is True
    assert talk.pressed() is False  # 'o' must not start a recording

    events.clear()
    events[ord("t")] = engagement_demo.p.KEY_WAS_TRIGGERED
    assert observe.pressed() is False


# --------------------------------------------------------------------------
# 9. The robot contract is untouched by any of this
# --------------------------------------------------------------------------


def test_scene_memory_did_not_widen_the_behavior_allowlist():
    assert character.ALLOWED_BEHAVIORS == ("none", "nod", "engage")
    assert set(BEHAVIORS) == {"none", "nod", "engage"}
    assert character.RESPONSE_SCHEMA["properties"]["behavior"]["enum"] == [
        "none",
        "nod",
        "engage",
    ]


def test_the_observation_contract_cannot_request_motion():
    """The vision call has no field that could carry a gesture or a joint."""
    fields = set(character.OBSERVATION_SCHEMA["properties"])

    assert fields == {"name", "color", "location", "detail", "confident"}
    assert character.OBSERVATION_SCHEMA["additionalProperties"] is False


def test_an_observation_never_moves_the_robot():
    lamp = FakeLamp()
    brain = _RecordingBrain(result=OBSERVATION)
    session = _inline_session(brain)
    session.start(b"jpeg")
    session.poll()

    assert lamp.calls == []


def test_scene_memory_does_not_import_the_robot_or_the_network():
    """The memory is plain local data; keep it that way."""
    source = (Path(__file__).resolve().parent.parent / "src" / "scene_memory.py").read_text()

    assert "import pybullet" not in source
    assert "openai" not in source
    assert "import cv2" not in source


def test_a_memory_read_failure_is_a_failed_turn_not_a_moved_robot():
    """Belt and braces: the note is read inside prepare(), before the gesture."""

    class Exploding(SceneMemory):
        def prompt_note(self, now=None):
            raise RuntimeError("memory unreadable")

    turn, lamp, _recorder, player = build_turn(behavior="nod")
    turn.memory = Exploding()

    with pytest.raises(TurnError, match="scene memory failed"):
        turn.run()

    assert lamp.calls == []
    assert player.played == []


# --------------------------------------------------------------------------
# 10. Review regressions
#
# Each test here corresponds to one defect found reviewing the first cut of
# Hour 4. They are grouped rather than scattered so the reason they exist stays
# attached to them.
# --------------------------------------------------------------------------


class _DrainingWindow:
    """Models ``p.getKeyboardEvents``: each event is reported to one caller.

    This is the behaviour that made the bug invisible in the earlier tests,
    which used a plain dict that every caller could read forever.
    """

    def __init__(self):
        self.pending: dict = {}
        self.calls = 0

    def __call__(self, physicsClientId):
        self.calls += 1
        events, self.pending = self.pending, {}
        return events


TRIGGERED = engagement_demo.p.KEY_WAS_TRIGGERED


def test_two_independent_pollers_lose_the_observe_press(monkeypatch):
    """The shape of the defect, kept as the reason the keys are shared.

    Two self-polling keys means two ``getKeyboardEvents`` calls per tick, and
    the first one drains the event the second was waiting for.
    """
    window = _DrainingWindow()
    monkeypatch.setattr(engagement_demo.p, "getKeyboardEvents", window)
    talk = engagement_demo.SimulatorTalkKey(client_id=0)
    observe = engagement_demo.SimulatorObserveKey(client_id=0)

    window.pending = {ord("o"): TRIGGERED}
    talk.pressed()  # the talk poll runs first, exactly as it does in the loop

    assert observe.pressed() is False
    assert window.calls == 2


def test_one_poll_per_tick_lets_both_keys_see_the_same_events(monkeypatch):
    """The fix: build_simulator_keys + one poll, in the loop's own order."""
    window = _DrainingWindow()
    monkeypatch.setattr(engagement_demo.p, "getKeyboardEvents", window)
    keyboard, talk, observe = engagement_demo.build_simulator_keys(0)

    window.pending = {ord("o"): TRIGGERED}
    keyboard.poll()                      # <- what run() does once per tick
    wants_to_talk = talk.pressed()       # <- then this
    wants_to_look = observe.pressed()    # <- then this

    assert window.calls == 1
    assert wants_to_talk is False
    assert wants_to_look is True


def test_push_to_talk_from_the_window_still_works_after_the_fix(monkeypatch):
    window = _DrainingWindow()
    monkeypatch.setattr(engagement_demo.p, "getKeyboardEvents", window)
    keyboard, talk, observe = engagement_demo.build_simulator_keys(0)

    for key in engagement_demo.SimulatorTalkKey.KEYS:
        window.pending = {key: TRIGGERED}
        keyboard.poll()

        assert talk.pressed() is True, f"key {key} no longer starts a turn"
        assert observe.pressed() is False


def test_a_held_key_is_not_a_new_request_after_the_fix(monkeypatch):
    window = _DrainingWindow()
    monkeypatch.setattr(engagement_demo.p, "getKeyboardEvents", window)
    keyboard, _talk, observe = engagement_demo.build_simulator_keys(0)

    window.pending = {ord("o"): engagement_demo.p.KEY_IS_DOWN}
    keyboard.poll()

    assert observe.pressed() is False


def test_the_window_keys_share_one_keyboard_and_do_not_self_poll():
    keyboard, talk, observe = engagement_demo.build_simulator_keys(7)

    assert talk.keyboard is keyboard and observe.keyboard is keyboard
    assert talk.self_polling is False and observe.self_polling is False
    assert talk.client_id == 7


def test_a_simulator_key_survives_a_window_that_went_away(monkeypatch):
    def no_gui(**_kwargs):
        raise Exception("Not connected to a GUI server")

    monkeypatch.setattr(engagement_demo.p, "getKeyboardEvents", no_gui)
    keyboard, talk, observe = engagement_demo.build_simulator_keys(0)
    keyboard.poll()

    assert talk.pressed() is False
    assert observe.pressed() is False


# -- the preview overlay must not reach the vision call ---------------------


def test_the_preview_overlay_never_reaches_the_vision_call(monkeypatch):
    """The preview draws in place; the same frame is what an observation sends.

    Before the fix the vision model received our own face boxes and status
    text burned into the image. The expected bytes are computed *before* the
    run, so mutating the original could not make this pass by accident.
    """
    shown = []
    monkeypatch.setattr(engagement_demo.cv2, "imshow", lambda _name, frame: shown.append(frame))
    monkeypatch.setattr(engagement_demo.cv2, "waitKey", lambda _delay: 255)
    monkeypatch.setattr(engagement_demo.cv2, "destroyAllWindows", lambda: None)

    clean = np.full((480, 640, 3), 40, dtype=np.uint8)
    expected_jpeg = attention.encode_frame_jpeg(clean.copy())
    pristine = clean.copy()

    brain = _RecordingBrain(result=OBSERVATION)
    engagement_demo.run(
        preview=True,
        headless=True,
        sensor=_FrameSensor([(True, 4.0)], frame=clean, faces=((10, 10, 120, 120),)),
        memory=SceneMemory(),
        observer=_inline_session(brain),
        observe_key=ScriptedKey(40),
    )

    assert brain.frames == [expected_jpeg]
    assert np.array_equal(clean, pristine), "the preview mutated the camera frame"
    # The preview did run, and drew on something other than the live frame.
    assert shown and shown[0] is not clean
    assert not np.array_equal(shown[0], pristine)


def test_drawing_the_preview_leaves_the_reading_untouched(monkeypatch):
    monkeypatch.setattr(engagement_demo.cv2, "imshow", lambda _name, _frame: None)
    frame = np.full((240, 320, 3), 90, dtype=np.uint8)
    pristine = frame.copy()
    reading = attention.AttentionReading(
        attending=True, faces=((5, 5, 60, 60),), frame=frame
    )

    engagement_demo._draw_preview(reading, engagement_demo.EngagementState.ENGAGED)

    assert np.array_equal(frame, pristine)


# -- observation and spoken turns must not race -----------------------------


class _StuckSpawn:
    """A spawn that never runs the work, so the session stays busy forever."""

    def __call__(self, _work):
        return None


def _run_interaction_demo(script, talk_key, observe_key, turn, observer, memory=None):
    memory = memory if memory is not None else SceneMemory()
    engagement_demo.run(
        preview=False,
        headless=True,
        sensor=_FrameSensor(script),
        voice_factory=(lambda _lamp: turn) if turn is not None else None,
        talk_key=talk_key,
        memory=memory,
        observer=observer,
        observe_key=observe_key,
    )
    return memory


def test_a_spoken_turn_waits_for_an_observation_in_flight(capsys):
    """Otherwise the answer depends on which network call returned first."""
    from test_voice_turn import RecordingTurn

    turn = RecordingTurn()
    observer = ObservationSession(_RecordingBrain(result=OBSERVATION), spawn=_StuckSpawn())

    _run_interaction_demo(
        [(True, 5.0)], ScriptedKey(45), ScriptedKey(40), turn, observer
    )
    out = capsys.readouterr().out

    assert observer.busy is True
    assert turn.prepares == 0, "the turn started against an unsettled memory"
    assert "still looking at the desk" in out


def test_an_observation_waits_for_a_turn_that_is_being_prepared(capsys):
    """The inverse: the turn has already read the memory, so do not change it."""
    from test_voice_turn import RecordingTurn

    gate = threading.Event()
    turn = RecordingTurn(gate=gate)
    brain = _RecordingBrain(result=OBSERVATION)
    sensor = _FrameSensor(
        [(True, 8.0)], faces=()
    )
    sensor._on_frame = lambda i: gate.set() if i == 100 else None

    engagement_demo.run(
        preview=False,
        headless=True,
        sensor=sensor,
        voice_factory=lambda _lamp: turn,
        talk_key=ScriptedKey(30),
        memory=SceneMemory(),
        observer=_inline_session(brain),
        observe_key=ScriptedKey(40),
    )
    out = capsys.readouterr().out

    assert turn.prepares == 1
    assert brain.frames == [], "a look happened while a turn was mid-prepare"
    assert "still thinking about what you said" in out


def test_a_turn_started_after_an_observation_settles_sees_the_new_record():
    """The point of both guards: the question meets a settled memory."""
    from test_voice_turn import RecordingTurn

    memory = SceneMemory()
    seen: list = []

    class NotingTurn(RecordingTurn):
        def prepare(self):
            seen.append(memory.latest)
            return super().prepare()

    turn = NotingTurn()
    _run_interaction_demo(
        [(True, 6.0)],
        ScriptedKey(60),
        ScriptedKey(40),
        turn,
        _inline_session(_RecordingBrain(result=OBSERVATION)),
        memory=memory,
    )

    assert len(seen) == 1
    assert seen[0] is not None and seen[0].name == "mug"


# -- a finished observation is banked before the next press is judged -------


class _DeferredSpawn:
    """Holds the worker until a test releases it, modelling a real thread.

    An inline worker finishes inside ``start`` and is drained later in the same
    tick, which hides this defect entirely. A real vision call finishes
    *between* ticks, leaving the result queued while the next tick decides what
    to do with a fresh press -- which is the state that used to refuse it, and
    refuse it silently.
    """

    def __init__(self):
        self.work = None

    def __call__(self, work):
        self.work = work

    def release(self) -> None:
        work, self.work = self.work, None
        if work is not None:
            work()


def test_a_second_look_is_not_dropped_while_the_first_result_waits(capsys):
    """The completed-but-unprocessed observation must be banked first."""
    spawn = _DeferredSpawn()
    brain = _RecordingBrain(result=OBSERVATION)
    observer = ObservationSession(brain, spawn=spawn)

    # The first look is released at the top of frame 45, so its result is
    # already queued when the second press is judged later in that same tick.
    # ...and the second look is released at frame 50, so both actually run.
    sensor = _FrameSensor([(True, 5.0)])
    sensor._on_frame = lambda i: spawn.release() if i in (45, 50) else None

    memory = SceneMemory()
    engagement_demo.run(
        preview=False,
        headless=True,
        sensor=sensor,
        memory=memory,
        observer=observer,
        observe_key=ScriptedKey(40, 45),
    )
    out = capsys.readouterr().out

    assert len(brain.frames) == 2, "the second press was dropped"
    assert out.count("remembered: mug (white, left)") >= 1
    assert memory.latest is not None


def test_a_pending_result_is_banked_before_a_new_look_is_judged():
    """The ordering itself: nothing is started while a result is uncollected."""
    brain = _RecordingBrain(result=OBSERVATION)
    session = _inline_session(brain)
    session.start(b"one")

    # Undrained: the session correctly refuses, which is why the loop drains
    # first rather than asking in this state.
    assert session.ready() is False
    assert session.start(b"two") is False
    assert session.poll() is not None
    assert session.start(b"two") is True


# -- the record is dated from the frame, not from the response --------------


def test_the_observation_is_stamped_when_the_frame_was_captured():
    """Memory age must include the vision call's own latency."""

    class SlowBrain(_RecordingBrain):
        def observe(self, jpeg):
            time.sleep(0.2)
            return super().observe(jpeg)

    session = ObservationSession(SlowBrain(result=OBSERVATION))
    started = time.monotonic()
    session.start(b"jpeg")
    outcome = _await_observation(session)
    finished = time.monotonic()

    assert finished - started >= 0.1, "the fake call did not actually take time"
    # Stamped at capture, so it is close to `started` and clearly not `finished`.
    assert outcome.observation.observed_monotonic - started < 0.05
    assert outcome.observation.observed_at > 0


def test_the_reported_age_counts_from_the_look_not_the_answer():
    observation = OBSERVATION.at(observed_at=1.0, observed_monotonic=100.0)
    memory = SceneMemory()
    memory.remember(observation)

    assert "about 12 seconds ago" in memory.prompt_note(now=112.0)


def test_restamping_keeps_every_observed_fact():
    stamped = parse_observation(observation_json()).at(5.0, 6.0)

    assert (stamped.name, stamped.color, stamped.location) == ("mug", "white", "left")
    assert stamped.detail == "it has a chipped handle"
    assert (stamped.observed_at, stamped.observed_monotonic) == (5.0, 6.0)
