"""Goal-action tests: fake camera frames, fake OpenAI client, no hardware.

Nothing here opens a device, reads a credential or makes a paid API call. The
whole Hour 5 path

    spoken goal -> find_and_light(target)
        -> fresh frame -> [locate] -> left/center/right
        -> a named LampController gesture
        -> ANOTHER fresh frame -> [locate] -> still there?
        -> light on

runs in-process against stand-ins.

Two properties are load bearing and are the reason most of this file exists.
The light must not come on without a *second*, *fresh* look -- so the tests
count frames and check they differ. And nothing the model says may reach the
robot except as a key into a three-entry literal table -- so the tests try to
smuggle joints, angles and method names through every field there is.

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
import audio_io  # noqa: E402
import character  # noqa: E402
import engagement_demo  # noqa: E402
import goal as goal_module  # noqa: E402
import robot_controller  # noqa: E402
import voice_turn  # noqa: E402
from attention import AttentionSensor  # noqa: E402
from character import CharacterBrain, CharacterError, parse_character_reply  # noqa: E402
from goal import (  # noqa: E402
    GOAL_KINDS,
    MAX_TARGET_CHARS,
    NO_GOAL,
    ORIENTATIONS,
    STAGE_HOLD,
    STAGE_LOCATE,
    STAGE_MOVED,
    STAGE_PENDING,
    STAGE_SPEAK,
    STAGE_VERIFY,
    Goal,
    GoalError,
    GoalOutcome,
    GoalSession,
    TargetSighting,
    normalize_target,
    parse_goal,
    parse_sighting,
)
from scene_memory import LOCATIONS, SceneMemory, SceneObservation  # noqa: E402
from voice_turn import BEHAVIORS, VoiceSession  # noqa: E402

from test_voice_turn import (  # noqa: E402
    FakeLamp,
    FakeRecorder,
    ScriptedSensor,
    ScriptedTalkKey,
    _Endpoint,
    _Response,
    build_turn,
    fake_client,
)

FOUND_LEFT = {"found": True, "location": "left", "confident": True}
STILL_THERE = {"found": True, "location": "center", "confident": True}


def sighting_json(**overrides) -> str:
    payload = dict(FOUND_LEFT)
    payload.update(overrides)
    return json.dumps(payload)


def reply_json(reply="Let me have a look.", behavior="none", **goal_fields) -> str:
    payload = {"reply": reply, "behavior": behavior}
    if goal_fields:
        payload["goal"] = goal_fields.get("goal", goal_fields)
    return json.dumps(payload)


# --------------------------------------------------------------------------
# 1. The goal contract: strict parsing and a tightly bounded target
# --------------------------------------------------------------------------


def test_an_absent_goal_is_no_goal():
    """A response without the field is an ordinary turn, not an error."""
    assert parse_goal(None) is NO_GOAL
    assert parse_character_reply('{"reply": "hi", "behavior": "none"}').goal is NO_GOAL


def test_a_find_and_light_goal_parses_into_a_bounded_target():
    parsed = parse_goal({"kind": "find_and_light", "target": "Mug"})

    assert parsed.kind == "find_and_light"
    assert parsed.target == "mug"
    assert parsed.is_action is True
    assert parsed.summary() == "find_and_light(mug)"


def test_a_none_goal_ignores_whatever_target_came_with_it():
    """'No goal' is no goal, so the noun is never even validated."""
    assert parse_goal({"kind": "none", "target": "mug"}) is NO_GOAL
    assert parse_goal({"kind": "none", "target": ""}).is_action is False


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "fetch", "target": "mug"},                      # outside the enum
        {"kind": "find_and_light"},                              # no target at all
        {"kind": "find_and_light", "target": ""},                # empty target
        {"kind": "find_and_light", "target": "mug", "joint": "neck_yaw_joint"},
        {"kind": "find_and_light", "target": "mug", "angle": 0.85},
        {"kind": ["none", "find_and_light"], "target": "mug"},   # two kinds
        {"target": "mug"},                                       # no kind
        ["find_and_light", "mug"],                               # not an object
        "find_and_light(mug)",                                   # not an object
    ],
    ids=[
        "unknown-kind",
        "missing-target",
        "empty-target",
        "smuggled-joint",
        "smuggled-angle",
        "two-kinds",
        "missing-kind",
        "a-list",
        "a-string",
    ],
)
def test_a_malformed_goal_fails_closed(payload):
    with pytest.raises(GoalError):
        parse_goal(payload)


@pytest.mark.parametrize(
    "target",
    [
        "neck_yaw_joint",          # a joint name
        "0.85",                    # an angle
        "mug; look_left()",        # a method call
        "mug\nSCENE MEMORY",       # a forged prompt header
        "the mug that is on the left of the desk over there",  # over the cap
        "mug!",                    # punctuation
        "mug2",                    # a digit
        "",
        "   ",
        None,
        42,
    ],
    ids=[
        "joint-name",
        "an-angle",
        "a-method-call",
        "a-forged-header",
        "too-long",
        "punctuation",
        "a-digit",
        "empty",
        "whitespace",
        "not-a-string",
        "a-number",
    ],
)
def test_only_a_plain_lowercase_noun_is_accepted_as_a_target(target):
    with pytest.raises(GoalError):
        normalize_target(target)


def test_a_two_word_target_is_still_a_noun():
    assert normalize_target("  Water   Bottle ") == "water bottle"
    assert len("water bottle") <= MAX_TARGET_CHARS


def test_the_goal_vocabulary_is_two_words_wide():
    """A second action means editing this tuple by hand. That is the cost."""
    assert GOAL_KINDS == ("none", "find_and_light")
    assert character.GOAL_SCHEMA["properties"]["kind"]["enum"] == list(GOAL_KINDS)
    assert character.GOAL_SCHEMA["additionalProperties"] is False
    assert set(character.GOAL_SCHEMA["properties"]) == {"kind", "target"}


def test_a_goal_travels_through_a_spoken_turn_without_touching_the_lamp():
    turn, lamp, _recorder, _player = build_turn(
        raw=reply_json(kind="find_and_light", target="mug")
    )

    result = turn.run()

    assert result.goal.target == "mug"
    assert result.behavior == "none"
    # The turn dispatches gestures. A goal is not a gesture, so nothing moved.
    assert lamp.calls == []


def test_an_unusable_goal_fails_the_whole_turn_rather_than_becoming_no_goal():
    turn, lamp, _recorder, player = build_turn(
        raw=reply_json(kind="find_and_light", target="neck_yaw_joint")
    )

    with pytest.raises(voice_turn.TurnError):
        turn.run()

    assert lamp.calls == []
    assert player.played == []


# --------------------------------------------------------------------------
# 2. The target lookup: only bounded semantic output is accepted
# --------------------------------------------------------------------------


def test_a_well_formed_sighting_becomes_a_record():
    sighting = parse_sighting(sighting_json())

    assert sighting.found is True
    assert sighting.location == "left"
    assert sighting.confident is True
    assert sighting.actionable is True
    assert sighting.verified is True


@pytest.mark.parametrize("location", LOCATIONS)
def test_every_declared_location_parses(location):
    assert parse_sighting(sighting_json(location=location)).location == location


@pytest.mark.parametrize(
    "raw",
    [
        sighting_json(location="over there"),          # outside the enum
        sighting_json(location="neck_yaw_joint"),      # a joint name
        sighting_json(location=0.85),                  # an angle
        sighting_json(found="yes"),                    # not a boolean
        sighting_json(confident="true"),               # not a boolean
        '{"found": true, "location": "left"}',         # missing confident
        '{"found": true, "location": "left", "confident": true, "joint": "neck_yaw_joint"}',
        '{"found": true, "location": "left", "confident": true, "angle": 0.85}',
        '{"found": true, "location": "left", "confident": true, "behavior": "nod"}',
        "not json at all",
        "[]",
    ],
    ids=[
        "free-text-location",
        "joint-as-location",
        "angle-as-location",
        "found-not-boolean",
        "confident-not-boolean",
        "missing-field",
        "smuggled-joint",
        "smuggled-angle",
        "smuggled-behavior",
        "not-json",
        "not-an-object",
    ],
)
def test_an_unbounded_lookup_result_fails_closed(raw):
    with pytest.raises(GoalError):
        parse_sighting(raw)


@pytest.mark.parametrize(
    "overrides",
    [
        {"found": False, "location": "unknown"},
        {"found": True, "location": "unknown"},
        {"found": True, "location": "left", "confident": False},
        {"found": False, "location": "left"},
    ],
    ids=["not-found", "unknown-side", "not-confident", "found-false-but-placed"],
)
def test_an_unusable_sighting_is_never_actionable(overrides):
    assert parse_sighting(sighting_json(**overrides)).actionable is False


def test_the_lookup_contract_cannot_request_motion():
    """The vision call has no field that could carry a gesture or a joint."""
    assert set(character.SIGHTING_SCHEMA["properties"]) == {
        "found",
        "location",
        "confident",
    }
    assert character.SIGHTING_SCHEMA["additionalProperties"] is False
    assert character.SIGHTING_SCHEMA["properties"]["location"]["enum"] == list(LOCATIONS)


# --------------------------------------------------------------------------
# 3. locate(): one frame, one bounded noun, one validated answer
# --------------------------------------------------------------------------


def _vision_client(raw: str = None, error: Exception = None):
    client = fake_client()
    client.responses = _Endpoint(
        result=_Response(raw if raw is not None else sighting_json()), error=error
    )
    return client


def test_locate_sends_one_image_and_the_target_noun():
    brain = CharacterBrain(client=_vision_client())

    sighting = brain.locate(b"jpeg-bytes", "Mug")

    (call,) = brain._client.responses.calls
    content = call["input"][0]["content"]
    assert call["model"] == character.VISION_MODEL
    assert call["instructions"] == character.LOCATE_PROMPT
    assert sum(1 for part in content if part["type"] == "input_image") == 1
    assert "mug" in content[0]["text"]
    assert call["timeout"] == character.LOCATE_TIMEOUT_SECONDS
    assert sighting.location == "left"


def test_locate_requests_the_schema_but_still_validates_locally():
    brain = CharacterBrain(client=_vision_client())
    brain.locate(b"jpeg", "mug")
    text = brain._client.responses.calls[0]["text"]

    assert text["format"]["type"] == "json_schema"
    assert text["format"]["schema"] is character.SIGHTING_SCHEMA
    # ...and the local validator rejects what the schema would have caught.
    with pytest.raises(GoalError):
        parse_sighting(sighting_json(location="wherever"))


def test_locate_refuses_an_unbounded_target_before_any_call():
    """A target is a noun in a prompt; anything else never leaves the process."""
    brain = CharacterBrain(client=_vision_client())

    with pytest.raises(CharacterError, match="unbounded target"):
        brain.locate(b"jpeg", "neck_yaw_joint 0.85")

    assert brain._client.responses.calls == []


def test_locate_with_no_frame_fails_before_any_call():
    brain = CharacterBrain(client=_vision_client())

    with pytest.raises(CharacterError, match="no camera frame"):
        brain.locate(b"", "mug")

    assert brain._client.responses.calls == []


def test_a_failed_lookup_is_a_character_error():
    brain = CharacterBrain(client=_vision_client(error=RuntimeError("connection reset")))

    with pytest.raises(CharacterError, match="looking for the mug failed"):
        brain.locate(b"jpeg", "mug")


def test_hour_four_observation_is_untouched_by_the_lookup():
    """Different prompt, different schema, different parser. Both still exist."""
    assert character.OBSERVE_PROMPT != character.LOCATE_PROMPT
    assert set(character.OBSERVATION_SCHEMA["properties"]) == {
        "name",
        "color",
        "location",
        "detail",
        "confident",
    }
    assert "most obvious object" in character.OBSERVE_PROMPT
    assert hasattr(CharacterBrain, "observe") and hasattr(CharacterBrain, "locate")


# --------------------------------------------------------------------------
# 4. The orientation allowlist
# --------------------------------------------------------------------------


class GoalLamp(FakeLamp):
    """FakeLamp plus the two glances and the light."""

    def __init__(self) -> None:
        super().__init__()
        self.light = False

    def look_left(self, *_a, **_k) -> None:
        self.calls.append("look_left")

    def look_right(self, *_a, **_k) -> None:
        self.calls.append("look_right")

    def light_on(self) -> None:
        self.calls.append("light_on")
        self.light = True

    def light_off(self) -> None:
        self.calls.append("light_off")
        self.light = False


@pytest.mark.parametrize(
    "location, expected",
    [("left", "look_left"), ("right", "look_right"), ("center", "engage")],
)
def test_each_location_maps_to_one_named_safe_behavior(location, expected):
    lamp = GoalLamp()

    ORIENTATIONS[location](lamp)

    assert lamp.calls == [expected]


def test_unknown_has_no_entry_so_it_cannot_cause_motion():
    assert "unknown" in LOCATIONS
    assert "unknown" not in ORIENTATIONS
    assert set(ORIENTATIONS) | {"unknown"} == set(LOCATIONS)


def test_orientation_is_a_static_allowlist_not_attribute_lookup():
    """A name the model invents must not reach the lamp even if it exists."""
    lamp = GoalLamp()
    for invented in ("nod", "neutral", "light_on", "set_joint_target", "move_to_pose"):
        assert invented not in ORIENTATIONS
    assert hasattr(lamp, "nod")  # a real, callable method that is not dispatchable


def test_the_goal_module_has_no_dynamic_dispatch_and_no_joints():
    source = (Path(__file__).resolve().parent.parent / "src" / "goal.py").read_text()

    for forbidden in ("eval(", "exec(", "getattr(", "importlib", "_joint", "radians"):
        assert forbidden not in source
    # ...and no PyBullet, no camera, no network.
    for forbidden in ("import pybullet", "import cv2", "import openai"):
        assert forbidden not in source


def test_hour_five_did_not_widen_the_conversational_allowlist():
    assert character.ALLOWED_BEHAVIORS == ("none", "nod", "engage")
    assert set(BEHAVIORS) == {"none", "nod", "engage"}
    assert character.RESPONSE_SCHEMA["properties"]["behavior"]["enum"] == [
        "none",
        "nod",
        "engage",
    ]


# --------------------------------------------------------------------------
# 5. The session: two looks, in order, with a movement between them
# --------------------------------------------------------------------------


class _LocatingBrain:
    """Answers ``locate`` from a script and records what it was given."""

    def __init__(self, *sightings, error=None):
        self.sightings = [
            parse_sighting(s) if isinstance(s, str) else s for s in sightings
        ]
        self.error = error
        #: (jpeg, target) per call, so a test can prove the frames differed.
        self.calls: list[tuple] = []
        self.spoken: list[str] = []
        self.threads: list[str] = []

    def locate(self, jpeg, target):
        self.calls.append((jpeg, target))
        self.threads.append(threading.current_thread().name)
        if self.error is not None:
            raise self.error
        index = min(len(self.calls) - 1, len(self.sightings) - 1)
        return self.sightings[index]

    def synthesize(self, text):
        self.spoken.append(text)
        return b"speech"

    def observe(self, jpeg):  # pragma: no cover - the assertion is the point
        raise AssertionError("the goal path must not use the Hour 4 observation")


class _GoalPlayer:
    def __init__(self) -> None:
        self.played: list = []

    def prepare(self, wav_bytes):
        return audio_io.PreparedPlayback(
            samples=wav_bytes, sample_rate=16_000, channels=1, seconds=0.4
        )

    def play(self, prepared) -> None:
        self.played.append(prepared.samples)


def _inline_goal_session(brain, player=None):
    return GoalSession(brain, player or _GoalPlayer(), spawn=lambda work: work())


MUG = Goal(kind="find_and_light", target="mug")


def test_the_session_refuses_anything_that_is_not_the_one_action():
    session = _inline_goal_session(_LocatingBrain(sighting_json()))

    assert session.request(NO_GOAL, 1) is False
    assert session.request("find_and_light(mug)", 1) is False
    assert session.active is False


def test_a_goal_runs_locate_then_move_then_locate_then_light():
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    session = _inline_goal_session(brain)
    lamp = GoalLamp()

    assert session.request(MUG, 1) is True
    assert session.stage == STAGE_PENDING

    # First fresh frame.
    assert session.supply_frame(b"frame-one") is True
    first = session.poll()
    assert first.stage == STAGE_LOCATE
    assert first.sighting.actionable is True
    assert lamp.calls == []  # nothing has moved yet

    ORIENTATIONS[first.sighting.location](lamp)
    assert session.oriented() is True
    assert session.stage == STAGE_MOVED
    assert lamp.light is False  # ...and nothing is lit yet either

    # Second fresh frame, after the movement.
    assert session.supply_frame(b"frame-two") is True
    second = session.poll()
    assert second.stage == STAGE_VERIFY
    assert second.sighting.verified is True

    lamp.light_on()
    assert lamp.calls == ["look_left", "light_on"]
    assert [frame for frame, _target in brain.calls] == [b"frame-one", b"frame-two"]
    assert session.looks == 2


def test_the_session_will_not_verify_before_the_robot_has_moved():
    """The only route to a second look is through ``oriented``."""
    brain = _LocatingBrain(sighting_json())
    session = _inline_goal_session(brain)
    session.request(MUG, 1)
    session.supply_frame(b"frame-one")
    session.poll()

    # No `oriented()` call: the session is not waiting for a frame...
    assert session.awaiting_frame is False
    assert session.supply_frame(b"frame-two") is False
    assert session.looks == 1
    assert len(brain.calls) == 1

    # ...and `oriented` is only legal straight after the first look.
    assert session.oriented() is True
    assert session.oriented() is False


def test_a_second_goal_is_refused_while_one_is_running():
    session = _inline_goal_session(_LocatingBrain(sighting_json()))
    assert session.request(MUG, 1) is True
    assert session.request(Goal(kind="find_and_light", target="bottle"), 1) is False


def test_a_failed_lookup_becomes_an_outcome_not_an_exception():
    session = _inline_goal_session(_LocatingBrain(error=GoalError("no clear answer")))
    session.request(MUG, 1)
    session.supply_frame(b"frame")

    outcome = session.poll()
    assert outcome.sighting is None
    assert isinstance(outcome.error, GoalError)
    assert session.busy is False


def test_the_session_survives_a_worker_that_raises_something_unexpected():
    session = _inline_goal_session(_LocatingBrain(error=KeyboardInterrupt("interrupted")))
    session.request(MUG, 1)
    session.supply_frame(b"frame")

    assert isinstance(session.poll().error, KeyboardInterrupt)
    assert session.busy is False


def test_the_worker_receives_bytes_and_a_noun_and_nothing_else():
    """Everything OpenCV-shaped and everything robot-shaped stays behind."""
    brain = _LocatingBrain(sighting_json())
    session = _inline_goal_session(brain)
    session.request(MUG, 1)
    session.supply_frame(b"jpeg")

    (frame, target), = brain.calls
    assert isinstance(frame, bytes)
    assert target == "mug"


# --------------------------------------------------------------------------
# 6. The vision calls do not run on the control thread
# --------------------------------------------------------------------------


def _await_goal(session, tries: int = 500):
    for _ in range(tries):
        outcome = session.poll()
        if outcome is not None:
            return outcome
        time.sleep(0.01)
    raise AssertionError("the goal worker never published a result")


def test_both_vision_calls_run_on_a_worker_thread():
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    session = GoalSession(brain, _GoalPlayer())  # the real threaded spawn
    control_thread = threading.current_thread().name

    session.request(MUG, 1)
    session.supply_frame(b"frame-one")
    assert _await_goal(session).stage == STAGE_LOCATE
    session.oriented()
    session.supply_frame(b"frame-two")
    assert _await_goal(session).stage == STAGE_VERIFY

    assert len(brain.threads) == 2
    assert control_thread not in brain.threads


def test_the_closing_sentence_is_rendered_on_a_worker_thread_too():
    brain = _LocatingBrain(sighting_json())
    player = _GoalPlayer()
    session = GoalSession(brain, player)
    session.request(MUG, 1)

    assert session.hold(goal_module.found_line("mug")) is True
    assert session.held is True
    assert session.say() is True
    outcome = _await_goal(session)

    assert outcome.stage == STAGE_SPEAK
    assert brain.spoken == [goal_module.found_line("mug")]
    # Rendered, not played: playing is the control thread's job.
    assert player.played == []


# --------------------------------------------------------------------------
# 7. The light lives in the controller, and starts and ends off
# --------------------------------------------------------------------------


@pytest.fixture
def headless_lamp():
    import pybullet as p  # noqa: PLC0415  (only the light tests need a simulator)
    from robot_controller import LampController, load_lamp  # noqa: PLC0415

    client = p.connect(p.DIRECT)
    try:
        yield LampController(load_lamp(client), client)
    finally:
        p.disconnect(client)


def test_the_lamp_starts_with_its_light_off(headless_lamp):
    assert headless_lamp.light_is_on is False


def test_the_light_switches_and_reports_its_state(headless_lamp):
    import robot_controller  # noqa: PLC0415
    import pybullet as p  # noqa: PLC0415

    headless_lamp.light_on()
    assert headless_lamp.light_is_on is True
    lit = p.getVisualShapeData(headless_lamp.body_id, physicsClientId=headless_lamp.client_id)

    headless_lamp.light_off()
    assert headless_lamp.light_is_on is False
    dark = p.getVisualShapeData(headless_lamp.body_id, physicsClientId=headless_lamp.client_id)

    def colour_of(shapes):
        return [shape[7] for shape in shapes if shape[1] == headless_lamp._light_index]

    assert colour_of(lit) == [robot_controller.LIGHT_ON_RGBA]
    assert colour_of(dark) == [robot_controller.LIGHT_OFF_RGBA]


def test_the_light_is_found_by_name_not_by_index(headless_lamp):
    import pybullet as p  # noqa: PLC0415
    import robot_controller  # noqa: PLC0415

    info = p.getJointInfo(
        headless_lamp.body_id, headless_lamp._light_index,
        physicsClientId=headless_lamp.client_id,
    )
    assert info[12].decode() == robot_controller.LIGHT_LINK


def test_switching_the_light_commands_no_joint(headless_lamp):
    before = dict(headless_lamp._targets)
    headless_lamp.light_on()
    assert headless_lamp._targets == before


# --------------------------------------------------------------------------
# 8. End to end through the demo loop
# --------------------------------------------------------------------------


class _FrameSensor(ScriptedSensor):
    """A scripted sensor whose every frame is visibly different from the last.

    Two frames that encode to the same bytes would make "was the second look
    fresh?" unanswerable, so each read returns a distinct flat image.
    """

    def read(self):
        index = self.frames_read
        reading = super().read()
        frame = np.full((48, 64, 3), (index * 7) % 256, dtype=np.uint8)
        return type(reading)(attending=reading.attending, faces=(), frame=frame)


class GoalTurn:
    """A VoiceTurn stand-in that hands the loop one goal and nothing else.

    *speech_seconds* is how long the reply claims to take to say. It is what
    holds the lamp's speaker busy, so a large value is how a test keeps the
    closing sentence waiting.
    """

    def __init__(self, goal=MUG, reply="Let me have a look.", speech_seconds=0.0):
        self.goal = goal
        self.reply = reply
        self.speech_seconds = speech_seconds
        self.commits = 0

    def prepare(self):
        return voice_turn.PreparedTurn(
            transcript="find my mug and shine your light on it",
            reply=self.reply,
            behavior="none",
            playback=audio_io.PreparedPlayback(
                samples=b"", sample_rate=16_000, channels=1, seconds=self.speech_seconds
            ),
            goal=self.goal,
        )

    def commit(self, prepared):
        self.commits += 1
        return voice_turn.TurnResult(
            transcript=prepared.transcript,
            reply=prepared.reply,
            behavior=prepared.behavior,
            moved=False,
            speech_seconds=self.speech_seconds,
            goal=prepared.goal,
        )


def _spy_lamp(monkeypatch):
    """Wrap the real LampController and log every physical thing it does."""
    log: list[str] = []
    original = engagement_demo.LampController

    class Spy(original):  # type: ignore[misc, valid-type]
        def look_left(self, *a, **k):
            log.append("look_left")
            super().look_left(*a, **k)

        def look_right(self, *a, **k):
            log.append("look_right")
            super().look_right(*a, **k)

        def engage(self, *a, **k):
            log.append("engage")
            super().engage(*a, **k)

        def nod(self, *a, **k):
            log.append("nod")
            super().nod(*a, **k)

        def light_on(self):
            log.append("light_on")
            super().light_on()

        def light_off(self):
            log.append("light_off")
            super().light_off()

    monkeypatch.setattr(engagement_demo, "LampController", Spy)
    return log


def _run_goal_demo(monkeypatch, brain, *, script=None, press=40, goal=MUG):
    """Drive the real loop headlessly with a scripted goal turn."""
    log = _spy_lamp(monkeypatch)
    player = _GoalPlayer()
    turn = GoalTurn(goal=goal)
    engagement_demo.run(
        preview=False,
        headless=True,
        sensor=_FrameSensor(script if script is not None else [(True, 8.0)]),
        voice_factory=lambda _lamp: turn,
        talk_key=ScriptedTalkKey(press),
        goals=_inline_goal_session(brain, player),
        player=player,
    )
    return log, player


def test_the_light_comes_on_only_after_a_second_fresh_look(monkeypatch, capsys):
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))

    log, player = _run_goal_demo(monkeypatch, brain)
    out = capsys.readouterr().out

    frames = [frame for frame, _target in brain.calls]
    assert len(frames) == 2                      # exactly two looks
    assert frames[0] != frames[1]                # ...and they are different frames
    assert [target for _f, target in brain.calls] == ["mug", "mug"]

    # The order that matters: glance, then the second look, then the light.
    physical = [entry for entry in log if entry in ("look_left", "light_on")]
    assert physical == ["look_left", "light_on"]
    assert log.index("look_left") < log.index("light_on")
    assert "light ON" in out
    assert brain.spoken == [goal_module.found_line("mug")]
    assert player.played == [b"speech"]


def test_the_second_frame_is_captured_after_the_head_moved(monkeypatch):
    """The verification frame must not be one taken before the gesture.

    The loop hands a frame over only on a tick that begins with a fresh
    ``sensor.read()``, so the two looks are necessarily separated by at least
    one camera read -- and the gesture happens in between.
    """
    reads_at_call = []
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    sensor_box = {}

    original = engagement_demo.encode_frame_jpeg

    def spy_encode(frame, *a, **k):
        reads_at_call.append(sensor_box["sensor"].frames_read)
        return original(frame, *a, **k)

    monkeypatch.setattr(engagement_demo, "encode_frame_jpeg", spy_encode)
    log = _spy_lamp(monkeypatch)
    player = _GoalPlayer()
    sensor = _FrameSensor([(True, 8.0)])
    sensor_box["sensor"] = sensor

    engagement_demo.run(
        preview=False,
        headless=True,
        sensor=sensor,
        voice_factory=lambda _lamp: GoalTurn(),
        talk_key=ScriptedTalkKey(40),
        goals=_inline_goal_session(brain, player),
        player=player,
    )

    assert len(reads_at_call) == 2
    assert reads_at_call[1] > reads_at_call[0]  # a later camera read entirely
    assert "look_left" in log and "light_on" in log


@pytest.mark.parametrize(
    "first",
    [
        {"found": False, "location": "unknown"},
        {"found": True, "location": "unknown"},
        {"found": True, "location": "left", "confident": False},
    ],
    ids=["not-found", "unknown-side", "not-confident"],
)
def test_a_target_that_is_not_confidently_placed_causes_no_motion_and_no_light(
    monkeypatch, capsys, first
):
    brain = _LocatingBrain(sighting_json(**first))

    log, _player = _run_goal_demo(monkeypatch, brain)
    capsys.readouterr()

    assert len(brain.calls) == 1                      # no second look was earned
    assert "look_left" not in log and "look_right" not in log
    assert "light_on" not in log
    assert brain.spoken == [goal_module.missing_line("mug")]


def test_a_failed_second_verification_leaves_the_light_off(monkeypatch, capsys):
    brain = _LocatingBrain(sighting_json(), sighting_json(found=False, location="unknown"))

    log, _player = _run_goal_demo(monkeypatch, brain)
    out = capsys.readouterr().out

    assert len(brain.calls) == 2      # it did look again...
    assert "look_left" in log         # ...and it did turn...
    assert "light_on" not in log      # ...but the light stayed off.
    assert brain.spoken == [goal_module.lost_line("mug")]
    assert "leaving the light off" in out


def test_a_failing_lookup_call_leaves_the_light_off(monkeypatch, capsys):
    brain = _LocatingBrain(error=CharacterError("connection reset"))

    log, _player = _run_goal_demo(monkeypatch, brain)
    err = capsys.readouterr().err

    assert "look_left" not in log
    assert "light_on" not in log
    assert "lookup failed" in err
    assert brain.spoken == [goal_module.missing_line("mug")]


def test_the_demo_leaves_the_light_off_when_the_loop_ends(monkeypatch, capsys):
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))

    log, _player = _run_goal_demo(monkeypatch, brain)
    capsys.readouterr()

    assert log[0] == "light_off"      # a known dark state at construction
    assert log[-1] == "light_off"     # ...and again on the way out
    assert "light_on" in log


def test_a_goal_that_outlives_its_engagement_is_dropped(monkeypatch, capsys):
    """The person left while the first look was in flight: fail closed."""
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    session = _inline_goal_session(brain)
    lamp = GoalLamp()

    session.request(MUG, epoch=1)
    session.supply_frame(b"frame-one")
    outcome = session.poll()

    class _Idle:
        state = engagement_demo.EngagementState.IDLE

    engagement_demo._handle_goal(outcome, session, lamp, _GoalPlayer(), _NoFlush(), _Idle(), 1)

    assert lamp.calls == ["light_off"]
    assert session.active is False


def test_a_goal_from_a_previous_conversation_is_dropped(monkeypatch, capsys):
    brain = _LocatingBrain(sighting_json())
    session = _inline_goal_session(brain)
    lamp = GoalLamp()
    session.request(MUG, epoch=1)
    session.supply_frame(b"frame-one")
    outcome = session.poll()

    class _Engaged:
        state = engagement_demo.EngagementState.ENGAGED

    # Same engagement state, but a newer epoch: they left and came back.
    engagement_demo._handle_goal(outcome, session, lamp, _GoalPlayer(), _NoFlush(), _Engaged(), 2)

    assert lamp.calls == ["light_off"]
    assert session.active is False


class _NoFlush:
    """A sensor stand-in whose freshness boundary always succeeds."""

    def flush(self) -> bool:
        return True


def test_a_goal_blocks_a_second_spoken_turn_until_it_finishes(monkeypatch, capsys):
    """Two presses: the second lands while the goal still owns the head.

    The goal's worker is deliberately never run, so the goal stays in flight
    for the rest of the script and the guard is exercised deterministically
    rather than by winning a race with a script that is long enough.
    """
    brain = _LocatingBrain(sighting_json())
    log = _spy_lamp(monkeypatch)
    player = _GoalPlayer()
    turn = GoalTurn()

    engagement_demo.run(
        preview=False,
        headless=True,
        sensor=_FrameSensor([(True, 8.0)]),
        voice_factory=lambda _lamp: turn,
        talk_key=ScriptedTalkKey(40, 60, 80),
        goals=GoalSession(brain, player, spawn=lambda _work: None),
        player=player,
    )
    out = capsys.readouterr().out

    assert "still looking for that" in out
    assert turn.commits == 1
    assert "light_on" not in log


def test_the_camera_only_demo_has_no_goal_path(capsys):
    engagement_demo.run(preview=False, headless=True, sensor=ScriptedSensor([(True, 2.0)]))
    out = capsys.readouterr().out

    assert "shine your light on it" not in out


def test_a_goal_with_no_goal_path_wired_is_reported_not_executed(monkeypatch, capsys):
    log = _spy_lamp(monkeypatch)
    engagement_demo.run(
        preview=False,
        headless=True,
        sensor=_FrameSensor([(True, 6.0)]),
        voice_factory=lambda _lamp: GoalTurn(),
        talk_key=ScriptedTalkKey(40),
    )
    out = capsys.readouterr().out

    assert "can't go and find a mug" in out
    assert "light_on" not in log


# --------------------------------------------------------------------------
# 9. Hour 4 recall is still recall
# --------------------------------------------------------------------------


class _NoLookingBrain(CharacterBrain):
    """Fails loudly if a question turns into a fresh look of any kind."""

    def observe(self, jpeg_bytes):  # pragma: no cover - the assertion is the point
        raise AssertionError("answering a question must not submit a frame")

    def locate(self, jpeg_bytes, target):  # pragma: no cover - ditto
        raise AssertionError("answering a question must not look for anything")


def test_answering_a_question_still_performs_no_fresh_observation():
    memory = SceneMemory()
    memory.remember(
        SceneObservation(name="mug", color="white", location="left", detail="")
    )
    turn = voice_turn.VoiceTurn(
        brain=_NoLookingBrain(client=fake_client(reply="It was a white mug.")),
        recorder=__import__("test_voice_turn").FakeRecorder(),
        player=__import__("test_voice_turn").FakePlayer(),
        lamp=GoalLamp(),
        memory=memory,
    )

    result = turn.run()

    assert result.reply == "It was a white mug."
    assert result.goal is NO_GOAL


class _Engaged:
    state = engagement_demo.EngagementState.ENGAGED


class _Idle:
    state = engagement_demo.EngagementState.IDLE


def _active_session(brain=None, player=None):
    """An inline session sitting on an accepted goal, ready to be poked."""
    session = _inline_goal_session(brain or _LocatingBrain(sighting_json()), player)
    session.request(MUG, 1)
    return session


# --------------------------------------------------------------------------
# 10. Repair regressions -- one section per Codex finding
# --------------------------------------------------------------------------

# -- finding 1: the result/busy race ---------------------------------------


class _PausingQueue:
    """Freezes a worker after it has published, before it clears ``busy``.

    That window is exactly the race: the outcome is visible in the queue while
    the session still calls itself busy, so a control thread that dequeued it
    would be told ``False`` by the very transition it was making.
    """

    def __init__(self, inner):
        self._inner = inner
        self.published = threading.Event()
        self.release = threading.Event()

    def put(self, item):
        self._inner.put(item)
        self.published.set()
        if not self.release.wait(10):  # pragma: no cover - a hung test, not a pass
            raise AssertionError("the test never released the worker")

    def get_nowait(self):
        return self._inner.get_nowait()

    def empty(self):
        return self._inner.empty()


def test_a_result_is_not_consumable_until_the_worker_has_cleared_busy():
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    session = GoalSession(brain, _GoalPlayer())
    gate = _PausingQueue(session._results)
    session._results = gate

    session.request(MUG, 1)
    session.supply_frame(b"frame-one")
    assert gate.published.wait(5)

    # Published, but not yet finished. The result must stay invisible...
    assert session.busy is True
    assert session.poll() is None
    assert session.occupied is True
    # ...and it must not look idle either, in the other direction.
    assert session.request(MUG, 1) is False

    gate.release.set()
    outcome = _await_goal(session)

    assert outcome.stage == STAGE_LOCATE
    assert session.busy is False
    # The transition that used to be silently refused now succeeds, so the
    # goal is not left permanently active.
    assert session.oriented() is True
    assert session.supply_frame(b"frame-two") is True
    assert _await_goal(session).stage == STAGE_VERIFY
    assert session.hold(goal_module.found_line("mug")) is True
    assert session.say() is True
    assert _await_goal(session).stage == STAGE_SPEAK
    session.finish()
    assert session.occupied is False


def test_a_transition_the_session_refuses_does_not_leave_the_goal_active(capsys):
    """A refused ``hold`` must end the goal, not strand it active and lit."""

    class _Refusing(GoalSession):
        def hold(self, line):
            return False

    brain = _LocatingBrain(sighting_json(found=False, location="unknown"))
    session = _Refusing(brain, _GoalPlayer(), spawn=lambda work: work())
    lamp = GoalLamp()
    session.request(MUG, 1)
    session.supply_frame(b"frame")
    outcome = session.poll()

    engagement_demo._handle_goal(
        outcome, session, lamp, _GoalPlayer(), _NoFlush(), _Engaged(), 1
    )

    assert session.active is False
    assert session.occupied is False
    assert "light_on" not in lamp.calls
    assert "dropped the goal" in capsys.readouterr().out


# -- finding 2: a new goal starts from the dark -----------------------------


class _QuietSession:
    """The slice of VoiceSession that the goal lifecycle actually uses."""

    busy = False

    def __init__(self) -> None:
        self.quiet_until = 0.0

    def mark_speaking(self, seconds, now):
        self.quiet_until = max(self.quiet_until, now + max(0.0, seconds))

    def speaking(self, now):
        return now < self.quiet_until


def _accept_goal(goals, lamp, epoch=1, session=None):
    """Push one committed spoken turn carrying a goal through the handler."""
    turn = GoalTurn()
    outcome = voice_turn.TurnOutcome(epoch=epoch, prepared=turn.prepare())
    engagement_demo._handle_outcome(
        outcome,
        session or _QuietSession(),
        turn,
        _NoFlush(),
        _Engaged(),
        epoch,
        goals,
        lamp,
    )


def test_a_new_goal_puts_out_a_light_the_previous_goal_left_on(capsys):
    lamp = GoalLamp()

    # Goal 1 succeeded and left the lamp illuminating the mug.
    lamp.light_on()
    assert lamp.light is True

    # Goal 2 is accepted...
    failing = _inline_goal_session(_LocatingBrain(sighting_json(found=False, location="unknown")))
    _accept_goal(failing, lamp)

    assert failing.active is True
    assert lamp.light is False  # ...and the lamp is dark before it looks at anything

    # ...and then goal 2's first lookup fails. The lamp must stay dark rather
    # than appearing to still be lighting the previous target.
    failing.supply_frame(b"frame")
    engagement_demo._handle_goal(
        failing.poll(), failing, lamp, _GoalPlayer(), _NoFlush(), _Engaged(), 1
    )
    capsys.readouterr()

    assert lamp.light is False
    assert lamp.calls.count("light_on") == 1  # only the first goal's


def test_a_goal_is_refused_when_the_light_will_not_go_out(capsys):
    """An unknown physical state is not a state to start a new goal from."""

    class _StuckLight(GoalLamp):
        def light_off(self):
            self.calls.append("light_off")
            raise RuntimeError("the simulator went away")

    lamp = _StuckLight()
    goals = _inline_goal_session(_LocatingBrain(sighting_json()))

    _accept_goal(goals, lamp)
    out = capsys.readouterr()

    assert goals.active is False
    assert "could not put the light out" in out.out
    assert "turning the light off failed" in out.err


@pytest.mark.parametrize(
    "sightings, stage",
    [
        ([sighting_json(found=False, location="unknown")], "first look"),
        ([sighting_json(), sighting_json(found=False, location="unknown")], "second look"),
    ],
    ids=["first-look-fails", "verification-fails"],
)
def test_every_unsuccessful_ending_leaves_the_light_off(monkeypatch, capsys, sightings, stage):
    brain = _LocatingBrain(*sightings)
    log, _player = _run_goal_demo(monkeypatch, brain)
    capsys.readouterr()

    assert "light_on" not in log
    assert log[-1] == "light_off"


# -- finding 3: the post-motion acquisition boundary ------------------------


class _BufferedCapture:
    """A capture whose driver queue holds several pre-motion frames.

    A queued frame comes back instantly; once the queue is empty the device
    makes the caller wait, exactly as a real sensor does. That difference is
    the only signal available to a caller that cannot know the queue depth.
    """

    def __init__(self, backlog, live_delay: float = 0.02):
        self.backlog = list(backlog)
        self.live_delay = live_delay
        self.live = "post-motion"
        self.grabbed: list = []

    def _next(self):
        if self.backlog:
            return self.backlog.pop(0)
        time.sleep(self.live_delay)
        return self.live

    def grab(self):
        self.grabbed.append(self._next())
        return True

    def read(self):
        return True, self._next()


def _sensor_with(capture):
    sensor = AttentionSensor.__new__(AttentionSensor)
    sensor._capture = capture
    sensor._detect_faces = lambda _frame: []
    return sensor


def test_the_freshness_boundary_outlasts_a_deep_pre_motion_buffer():
    """Six queued pre-motion frames, not two.

    The old implementation grabbed a fixed two and would hand ``pre-2`` to the
    verification lookup. The boundary is established by waiting for a grab that
    the device actually made us wait for, so the depth does not matter.
    """
    capture = _BufferedCapture([f"pre-{index}" for index in range(6)])
    sensor = _sensor_with(capture)

    assert sensor.flush() is True

    # Every stale frame was consumed and discarded by the boundary...
    assert [frame for frame in capture.grabbed if str(frame).startswith("pre-")] == [
        f"pre-{index}" for index in range(6)
    ]
    # ...and what the caller gets next is a live exposure.
    assert sensor.read().frame == "post-motion"


def test_the_boundary_reports_failure_when_the_device_never_waits():
    """A capture that always answers instantly cannot prove its queue is empty."""

    class _NeverWaits:
        def __init__(self):
            self.grabs = 0

        def grab(self):
            self.grabs += 1
            return True

    capture = _NeverWaits()

    assert _sensor_with(capture).flush() is False
    assert capture.grabs <= attention.FRESH_MAX_GRABS


def test_the_boundary_reports_failure_instead_of_raising_on_a_dead_camera():
    """Hour 2 behaviour is unchanged: the *read* is still what raises."""

    class _Dead:
        def grab(self):
            return False

        def read(self):
            return False, None

    sensor = _sensor_with(_Dead())

    assert sensor.flush() is False
    with pytest.raises(attention.CameraError, match="no frame returned"):
        sensor.read()


def test_a_boundary_that_cannot_be_established_fails_the_goal_closed(capsys):
    """No provable freshness means no second look and no light."""

    class _UnprovableSensor:
        def flush(self):
            return False

    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    session = _inline_goal_session(brain)
    lamp = GoalLamp()
    session.request(MUG, 1)
    session.supply_frame(b"frame-one")

    engagement_demo._handle_goal(
        session.poll(), session, lamp, _GoalPlayer(), _UnprovableSensor(), _Engaged(), 1
    )
    out = capsys.readouterr().out

    assert session.active is False
    assert session.awaiting_frame is False
    assert len(brain.calls) == 1               # no verification look happened
    assert "light_on" not in lamp.calls
    assert "not provably past the turn" in out


def test_a_flush_that_raises_fails_the_goal_closed(capsys):
    class _RaisingSensor:
        def flush(self):
            raise RuntimeError("the camera went away")

    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    session = _inline_goal_session(brain)
    lamp = GoalLamp()
    session.request(MUG, 1)
    session.supply_frame(b"frame-one")

    engagement_demo._handle_goal(
        session.poll(), session, lamp, _GoalPlayer(), _RaisingSensor(), _Engaged(), 1
    )
    captured = capsys.readouterr()

    assert session.active is False
    assert "light_on" not in lamp.calls
    assert "clearing the camera after the turn failed" in captured.err


# -- finding 4: abandoned goals stay occupied until they drain --------------


def test_an_abandoned_goal_stays_occupied_until_its_worker_drains():
    gate = threading.Event()

    class _Slow(_LocatingBrain):
        def locate(self, jpeg, target):
            gate.wait(10)
            return super().locate(jpeg, target)

    session = GoalSession(_Slow(sighting_json()), _GoalPlayer())
    session.request(MUG, 1)
    session.supply_frame(b"frame")

    session.finish()  # abandoned while the worker is still running
    assert session.active is False
    assert session.occupied is True     # ...but nothing else may start

    gate.set()
    for _ in range(500):
        if not session.busy:
            break
        time.sleep(0.01)
    assert session.busy is False
    assert session.occupied is True     # the result is queued and undrained

    assert session.poll() is not None
    assert session.occupied is False    # only now is the session free


def test_a_stale_result_is_drained_without_moving_or_lighting(capsys):
    session = _inline_goal_session(_LocatingBrain(sighting_json()))
    lamp = GoalLamp()
    session.request(MUG, 1)
    session.supply_frame(b"frame")
    session.finish()  # abandoned with the result already in the queue

    outcome = session.poll()
    assert outcome is not None
    engagement_demo._handle_goal(
        outcome, session, lamp, _GoalPlayer(), _NoFlush(), _Engaged(), 1
    )

    assert lamp.calls == []             # no motion, no light, nothing at all
    assert session.occupied is False
    assert "abandoned goal" in capsys.readouterr().out


#: A script that engages, presses talk, loses the person entirely, then brings
#: them back. The goal's worker is never run, so it is still holding the shared
#: brain when the second conversation starts.
ABANDON_SCRIPT = [(True, 3.0), (False, 3.0), (True, 4.0)]


def _run_abandoning_demo(monkeypatch, *, talk=(), observe=()):
    log = _spy_lamp(monkeypatch)
    player = _GoalPlayer()
    brain = _LocatingBrain(sighting_json())
    observer_brain = _RecordingObserver()
    engagement_demo.run(
        preview=False,
        headless=True,
        sensor=_FrameSensor(ABANDON_SCRIPT),
        voice_factory=lambda _lamp: GoalTurn(),
        talk_key=ScriptedTalkKey(*talk),
        observe_key=ScriptedKey(*observe) if observe else None,
        observer=scene_memory_session(observer_brain) if observe else None,
        goals=GoalSession(brain, player, spawn=lambda _work: None),
        player=player,
    )
    return log, brain, observer_brain


class _RecordingObserver:
    """Stands in for the Hour 4 vision call; records every frame it is given."""

    def __init__(self) -> None:
        self.frames: list = []

    def observe(self, jpeg):
        self.frames.append(jpeg)
        return SceneObservation(name="mug", color="white", location="left", detail="")


class ScriptedKey:
    def __init__(self, *frames: int):
        self.frames = set(frames)
        self.index = -1

    def pressed(self) -> bool:
        self.index += 1
        return self.index in self.frames


def scene_memory_session(brain):
    from scene_memory import ObservationSession  # noqa: PLC0415

    return ObservationSession(brain, spawn=lambda work: work())


def test_a_new_turn_is_blocked_while_an_abandoned_goal_worker_is_still_running(
    monkeypatch, capsys
):
    log, brain, _observer = _run_abandoning_demo(monkeypatch, talk=(20, 120))
    out = capsys.readouterr().out

    assert "dropped the goal: the person left" in out
    # The re-engaged conversation's press is refused by the *occupied* guard,
    # not by `active` -- the abandoned goal is no longer active.
    assert "still looking for that" in out
    assert "light_on" not in log


def test_an_hour_four_observation_is_blocked_while_an_abandoned_goal_drains(
    monkeypatch, capsys
):
    _log, _brain, observer = _run_abandoning_demo(monkeypatch, talk=(20,), observe=(120,))
    out = capsys.readouterr().out

    assert "dropped the goal: the person left" in out
    assert "still looking for that" in out
    assert observer.frames == []  # no frame was sent while the goal still held the brain


# -- finding 5: one lamp, one speaker, one quiet deadline -------------------


def test_the_closing_line_waits_for_the_reply_to_finish_playing(monkeypatch, capsys):
    """The reply claims 30 s of speech -- longer than the whole script."""
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    log = _spy_lamp(monkeypatch)
    player = _GoalPlayer()
    goals = _inline_goal_session(brain, player)

    engagement_demo.run(
        preview=False,
        headless=True,
        sensor=_FrameSensor([(True, 8.0)]),
        voice_factory=lambda _lamp: GoalTurn(speech_seconds=30.0),
        talk_key=ScriptedTalkKey(40),
        goals=goals,
        player=player,
    )
    capsys.readouterr()

    # The visual half of the goal ran to completion...
    assert len(brain.calls) == 2
    assert "light_on" in log
    # ...but the closing sentence was never prepared on the shared player,
    # because the reply was still coming out of it.
    assert brain.spoken == []
    assert player.played == []
    assert goals.held is True


def test_the_closing_line_is_spoken_once_the_speaker_is_free(monkeypatch, capsys):
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    _log, player = _run_goal_demo(monkeypatch, brain)  # a 0 s reply
    capsys.readouterr()

    assert brain.spoken == [goal_module.found_line("mug")]
    assert player.played == [b"speech"]


def test_goal_closing_playback_extends_the_shared_quiet_deadline():
    session = VoiceSession(turn=None, spawn=lambda _work: None)
    goals = _active_session()
    goals.hold("there it is")
    lamp = GoalLamp()
    now = time.monotonic()

    assert session.ready(now) is True
    engagement_demo._handle_goal(
        GoalOutcome(
            stage=STAGE_SPEAK,
            epoch=1,
            target="mug",
            playback=audio_io.PreparedPlayback(
                samples=b"x", sample_rate=16_000, channels=1, seconds=2.0
            ),
        ),
        goals,
        lamp,
        _GoalPlayer(),
        _NoFlush(),
        _Engaged(),
        1,
        session,
    )

    # The microphone stays shut for the length of the closing sentence...
    assert session.speaking(time.monotonic()) is True
    assert session.ready(time.monotonic()) is False
    # ...and normal turns work again once it has elapsed.
    assert session.ready(time.monotonic() + 3.0) is True
    assert goals.active is False


def test_voice_and_goal_never_prepare_the_shared_player_at_once(monkeypatch, capsys):
    """One player, two workers, and the guards that keep them apart."""

    class _ExclusivePlayer(_GoalPlayer):
        def __init__(self) -> None:
            super().__init__()
            self._busy = threading.Lock()
            self.overlaps = 0
            self.prepares = 0

        def prepare(self, wav_bytes):
            if not self._busy.acquire(blocking=False):
                self.overlaps += 1
                raise AssertionError("two prepares on the shared player at once")
            try:
                self.prepares += 1
                time.sleep(0.02)
                return super().prepare(wav_bytes)
            finally:
                self._busy.release()

    player = _ExclusivePlayer()
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    log = _spy_lamp(monkeypatch)

    def factory(lamp):
        return voice_turn.VoiceTurn(
            brain=CharacterBrain(
                client=fake_client(raw=reply_json(kind="find_and_light", target="mug"))
            ),
            recorder=FakeRecorder(),
            player=player,
            lamp=lamp,
        )

    engagement_demo.run(
        preview=False,
        headless=True,
        sensor=_FrameSensor([(True, 10.0)]),
        voice_factory=factory,
        talk_key=ScriptedTalkKey(40, 60, 90),
        goals=_inline_goal_session(brain, player),
        player=player,
    )
    capsys.readouterr()

    assert player.overlaps == 0
    assert player.prepares >= 2  # at least one reply and one closing sentence
    assert "light_on" in log


# -- finding 6: a physical failure must not kill the loop -------------------


def test_a_failing_orientation_fails_the_goal_without_raising(capsys):
    class _StuckHead(GoalLamp):
        def look_left(self, *_a, **_k):
            self.calls.append("look_left")
            raise RuntimeError("joint motor rejected the command")

    lamp = _StuckHead()
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    session = _inline_goal_session(brain)
    session.request(MUG, 1)
    session.supply_frame(b"frame-one")

    engagement_demo._handle_goal(
        session.poll(), session, lamp, _GoalPlayer(), _NoFlush(), _Engaged(), 1
    )
    captured = capsys.readouterr()

    assert session.active is False
    assert len(brain.calls) == 1          # the goal did not advance to verification
    assert "light_on" not in lamp.calls
    assert "light_off" in lamp.calls      # ...and it tried to leave the lamp dark
    assert "orienting towards the target failed" in captured.err


def test_a_failing_light_on_fails_the_goal_without_raising(capsys):
    class _DeadBulb(GoalLamp):
        def light_on(self):
            self.calls.append("light_on")
            raise RuntimeError("changeVisualShape failed")

    lamp = _DeadBulb()
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    session = _inline_goal_session(brain)
    session.request(MUG, 1)
    session.supply_frame(b"frame-one")
    engagement_demo._handle_goal(
        session.poll(), session, lamp, _GoalPlayer(), _NoFlush(), _Engaged(), 1
    )
    session.supply_frame(b"frame-two")
    engagement_demo._handle_goal(
        session.poll(), session, lamp, _GoalPlayer(), _NoFlush(), _Engaged(), 1
    )
    captured = capsys.readouterr()

    assert session.active is False
    assert lamp.light is False
    assert "turning the light on failed" in captured.err
    assert "the light would not come on" in captured.out


def test_a_failing_cleanup_is_reported_and_does_not_mask_the_original_failure(capsys):
    class _Doomed(GoalLamp):
        def look_left(self, *_a, **_k):
            raise RuntimeError("motor fault")

        def light_off(self):
            raise RuntimeError("simulator disconnected")

    lamp = _Doomed()
    session = _inline_goal_session(_LocatingBrain(sighting_json()))
    session.request(MUG, 1)
    session.supply_frame(b"frame-one")

    engagement_demo._handle_goal(
        session.poll(), session, lamp, _GoalPlayer(), _NoFlush(), _Engaged(), 1
    )
    captured = capsys.readouterr()

    assert session.active is False
    assert "orienting towards the target failed" in captured.err   # the original
    assert "turning the light off failed" in captured.err          # ...and the cleanup
    assert "the orientation gesture failed" in captured.out


def test_the_live_loop_survives_a_physical_goal_failure(monkeypatch, capsys):
    """The whole demo must keep running, engagement and all."""
    original = engagement_demo.LampController

    class _StuckHead(original):  # type: ignore[misc, valid-type]
        def look_left(self, *_a, **_k):
            raise RuntimeError("joint motor rejected the command")

    monkeypatch.setattr(engagement_demo, "LampController", _StuckHead)
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    player = _GoalPlayer()

    engagement_demo.run(
        preview=False,
        headless=True,
        sensor=_FrameSensor([(True, 5.0), (False, 4.0)]),
        voice_factory=lambda _lamp: GoalTurn(),
        talk_key=ScriptedTalkKey(40),
        goals=_inline_goal_session(brain, player),
        player=player,
    )
    out = capsys.readouterr()

    # The loop ran to the end of its script, including the disengagement.
    assert "ENGAGED -> IDLE" in out.out
    assert "the orientation gesture failed" in out.out
    assert len(brain.calls) == 1


# -- finding 7: exactly one light link --------------------------------------


class _StubBody:
    body_id = 0
    client_id = 0


def _patch_links(monkeypatch, names):
    monkeypatch.setattr(
        robot_controller.p, "getNumJoints", lambda body, physicsClientId: len(names)
    )

    def joint_info(body, index, physicsClientId):
        row = [None] * 13
        row[12] = names[index].encode()
        return row

    monkeypatch.setattr(robot_controller.p, "getJointInfo", joint_info)


def test_duplicate_light_links_are_refused(monkeypatch):
    _patch_links(monkeypatch, ["light_emitter_link", "camera_link", "light_emitter_link"])

    with pytest.raises(ValueError, match="exactly one"):
        robot_controller.LampController._discover_light_link(_StubBody())


def test_a_missing_light_link_is_refused(monkeypatch):
    _patch_links(monkeypatch, ["camera_link", "speaker_link"])

    with pytest.raises(ValueError, match="exactly one"):
        robot_controller.LampController._discover_light_link(_StubBody())


def test_exactly_one_light_link_is_accepted(monkeypatch):
    _patch_links(monkeypatch, ["camera_link", "light_emitter_link", "speaker_link"])

    assert robot_controller.LampController._discover_light_link(_StubBody()) == 1


# --------------------------------------------------------------------------
# 11. Disengagement at every point in the goal lifecycle
# --------------------------------------------------------------------------


def test_leaving_before_the_first_look_lands_stops_the_goal_dead(capsys):
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    session = _inline_goal_session(brain)
    lamp = GoalLamp()
    session.request(MUG, 1)
    session.supply_frame(b"frame-one")

    engagement_demo._handle_goal(
        session.poll(), session, lamp, _GoalPlayer(), _NoFlush(), _Idle(), 1
    )
    capsys.readouterr()

    assert session.active is False
    assert len(brain.calls) == 1        # never looked again
    assert "look_left" not in lamp.calls
    assert lamp.light is False


def test_leaving_after_the_turn_but_before_verification_stops_the_goal_dead(capsys):
    brain = _LocatingBrain(sighting_json(), json.dumps(STILL_THERE))
    session = _inline_goal_session(brain)
    lamp = GoalLamp()
    session.request(MUG, 1)
    session.supply_frame(b"frame-one")
    engagement_demo._handle_goal(
        session.poll(), session, lamp, _GoalPlayer(), _NoFlush(), _Engaged(), 1
    )
    assert session.stage == STAGE_MOVED

    session.supply_frame(b"frame-two")
    engagement_demo._handle_goal(
        session.poll(), session, lamp, _GoalPlayer(), _NoFlush(), _Idle(), 1
    )
    capsys.readouterr()

    assert session.active is False
    assert "light_on" not in lamp.calls  # verified or not, they are gone
    assert lamp.light is False


def test_disengaging_during_the_closing_speech_still_puts_the_light_out(capsys):
    session = _active_session()
    session.hold("there it is")
    lamp = GoalLamp()
    lamp.light_on()

    engagement_demo._handle_goal(
        GoalOutcome(stage=STAGE_SPEAK, epoch=1, target="mug", playback=None),
        session,
        lamp,
        _GoalPlayer(),
        _NoFlush(),
        _Idle(),
        1,
    )
    capsys.readouterr()

    assert session.active is False
    assert lamp.light is False


def test_the_voice_turn_still_has_no_route_to_a_camera():
    source = (Path(__file__).resolve().parent.parent / "src" / "voice_turn.py").read_text()

    # Prose about observation is fine; a call that performs one is not.
    for forbidden in (
        ".locate(",
        ".observe(",
        "cv2",
        "VideoCapture",
        "encode_frame_jpeg",
        "import attention",
    ):
        assert forbidden not in source
