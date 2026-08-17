"""Spoken-turn tests: fake microphone, fake OpenAI client, fake lamp, fake speaker.

Nothing here opens a device, reads a credential or makes a paid API call. The
four collaborators of ``VoiceTurn`` are all injected, so the whole
    record -> STT -> character -> validate -> TTS -> move -> play
path runs in-process against stand-ins.

    python -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import audio_io  # noqa: E402
import character  # noqa: E402
import engagement_demo  # noqa: E402
import voice_turn  # noqa: E402
from character import (  # noqa: E402
    ALLOWED_BEHAVIORS,
    CharacterBrain,
    CharacterError,
    MissingCredentialsError,
    parse_character_reply,
    require_api_key,
)
from voice_turn import BEHAVIORS, TurnError, VoiceTurn  # noqa: E402

WAV = audio_io.encode_wav(b"\x00\x00" * 800)
SPEECH = audio_io.encode_wav(b"\x01\x00" * 800)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeRecorder:
    def __init__(self, payload: bytes = WAV, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def record(self) -> bytes:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.payload


class FakePlayer:
    """Two-phase player. ``prepare`` validates for real; ``play`` is a no-op."""

    def __init__(self, error: Exception | None = None, prepare_error: Exception | None = None):
        self.error = error
        self.prepare_error = prepare_error
        self.prepared: list[bytes] = []
        self.played: list[bytes] = []

    def prepare(self, wav_bytes: bytes):
        if self.prepare_error is not None:
            raise self.prepare_error
        # Uses the real decoder, so malformed audio fails here as it would live.
        _pcm, rate, channels, _width = audio_io.decode_wav(wav_bytes)
        self.prepared.append(wav_bytes)
        return audio_io.PreparedPlayback(
            samples=wav_bytes, sample_rate=rate, channels=channels, seconds=0.5
        )

    def play(self, prepared) -> None:
        if self.error is not None:
            raise self.error
        self.played.append(prepared.samples)


class FakeLamp:
    """Stands in for LampController and records the gestures it was asked for."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def nod(self, *_a, **_k) -> None:
        self.calls.append("nod")

    def engage(self, *_a, **_k) -> None:
        self.calls.append("engage")

    def neutral(self, *_a, **_k) -> None:
        self.calls.append("neutral")


class _Endpoint:
    """One fake SDK endpoint: returns a canned value or raises."""

    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


#: Response formats the gpt-4o-*-transcribe family actually accepts on a
#: non-streaming request. whisper-1's text/srt/vtt/verbose_json are rejected.
SUPPORTED_STT_FORMATS = {"json"}


class _TranscriptionEndpoint(_Endpoint):
    """Fake STT endpoint that enforces the real request contract.

    Deliberately strict: a permissive mock is what let an unsupported
    ``response_format`` ship green, so this one 400s exactly where the service
    would, and returns the ``json``-shaped object rather than a bare string.
    """

    def create(self, **kwargs):
        for required in ("model", "file", "response_format"):
            if required not in kwargs:
                raise TypeError(f"transcription request is missing '{required}'")
        response_format = kwargs["response_format"]
        if response_format not in SUPPORTED_STT_FORMATS:
            raise ValueError(
                f"400 response_format '{response_format}' is not compatible with "
                f"model '{kwargs['model']}'"
            )
        if not hasattr(kwargs["file"], "read"):
            raise TypeError("transcription 'file' must be a file-like object")
        result = super().create(**kwargs)
        return result


class _Transcription:
    """The parsed object the SDK returns for ``response_format='json'``."""

    def __init__(self, text: str):
        self.text = text


class _Response:
    def __init__(self, output_text: str):
        self.output_text = output_text


class _Speech:
    def __init__(self, content: bytes):
        self.content = content


def fake_client(
    transcript: str = "hello there",
    reply: str = "Hello! Nice to see you.",
    behavior: str = "none",
    raw: str | None = None,
    audio: bytes = SPEECH,
    stt_error: Exception | None = None,
    character_error: Exception | None = None,
    tts_error: Exception | None = None,
):
    """Assemble a stand-in for the OpenAI client with the surface we use."""
    payload = raw if raw is not None else json.dumps({"reply": reply, "behavior": behavior})

    class Client:
        pass

    client = Client()
    client.transcriptions = _TranscriptionEndpoint(
        result=_Transcription(transcript), error=stt_error
    )
    client.responses = _Endpoint(result=_Response(payload), error=character_error)
    client.speech = _Endpoint(result=_Speech(audio), error=tts_error)

    class Audio:
        transcriptions = client.transcriptions
        speech = client.speech

    client.audio = Audio()
    return client


def build_turn(lamp=None, recorder=None, player=None, **client_kwargs):
    lamp = lamp if lamp is not None else FakeLamp()
    recorder = recorder if recorder is not None else FakeRecorder()
    player = player if player is not None else FakePlayer()
    turn = VoiceTurn(
        brain=CharacterBrain(client=fake_client(**client_kwargs)),
        recorder=recorder,
        player=player,
        lamp=lamp,
    )
    return turn, lamp, recorder, player


# --------------------------------------------------------------------------
# 1. Happy path
# --------------------------------------------------------------------------


def test_happy_path_transcript_reply_behavior_playback():
    turn, lamp, recorder, player = build_turn(
        transcript="can you nod for me?",
        reply="Of course, here you go.",
        behavior="nod",
    )

    result = turn.run()

    assert recorder.calls == 1
    assert result.transcript == "can you nod for me?"
    assert result.reply == "Of course, here you go."
    assert result.behavior == "nod"
    assert result.moved is True
    assert lamp.calls == ["nod"]
    assert player.played == [SPEECH]


def test_the_configured_models_are_the_ones_actually_requested():
    turn, _lamp, _recorder, _player = build_turn()
    client = turn.brain._client
    turn.run()

    assert client.transcriptions.calls[0]["model"] == "gpt-4o-mini-transcribe"
    assert client.responses.calls[0]["model"] == "gpt-5.6-luna"
    assert client.speech.calls[0]["model"] == "tts-1"
    # WAV all the way through: no ffmpeg in the loop.
    assert client.speech.calls[0]["response_format"] == "wav"


def test_stt_uses_the_response_format_the_transcribe_models_support():
    """gpt-4o-*-transcribe rejects whisper-1's 'text'; we must ask for json."""
    turn, *_ = build_turn()
    turn.run()

    assert turn.brain._client.transcriptions.calls[0]["response_format"] == "json"
    assert character.STT_RESPONSE_FORMAT in SUPPORTED_STT_FORMATS


def test_the_mock_rejects_an_unsupported_stt_response_format(monkeypatch):
    """The fake endpoint must fail the contract the real service fails."""
    monkeypatch.setattr(character, "STT_RESPONSE_FORMAT", "text")
    turn, lamp, _recorder, player = build_turn(behavior="nod")

    with pytest.raises(TurnError, match="not compatible with model"):
        turn.run()

    assert lamp.calls == []
    assert player.played == []


def test_a_json_shaped_transcription_result_is_parsed():
    """The 'json' format returns an object with .text, not a bare string."""
    assert character._transcript_text(_Transcription("  hi there ")) == "hi there"
    assert character._transcript_text({"text": "hi"}) == "hi"
    assert character._transcript_text('{"text": "hi"}') == "hi"

    with pytest.raises(CharacterError, match="no 'text' field"):
        character._transcript_text(object())


def test_every_api_call_carries_an_explicit_timeout():
    turn, *_ = build_turn()
    client = turn.brain._client
    turn.run()

    for endpoint in (client.transcriptions, client.responses, client.speech):
        timeout = endpoint.calls[0]["timeout"]
        assert 0 < timeout <= 30, f"implausible timeout {timeout}"


# --------------------------------------------------------------------------
# 2 / 3. Behaviour dispatch
# --------------------------------------------------------------------------


def test_none_produces_no_robot_action():
    turn, lamp, _recorder, player = build_turn(behavior="none")

    result = turn.run()

    assert result.behavior == "none"
    assert result.moved is False
    assert lamp.calls == []
    # The reply is still spoken; only the body stays still.
    assert player.played == [SPEECH]


def test_nod_produces_exactly_one_nod():
    turn, lamp, _recorder, _player = build_turn(behavior="nod")

    turn.run()

    assert lamp.calls == ["nod"]


def test_engage_produces_exactly_one_engage():
    turn, lamp, _recorder, _player = build_turn(behavior="engage")

    turn.run()

    assert lamp.calls == ["engage"]


def test_dispatch_table_matches_the_contract_exactly():
    assert set(BEHAVIORS) == set(ALLOWED_BEHAVIORS) == {"none", "nod", "engage"}


def test_dispatch_is_a_static_allowlist_not_attribute_lookup():
    """A name the model invents must not reach the lamp even if it exists."""
    lamp = FakeLamp()
    assert hasattr(lamp, "neutral")  # a real, callable LampController method...
    assert "neutral" not in BEHAVIORS  # ...that is deliberately not dispatchable


# --------------------------------------------------------------------------
# 4. Invalid / unrecognized action fails closed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"reply": "sure", "behavior": "jump"}',           # outside the enum
        '{"reply": "sure", "behavior": "look_left"}',      # a real gesture, not allowed here
        '{"reply": "sure"}',                               # missing behavior
        '{"behavior": "nod"}',                             # missing reply
        '{"reply": "", "behavior": "nod"}',                # empty reply
        '{"reply": "sure", "behavior": "nod", "joint": "neck_yaw_joint"}',  # smuggled field
        '{"reply": "sure", "behavior": ["nod", "engage"]}',  # two behaviours
        "not json at all",
        "[]",
    ],
    ids=[
        "unknown-behavior",
        "unallowed-gesture",
        "missing-behavior",
        "missing-reply",
        "empty-reply",
        "extra-field",
        "two-behaviors",
        "not-json",
        "not-an-object",
    ],
)
def test_invalid_structured_response_fails_closed(raw):
    turn, lamp, _recorder, player = build_turn(raw=raw)

    with pytest.raises(TurnError):
        turn.run()

    assert lamp.calls == []
    assert player.played == []


def test_overlong_reply_is_rejected_locally():
    with pytest.raises(CharacterError, match="over the"):
        parse_character_reply(json.dumps({"reply": "a" * 5000, "behavior": "none"}))


def test_structured_output_is_requested_but_still_validated_locally():
    turn, *_ = build_turn()
    turn.run()
    text = turn.brain._client.responses.calls[0]["text"]

    assert text["format"]["type"] == "json_schema"
    assert text["format"]["schema"]["properties"]["behavior"]["enum"] == list(ALLOWED_BEHAVIORS)
    # ...and the local validator rejects a response the schema would have caught.
    with pytest.raises(CharacterError):
        parse_character_reply('{"reply": "x", "behavior": "spin"}')


# --------------------------------------------------------------------------
# 5. Credentials
# --------------------------------------------------------------------------


def test_missing_api_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingCredentialsError, match="OPENAI_API_KEY is not set"):
        require_api_key()


def test_blank_api_key_is_treated_as_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")

    with pytest.raises(MissingCredentialsError):
        require_api_key()


def test_brain_without_an_injected_client_requires_the_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingCredentialsError):
        CharacterBrain()


def test_demo_startup_refuses_to_launch_without_a_key(monkeypatch):
    """The key is checked before the camera or the GUI is touched."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["engagement_demo.py"])

    def fail(*_a, **_k):
        raise AssertionError("run() must not start without a credential")

    monkeypatch.setattr(engagement_demo, "run", fail)

    with pytest.raises(MissingCredentialsError):
        engagement_demo.main()


def test_no_voice_flag_starts_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["engagement_demo.py", "--no-voice", "--headless"])
    seen = {}

    monkeypatch.setattr(engagement_demo, "run", lambda **kwargs: seen.update(kwargs))
    engagement_demo.main()

    assert seen["voice_factory"] is None


SECRET = "sk-test-DO-NOT-LEAK-abcdef123456"


@pytest.mark.parametrize(
    "stage",
    ["stt_error", "character_error", "tts_error"],
)
def test_the_key_never_appears_in_a_turn_error(monkeypatch, stage):
    """A server error body that quotes the key back must not reach the user."""
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    leaky = RuntimeError(f"401 invalid_api_key: incorrect API key provided: {SECRET}")
    turn, lamp, _recorder, _player = build_turn(behavior="nod", **{stage: leaky})

    with pytest.raises(TurnError) as excinfo:
        turn.run()

    message = str(excinfo.value)
    assert SECRET not in message
    assert "<redacted>" in message
    assert lamp.calls == []
    assert require_api_key() == SECRET  # read and returned, never printed


def test_the_key_is_redacted_from_reported_turn_failures(monkeypatch, capsys):
    """The demo's own failure log is the last place it could escape."""
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    turn = RecordingTurn(error=TurnError(f"speech-to-text failed: bad key {SECRET}"))
    _run_demo([(True, 4.0)], ScriptedTalkKey(40), lambda _lamp: turn)
    captured = capsys.readouterr()

    assert SECRET not in captured.out
    assert SECRET not in captured.err
    assert "turn failed" in captured.err


def test_redaction_covers_key_shaped_tokens_we_did_not_issue(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    other = "sk-proj-someoneelseskey0000"

    assert other not in character.redact(f"upstream said {other} was bad")


def test_redaction_leaves_ordinary_messages_alone(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)

    assert character.redact("connection reset by peer") == "connection reset by peer"


# --------------------------------------------------------------------------
# 6 / 7 / 8. Stage failures cause no robot movement
# --------------------------------------------------------------------------


def test_stt_failure_causes_no_robot_movement():
    turn, lamp, _recorder, player = build_turn(
        behavior="nod", stt_error=RuntimeError("gateway timeout")
    )

    with pytest.raises(TurnError, match="speech-to-text failed"):
        turn.run()

    assert lamp.calls == []
    assert player.played == []


def test_empty_transcript_causes_no_robot_movement():
    turn, lamp, _recorder, player = build_turn(behavior="nod", transcript="   ")

    with pytest.raises(TurnError, match="no text"):
        turn.run()

    assert lamp.calls == []
    assert player.played == []


def test_character_api_failure_causes_no_robot_movement():
    turn, lamp, _recorder, player = build_turn(
        behavior="nod", character_error=RuntimeError("rate limited")
    )

    with pytest.raises(TurnError, match="character response failed"):
        turn.run()

    assert lamp.calls == []
    assert player.played == []


def test_tts_generation_failure_causes_no_robot_movement():
    """TTS runs *before* the gesture precisely so this case stays still."""
    turn, lamp, _recorder, player = build_turn(
        behavior="nod", tts_error=RuntimeError("service unavailable")
    )

    with pytest.raises(TurnError, match="text-to-speech failed"):
        turn.run()

    assert lamp.calls == []
    assert player.played == []


def test_malformed_tts_audio_fails_before_the_gesture():
    """Decoding happens in prepare(), so unplayable audio never reaches the lamp."""
    turn, lamp, _recorder, player = build_turn(behavior="nod", audio=b"not a wav file")

    with pytest.raises(TurnError, match="not a readable WAV stream"):
        turn.run()

    assert lamp.calls == []
    assert player.played == []


@pytest.mark.parametrize(
    "audio, expected",
    [
        (audio_io.encode_wav(b"")[:20], "not a readable WAV stream"),  # truncated header
        (audio_io.encode_wav(b""), "no audio frames"),                 # valid but empty
        (b"RIFF" + b"\x00" * 8, "not a readable WAV stream"),          # bogus container
    ],
    ids=["truncated", "empty", "bogus"],
)
def test_unplayable_audio_variants_fail_before_the_gesture(audio, expected):
    turn, lamp, _recorder, player = build_turn(behavior="nod", audio=audio)

    with pytest.raises(TurnError, match=expected):
        turn.run()

    assert lamp.calls == []
    assert player.played == []


def test_playback_preparation_failure_happens_before_the_gesture():
    """Device rejection, NumPy trouble and missing PortAudio all land here."""
    for error in (
        audio_io.AudioError("the default output device will not accept 24000 Hz"),
        ImportError("libportaudio2 not found"),
        ValueError("cannot reshape array"),
        EOFError(),
    ):
        player = FakePlayer(prepare_error=error)
        turn, lamp, _recorder, _player = build_turn(behavior="nod", player=player)

        with pytest.raises(TurnError):
            turn.run()

        assert lamp.calls == [], f"{type(error).__name__} moved the robot"
        assert player.played == []


def test_a_failing_behavior_is_a_failed_turn_not_a_dead_demo():
    """A LampController raising (dead simulator) must stay recoverable."""

    class BrokenLamp(FakeLamp):
        def nod(self, *_a, **_k):
            raise RuntimeError("Not connected to physics server")

    turn, _lamp, _recorder, player = build_turn(behavior="nod", lamp=BrokenLamp())

    with pytest.raises(TurnError, match="behavior failed"):
        turn.run()

    assert player.played == []


def test_empty_tts_audio_causes_no_robot_movement():
    turn, lamp, _recorder, player = build_turn(behavior="nod", audio=b"")

    with pytest.raises(TurnError, match="no audio"):
        turn.run()

    assert lamp.calls == []
    assert player.played == []


def test_microphone_failure_causes_no_robot_movement():
    recorder = FakeRecorder(error=audio_io.AudioError("no input device"))
    turn, lamp, _recorder, player = build_turn(behavior="nod", recorder=recorder)

    with pytest.raises(TurnError, match="recording failed"):
        turn.run()

    assert lamp.calls == []
    assert player.played == []


def test_speaker_failure_is_reported_after_the_gesture():
    """Playback is the only stage past the commit point; it cannot be undone."""
    player = FakePlayer(error=audio_io.AudioError("no output device"))
    turn, lamp, _recorder, _player = build_turn(behavior="nod", player=player)

    with pytest.raises(TurnError, match="playback failed"):
        turn.run()

    assert lamp.calls == ["nod"]


# --------------------------------------------------------------------------
# 9. Repeatability
# --------------------------------------------------------------------------


def test_repeated_turns_work():
    turn, lamp, recorder, player = build_turn(behavior="nod")

    results = [turn.run() for _ in range(3)]

    assert recorder.calls == 3
    assert [r.behavior for r in results] == ["nod", "nod", "nod"]
    assert lamp.calls == ["nod", "nod", "nod"]
    assert len(player.played) == 3


def test_a_failed_turn_does_not_poison_the_next_one():
    lamp = FakeLamp()
    player = FakePlayer()
    recorder = FakeRecorder()
    brain = CharacterBrain(client=fake_client(raw='{"reply": "hi", "behavior": "jump"}'))
    turn = VoiceTurn(brain=brain, recorder=recorder, player=player, lamp=lamp)

    with pytest.raises(TurnError):
        turn.run()
    assert lamp.calls == []

    turn.brain = CharacterBrain(client=fake_client(behavior="nod"))
    assert turn.run().behavior == "nod"
    assert lamp.calls == ["nod"]


# --------------------------------------------------------------------------
# WAV plumbing (no hardware)
# --------------------------------------------------------------------------


def test_wav_round_trip_preserves_the_capture_format():
    pcm = b"\x01\x02" * 500
    payload, rate, channels, width = audio_io.decode_wav(audio_io.encode_wav(pcm))

    assert payload == pcm
    assert (rate, channels, width) == (audio_io.SAMPLE_RATE, audio_io.CHANNELS, 2)


def test_decoding_a_non_wav_buffer_is_an_audio_error():
    with pytest.raises(audio_io.AudioError):
        audio_io.decode_wav(b"this is not audio")


def test_recorded_utterance_is_bounded():
    assert 1.0 <= audio_io.RECORD_SECONDS <= 10.0


# --------------------------------------------------------------------------
# Integration with the Hour 2 engagement loop
# --------------------------------------------------------------------------


class ScriptedSensor:
    """Replays a fixed attention pattern at DETECT_HZ, then stops the loop.

    *on_frame* is called with the frame index before each read, which is how
    the concurrency tests release a blocked worker at a chosen point in the
    engagement script.
    """

    def __init__(self, script, on_frame=None):
        from attention import AttentionReading, DETECT_HZ

        self._reading = AttentionReading
        self._frames = [
            attending
            for attending, seconds in script
            for _ in range(int(round(seconds * DETECT_HZ)))
        ]
        self._index = 0
        self._on_frame = on_frame
        self.flushes = 0

    @property
    def frames_read(self) -> int:
        return self._index

    def read(self):
        if self._index >= len(self._frames):
            raise StopIteration
        if self._on_frame is not None:
            self._on_frame(self._index)
        attending = self._frames[self._index]
        self._index += 1
        return self._reading(attending=attending, faces=(), frame=None)

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        pass


class ScriptedTalkKey:
    """Reports a keypress on the given loop iterations."""

    def __init__(self, *frames: int):
        self.frames = set(frames)
        self.index = -1

    def pressed(self) -> bool:
        self.index += 1
        return self.index in self.frames


def _run_demo(script, talk_key, turn_factory, sensor=None):
    turns = []

    def factory(lamp):
        turn = turn_factory(lamp)
        turns.append(turn)
        return turn

    engagement_demo.run(
        preview=False,
        headless=True,
        sensor=sensor if sensor is not None else ScriptedSensor(script),
        voice_factory=factory,
        talk_key=talk_key,
    )
    return turns


class RecordingTurn:
    """A VoiceTurn stand-in that logs when the loop asked it to run.

    *gate* is an optional ``threading.Event`` ``prepare`` waits on, which is
    how a test holds a turn "in flight" while the engagement loop carries on.
    """

    def __init__(self, behavior="nod", error=None, gate=None, lamp=None):
        self.behavior = behavior
        self.error = error
        self.gate = gate
        self.lamp = lamp
        self.prepares = 0
        self.commits = 0
        self.entered = threading.Event()

    def prepare(self):
        self.prepares += 1
        self.entered.set()
        if self.gate is not None and not self.gate.wait(timeout=10):
            raise TurnError("test gate timed out")
        if self.error is not None:
            raise self.error
        return voice_turn.PreparedTurn(
            transcript="hello",
            reply="hi there",
            behavior=self.behavior,
            playback=audio_io.PreparedPlayback(
                samples=b"", sample_rate=24_000, channels=1, seconds=0.0
            ),
        )

    def commit(self, prepared):
        self.commits += 1
        if self.lamp is not None:
            voice_turn.BEHAVIORS[prepared.behavior](self.lamp)
        return voice_turn.TurnResult(
            transcript=prepared.transcript,
            reply=prepared.reply,
            behavior=prepared.behavior,
            moved=prepared.behavior != "none",
            speech_seconds=0.0,
        )

    #: Older name kept for the synchronous assertions below.
    @property
    def runs(self) -> int:
        return self.commits


def test_a_spoken_turn_only_happens_once_engaged(capsys):
    # Frame 0 is IDLE (engagement needs ~0.8 s of attention); frame 40 is well
    # past the ENGAGED transition.
    turn = RecordingTurn()
    _run_demo([(True, 4.0)], ScriptedTalkKey(0, 40), lambda _lamp: turn)
    out = capsys.readouterr().out

    assert turn.runs == 1
    assert "not engaged yet" in out
    assert 'heard: "hello"' in out


def test_repeated_spoken_turns_and_engagement_keep_working(capsys):
    turn = RecordingTurn()
    # Frames 30 and 40 land while ENGAGED; frame 90 lands after the character
    # has disengaged, so that press must be refused.
    _run_demo([(True, 3.0), (False, 4.0), (True, 3.0)], ScriptedTalkKey(30, 40, 90),
              lambda _lamp: turn)
    out = capsys.readouterr().out
    transitions = [line.split("] ")[-1] for line in out.splitlines() if " -> " in line]

    assert turn.runs == 2
    # The camera state machine is untouched by the spoken turns.
    assert transitions == ["IDLE -> ENGAGED", "ENGAGED -> IDLE", "IDLE -> ENGAGED"]


def test_a_failed_turn_is_reported_and_the_loop_survives(capsys):
    turn = RecordingTurn(error=TurnError("text-to-speech failed: ServiceUnavailable"))
    _run_demo([(True, 4.0)], ScriptedTalkKey(40), lambda _lamp: turn)
    captured = capsys.readouterr()

    assert "turn failed: text-to-speech failed" in captured.err
    # Engagement still ran to completion afterwards.
    assert "IDLE -> ENGAGED" in captured.out


def test_the_camera_only_demo_never_builds_a_voice_path(capsys):
    engagement_demo.run(preview=False, headless=True, sensor=ScriptedSensor([(True, 2.0)]))

    assert "press Enter" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# The voice turn must not block the engagement loop
# --------------------------------------------------------------------------


def test_slow_voice_work_does_not_block_the_engagement_loop(capsys):
    """A turn stuck in prepare() must not stop the camera or the tracker.

    The worker is held inside prepare() from frame ~30 until frame 120. If the
    loop were still synchronous, no frames would be read and no transition
    could be logged in that window; instead the character notices the person
    leaving and disengages while it is still thinking.
    """
    gate = threading.Event()
    turn = RecordingTurn(gate=gate)
    sensor = ScriptedSensor(
        [(True, 3.0), (False, 5.0), (True, 2.0)],
        # Release the worker long after disengagement should have happened.
        on_frame=lambda i: gate.set() if i == 120 else None,
    )

    _run_demo(None, ScriptedTalkKey(30), lambda _lamp: turn, sensor=sensor)
    out = capsys.readouterr().out
    transitions = [line.split("] ")[-1] for line in out.splitlines() if " -> " in line]

    assert turn.prepares == 1
    assert turn.entered.is_set()
    # The whole script was consumed even though a turn was pending for ~90
    # frames, and engagement kept moving underneath it.
    assert sensor.frames_read == 150
    assert transitions == ["IDLE -> ENGAGED", "ENGAGED -> IDLE", "IDLE -> ENGAGED"]


def test_a_stale_turn_is_discarded_instead_of_gesturing_late(capsys):
    """Disengaging mid-turn must not produce a gesture when the API returns."""
    gate = threading.Event()
    lamp = FakeLamp()
    turn = RecordingTurn(gate=gate, lamp=lamp)
    sensor = ScriptedSensor(
        [(True, 3.0), (False, 6.0)],
        on_frame=lambda i: gate.set() if i == 110 else None,
    )

    _run_demo(None, ScriptedTalkKey(30), lambda _lamp: turn, sensor=sensor)
    captured = capsys.readouterr()

    assert turn.prepares == 1
    assert turn.commits == 0, "a stale turn must never be committed"
    assert lamp.calls == []
    assert "discarded a stale turn" in captured.out


def test_a_turn_that_outlives_its_engagement_is_stale_even_if_re_engaged(capsys):
    """Leaving and coming back starts a new conversation, not a late answer."""
    gate = threading.Event()
    lamp = FakeLamp()
    turn = RecordingTurn(gate=gate, lamp=lamp)
    # Engage, ask, leave (disengage), come back and re-engage; only then does
    # the pending API call return.
    sensor = ScriptedSensor(
        [(True, 3.0), (False, 5.0), (True, 3.0)],
        on_frame=lambda i: gate.set() if i == 150 else None,
    )

    _run_demo(None, ScriptedTalkKey(30), lambda _lamp: turn, sensor=sensor)
    out = capsys.readouterr().out

    assert turn.commits == 0
    assert lamp.calls == []
    assert "discarded a stale turn" in out


def test_a_second_press_while_a_turn_is_in_flight_is_refused(capsys):
    gate = threading.Event()
    turn = RecordingTurn(gate=gate)
    sensor = ScriptedSensor(
        [(True, 6.0)],
        on_frame=lambda i: gate.set() if i == 70 else None,
    )

    _run_demo(None, ScriptedTalkKey(30, 40, 50), lambda _lamp: turn, sensor=sensor)
    out = capsys.readouterr().out

    assert turn.prepares == 1
    assert out.count("still working on the last thing you said.") == 2


# --------------------------------------------------------------------------
# VoiceSession in isolation
# --------------------------------------------------------------------------


def _inline_session(turn):
    """A session whose 'worker' runs inline, for deterministic assertions."""
    return voice_turn.VoiceSession(turn, spawn=lambda work: work())


def test_session_reports_a_prepared_turn_with_its_epoch():
    turn, *_ = build_turn(behavior="nod")
    session = _inline_session(turn)

    assert session.start(epoch=7, now=0.0) is True
    outcome = session.poll()

    assert outcome.epoch == 7
    assert outcome.error is None
    assert outcome.prepared.behavior == "nod"
    assert session.busy is False


def test_session_turns_a_failed_prepare_into_an_outcome_not_an_exception():
    turn, lamp, _recorder, _player = build_turn(raw='{"reply": "x", "behavior": "spin"}')
    session = _inline_session(turn)
    session.start(epoch=1, now=0.0)
    outcome = session.poll()

    assert isinstance(outcome.error, TurnError)
    assert outcome.prepared is None
    assert lamp.calls == []


def test_session_survives_a_worker_that_raises_something_unexpected():
    class Exploding:
        def prepare(self):
            raise KeyboardInterrupt("worker interrupted")

    session = _inline_session(Exploding())
    session.start(epoch=1, now=0.0)
    outcome = session.poll()

    assert isinstance(outcome.error, TurnError)
    assert "voice worker crashed" in str(outcome.error)
    assert session.busy is False


def test_session_refuses_to_start_while_the_character_is_still_speaking():
    turn, *_ = build_turn()
    session = _inline_session(turn)
    session.start(epoch=1, now=0.0)
    session.poll()
    session.mark_speaking(3.0, now=10.0)

    assert session.ready(now=11.0) is False
    assert session.start(epoch=2, now=11.0) is False
    assert session.ready(now=13.5) is True


def test_session_refuses_to_start_while_a_result_is_undrained():
    turn, *_ = build_turn()
    session = _inline_session(turn)
    session.start(epoch=1, now=0.0)

    assert session.start(epoch=2, now=0.0) is False  # result not yet polled
    session.poll()
    assert session.start(epoch=3, now=0.0) is True


def test_the_worker_half_of_a_turn_never_touches_the_robot():
    """prepare() is what runs off-thread, so it must not reach LampController."""
    turn, lamp, _recorder, player = build_turn(behavior="engage")

    prepared = turn.prepare()

    assert lamp.calls == []
    assert player.played == []
    assert prepared.behavior == "engage"

    turn.commit(prepared)
    assert lamp.calls == ["engage"]
