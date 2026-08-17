"""OpenAI-backed language, speech, and bounded visual observation layer.

This is the only module that talks to OpenAI. Model output remains semantic:
spoken text, one allowlisted named behaviour, or one structured scene
observation. It never imports PyBullet and never sees robot joints.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from scene_memory import SceneObservation

STT_MODEL = "gpt-4o-mini-transcribe"
CHARACTER_MODEL = "gpt-5.6-luna"
TTS_MODEL = "tts-1"
STT_RESPONSE_FORMAT = "json"
TTS_VOICE = "alloy"

STT_TIMEOUT_SECONDS = 20.0
CHARACTER_TIMEOUT_SECONDS = 20.0
TTS_TIMEOUT_SECONDS = 20.0
MAX_RETRIES = 1
MAX_OUTPUT_TOKENS = 300
MAX_REPLY_CHARS = 400
MAX_IMAGE_BYTES = 2_000_000

ALLOWED_BEHAVIORS = ("none", "nod", "engage")

SYSTEM_PROMPT = (
    "You are the voice of a small desk lamp robot with an expressive, movable head. "
    "You are warm, curious and brief. Reply to what the person just said in one or two "
    "short spoken sentences of plain speech: no emoji, no markdown, no stage directions, "
    "no lists.\n\n"
    "If RETAINED SCENE MEMORY is supplied, it contains facts from an earlier camera "
    "observation. Treat those facts as memory, not as a live camera view. Use only those "
    "facts for questions about the remembered object; do not invent unseen details. If no "
    "memory is supplied, do not pretend you remember an object.\n\n"
    "Then choose the 'behavior' field. The default is 'none', and 'none' is the correct "
    "answer for the large majority of turns. Moving is not how you are polite: your voice "
    "already carries warmth, interest and acknowledgement, so a gesture adds nothing to an "
    "ordinary reply. A real lamp that bobbed its head at every sentence would read as "
    "twitchy rather than alive. Gestures land only when they are rare.\n"
    "Choose 'nod' ONLY when one of these is true:\n"
    "  - the person explicitly asks for the movement ('can you nod?', 'nod if you can hear "
    "me', 'move your head');\n"
    "  - your reply is a direct yes or a genuine agreement to something they asked or "
    "proposed ('yes, that's right', 'agreed').\n"
    "Choose 'engage' ONLY when the person is asking for your attention itself -- calling you "
    "over, asking you to look at them, or checking whether you are listening.\n"
    "In every other case choose 'none'. In particular, choose 'none' for greetings, for "
    "questions you answer with information, for small talk, for compliments, for thanks, and "
    "for anything you are merely acknowledging. If you are unsure, choose 'none'."
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reply": {"type": "string", "description": "One or two short sentences to speak aloud."},
        "behavior": {
            "type": "string",
            "enum": list(ALLOWED_BEHAVIORS),
            "description": (
                "At most one physical gesture. Defaults to 'none', which is correct for most "
                "turns. 'nod' only for an explicit request to move or a direct yes/agreement; "
                "'engage' only when the person is asking for your attention itself."
            ),
        },
    },
    "required": ["reply", "behavior"],
    "additionalProperties": False,
}

OBSERVATION_PROMPT = (
    "Inspect this single current laptop-camera frame for the scene-memory demo. "
    "Choose exactly one salient physical desk object that is clearly visible. Ignore people, "
    "body parts, the robot/simulator, screens, and background furniture. Return only simple "
    "visible facts: object type, visible color, rough location in the image (left/center/right "
    "and near/far if useful), and one other short visible detail. Use 'unknown' rather than "
    "guessing. Do not infer ownership, hidden contents, brand, material, or purpose unless it "
    "is visually unambiguous."
)

OBSERVATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "object_type": {"type": "string"},
        "color": {"type": "string"},
        "relative_location": {"type": "string"},
        "detail": {"type": "string"},
    },
    "required": ["object_type", "color", "relative_location", "detail"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class CharacterReply:
    reply: str
    behavior: str


class CharacterError(RuntimeError):
    def __init__(self, message: str = ""):
        super().__init__(redact(str(message)))


class MissingCredentialsError(CharacterError):
    pass


def require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise MissingCredentialsError(
            "OPENAI_API_KEY is not set. Export it before starting the demo "
            "(export OPENAI_API_KEY=...), or run with --no-voice to use the "
            "camera-only engagement demo."
        )
    return key


def parse_character_reply(raw: str) -> CharacterReply:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise CharacterError(f"character response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CharacterError("character response was not a JSON object")
    unexpected = set(payload) - {"reply", "behavior"}
    if unexpected:
        raise CharacterError(f"character response has unexpected fields: {sorted(unexpected)}")
    reply = payload.get("reply")
    behavior = payload.get("behavior")
    if not isinstance(reply, str) or not reply.strip():
        raise CharacterError("character response has no usable 'reply' text")
    if len(reply) > MAX_REPLY_CHARS:
        raise CharacterError(
            f"character reply is {len(reply)} characters, over the {MAX_REPLY_CHARS} limit"
        )
    if behavior not in ALLOWED_BEHAVIORS:
        raise CharacterError(
            f"character response requested unknown behavior {behavior!r}; "
            f"allowed: {list(ALLOWED_BEHAVIORS)}"
        )
    return CharacterReply(reply=reply.strip(), behavior=behavior)


def parse_scene_observation(raw: str) -> SceneObservation:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise CharacterError(f"visual observation was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CharacterError("visual observation was not a JSON object")
    expected = {"object_type", "color", "relative_location", "detail"}
    unexpected = set(payload) - expected
    missing = expected - set(payload)
    if unexpected or missing:
        raise CharacterError(
            f"visual observation fields invalid; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    try:
        return SceneObservation(**payload)
    except (TypeError, ValueError) as exc:
        raise CharacterError(f"visual observation was unusable: {exc}") from exc


def build_client(api_key: Optional[str] = None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise CharacterError(
            f"the 'openai' package is not installed ({exc}); install it from requirements.txt."
        ) from exc
    return OpenAI(api_key=api_key or require_api_key(), max_retries=MAX_RETRIES)


class CharacterBrain:
    def __init__(self, client=None):
        self._client = client if client is not None else build_client()

    def transcribe(self, wav_bytes: bytes) -> str:
        upload = io.BytesIO(wav_bytes)
        upload.name = "utterance.wav"
        try:
            result = self._client.audio.transcriptions.create(
                model=STT_MODEL,
                file=upload,
                response_format=STT_RESPONSE_FORMAT,
                timeout=STT_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise CharacterError(f"speech-to-text failed: {_safe(exc)}") from exc
        text = _transcript_text(result)
        if not text:
            raise CharacterError("speech-to-text returned no text")
        return text

    def respond(self, transcript: str, scene_context: Optional[str] = None) -> CharacterReply:
        """Answer one utterance, optionally grounded in earlier retained scene facts."""
        user_input = transcript
        if scene_context:
            user_input = f"{transcript}\n\nRETAINED SCENE MEMORY:\n{scene_context}"
        try:
            response = self._client.responses.create(
                model=CHARACTER_MODEL,
                instructions=SYSTEM_PROMPT,
                input=user_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "lamp_turn",
                        "strict": True,
                        "schema": RESPONSE_SCHEMA,
                    }
                },
                max_output_tokens=MAX_OUTPUT_TOKENS,
                timeout=CHARACTER_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise CharacterError(f"character response failed: {_safe(exc)}") from exc
        raw = getattr(response, "output_text", None)
        if not isinstance(raw, str) or not raw.strip():
            raise CharacterError("character response was empty")
        return parse_character_reply(raw)

    def observe(self, jpeg_bytes: bytes) -> SceneObservation:
        """Turn one earlier camera frame into a bounded semantic observation."""
        if not isinstance(jpeg_bytes, (bytes, bytearray)) or not jpeg_bytes:
            raise CharacterError("visual observation received no image bytes")
        if len(jpeg_bytes) > MAX_IMAGE_BYTES:
            raise CharacterError(
                f"visual observation image is {len(jpeg_bytes)} bytes, over the "
                f"{MAX_IMAGE_BYTES} byte limit"
            )
        image_url = "data:image/jpeg;base64," + base64.b64encode(bytes(jpeg_bytes)).decode("ascii")
        try:
            response = self._client.responses.create(
                model=CHARACTER_MODEL,
                instructions=OBSERVATION_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Observe one useful desk object."},
                            {"type": "input_image", "image_url": image_url},
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "scene_observation",
                        "strict": True,
                        "schema": OBSERVATION_SCHEMA,
                    }
                },
                max_output_tokens=180,
                timeout=CHARACTER_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise CharacterError(f"visual observation failed: {_safe(exc)}") from exc
        raw = getattr(response, "output_text", None)
        if not isinstance(raw, str) or not raw.strip():
            raise CharacterError("visual observation was empty")
        return parse_scene_observation(raw)

    def synthesize(self, reply: str) -> bytes:
        try:
            speech = self._client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=reply,
                response_format="wav",
                timeout=TTS_TIMEOUT_SECONDS,
            )
            audio = speech if isinstance(speech, (bytes, bytearray)) else _read_audio(speech)
        except CharacterError:
            raise
        except Exception as exc:
            raise CharacterError(f"text-to-speech failed: {_safe(exc)}") from exc
        if not audio:
            raise CharacterError("text-to-speech returned no audio")
        return bytes(audio)


def _transcript_text(result) -> str:
    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    if text is None and isinstance(result, str):
        try:
            payload = json.loads(result)
        except ValueError as exc:
            raise CharacterError(
                f"speech-to-text returned an unparseable body: {_safe(exc)}"
            ) from exc
        if not isinstance(payload, dict):
            raise CharacterError("speech-to-text returned a body with no 'text' field")
        text = payload.get("text")
    if not isinstance(text, str):
        raise CharacterError("speech-to-text response had no 'text' field")
    return text.strip()


def _read_audio(response) -> bytes:
    for attribute in ("content", "read"):
        value = getattr(response, attribute, None)
        if callable(value):
            return value()
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
    raise CharacterError("text-to-speech response contained no audio payload")


_KEY_PATTERN = re.compile(r"\b(sk-[A-Za-z0-9_\-]{8,})")
_REDACTED = "<redacted>"


def redact(text: str) -> str:
    if not text:
        return text
    secret = os.environ.get("OPENAI_API_KEY", "").strip()
    if secret and secret in text:
        text = text.replace(secret, _REDACTED)
    return _KEY_PATTERN.sub(_REDACTED, text)


def _safe(exc: BaseException) -> str:
    try:
        message = str(exc).splitlines()[0] if str(exc) else ""
    except Exception:
        message = ""
    name = type(exc).__name__
    return redact(f"{name}: {message}" if message else name)
