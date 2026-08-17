"""Language layer: transcript in, {spoken reply, semantic behaviour} out.

This module owns every call to OpenAI and nothing else. It never imports
PyBullet, never sees a joint, and never learns that joints exist -- the only
robot vocabulary it knows is the three-word enum in :data:`ALLOWED_BEHAVIORS`.
Turning one of those words into motion is the job of ``voice_turn`` and
``robot_controller``.

The boundary matters for safety: the model cannot emit an angle, a joint name,
a pose or a tool call, because there is no field in the response contract that
could carry one. The worst a compromised or confused model can do is pick the
wrong word out of three, and every one of those three is a pre-authored,
limit-clamped gesture.

    transcript --> [gpt-5.6-luna] --> {reply: str, behavior: none|nod|engage}

Structured output is requested from the Responses API, but the result is still
validated locally (:func:`parse_character_reply`). Model-side schema
enforcement is a convenience, not a trust boundary.
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

STT_MODEL = "gpt-4o-mini-transcribe"
CHARACTER_MODEL = "gpt-5.6-luna"
TTS_MODEL = "tts-1"

#: The transcription models in the gpt-4o-*-transcribe family only support the
#: ``json`` response format on non-streaming requests -- unlike whisper-1, they
#: reject ``text``, ``srt``, ``vtt`` and ``verbose_json``. So we ask for JSON
#: and read the ``text`` field off the parsed result.
STT_RESPONSE_FORMAT = "json"

#: A neutral, unhurried voice that suits a small desk lamp.
TTS_VOICE = "alloy"

# -- timeouts ---------------------------------------------------------------
#
# The SDK default is minutes long, which for a face-to-face interaction means
# the character appears to freeze. Each stage gets a budget a person would
# still tolerate standing in front of the robot, and one retry -- enough to
# ride out a single dropped connection, not enough to double the wait twice.

STT_TIMEOUT_SECONDS = 20.0
CHARACTER_TIMEOUT_SECONDS = 20.0
TTS_TIMEOUT_SECONDS = 20.0
MAX_RETRIES = 1

#: Ceiling on the spoken reply. Two short sentences fit comfortably; the cap is
#: a backstop against a monologue, not the primary control (the prompt is).
MAX_OUTPUT_TOKENS = 300

#: Hard local cap on what we are willing to send to the speech endpoint.
MAX_REPLY_CHARS = 400


# --------------------------------------------------------------------------
# The action contract
# --------------------------------------------------------------------------

#: Every semantic behaviour the character may request this milestone. These are
#: names of existing LampController gestures, deliberately not parameters.
ALLOWED_BEHAVIORS = ("none", "nod", "engage")

SYSTEM_PROMPT = (
    "You are the voice of a small desk lamp robot with an expressive, movable head. "
    "You are warm, curious and brief. Reply to what the person just said in one or two "
    "short spoken sentences of plain speech: no emoji, no markdown, no stage directions, "
    "no lists. Then pick at most one physical behaviour: 'nod' to acknowledge or agree, "
    "'engage' to perk up and lean toward the person, or 'none' when no movement is needed. "
    "Prefer 'none' unless a gesture genuinely adds something. "
    "You do not remember earlier conversations and should not pretend to."
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reply": {
            "type": "string",
            "description": "One or two short sentences to speak aloud.",
        },
        "behavior": {
            "type": "string",
            "enum": list(ALLOWED_BEHAVIORS),
            "description": "At most one physical gesture, or 'none'.",
        },
    },
    "required": ["reply", "behavior"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class CharacterReply:
    """A validated turn: what to say, and the single gesture to play."""

    reply: str
    behavior: str


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class CharacterError(RuntimeError):
    """Any failure in the language/speech path. Always fails a turn closed.

    Redacts its own message on construction, so there is no path by which a
    credential reaches a caller through this type -- callers cannot forget.
    """

    def __init__(self, message: str = ""):
        super().__init__(redact(str(message)))


class MissingCredentialsError(CharacterError):
    """OPENAI_API_KEY is absent or empty."""


def require_api_key() -> str:
    """Return the API key from the environment, or explain how to set it.

    The key is read here and nowhere else, is never logged, and is never
    included in an exception message.
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise MissingCredentialsError(
            "OPENAI_API_KEY is not set. Export it before starting the demo "
            "(export OPENAI_API_KEY=...), or run with --no-voice to use the "
            "camera-only engagement demo."
        )
    return key


# --------------------------------------------------------------------------
# Local validation
# --------------------------------------------------------------------------


def parse_character_reply(raw: str) -> CharacterReply:
    """Validate a model response against the contract. Raises on anything odd.

    Applied even when the model was asked for schema-constrained output: this
    is the check that actually gates robot motion, so it does not delegate.
    """
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


# --------------------------------------------------------------------------
# OpenAI-backed implementation
# --------------------------------------------------------------------------


def build_client(api_key: Optional[str] = None):
    """Construct the official OpenAI SDK client with interactive-grade limits."""
    try:
        from openai import OpenAI  # noqa: PLC0415  (lazy: keeps tests hermetic)
    except ImportError as exc:
        raise CharacterError(
            f"the 'openai' package is not installed ({exc}); install it from requirements.txt."
        ) from exc
    return OpenAI(api_key=api_key or require_api_key(), max_retries=MAX_RETRIES)


class CharacterBrain:
    """Speech-to-text, character reasoning and text-to-speech, in three calls.

    *client* is injected so unit tests can pass a fake with the same surface;
    left None it builds a real SDK client and therefore requires the key.
    """

    def __init__(self, client=None):
        self._client = client if client is not None else build_client()

    # -- stage 1: hearing --------------------------------------------------

    def transcribe(self, wav_bytes: bytes) -> str:
        """Turn one recorded utterance into text."""
        # A named file-like object is what the SDK's multipart encoder wants;
        # the name is only used to infer the content type.
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

    # -- stage 2: thinking -------------------------------------------------

    def respond(self, transcript: str) -> CharacterReply:
        """Answer *transcript* in character and choose zero or one gesture."""
        try:
            response = self._client.responses.create(
                model=CHARACTER_MODEL,
                instructions=SYSTEM_PROMPT,
                input=transcript,
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
        # Validated locally regardless of the schema request above.
        return parse_character_reply(raw)

    # -- stage 3: speaking -------------------------------------------------

    def synthesize(self, reply: str) -> bytes:
        """Render *reply* as a WAV buffer ready for the speaker."""
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
    """Pull the transcript out of a ``json``-format transcription response.

    The SDK normally returns a parsed ``Transcription`` model; a plain dict or
    a raw JSON string are accepted too so that a stubbed client or a future
    SDK shape does not crash the turn.
    """
    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    if text is None and isinstance(result, str):
        # Tolerated, not requested: a JSON body the SDK did not parse.
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
    """Pull bytes out of whichever binary-response wrapper the SDK returned."""
    for attribute in ("content", "read"):
        value = getattr(response, attribute, None)
        if callable(value):
            return value()
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
    raise CharacterError("text-to-speech response contained no audio payload")


# --------------------------------------------------------------------------
# Credential hygiene
# --------------------------------------------------------------------------

#: Anything shaped like an OpenAI secret, whatever its origin. Used as a
#: backstop for keys we did not issue (proxy tokens, a key echoed back inside a
#: server error body) alongside the exact-value check below.
_KEY_PATTERN = re.compile(r"\b(sk-[A-Za-z0-9_\-]{8,})")

_REDACTED = "<redacted>"


def redact(text: str) -> str:
    """Strip the API key -- exact value and key-shaped tokens -- from *text*.

    Applied to every message that can reach a terminal, a log or an exception
    the user will see. The environment is read fresh each call rather than
    cached, so a key rotated mid-run is still covered.
    """
    if not text:
        return text
    secret = os.environ.get("OPENAI_API_KEY", "").strip()
    if secret and secret in text:
        text = text.replace(secret, _REDACTED)
    return _KEY_PATTERN.sub(_REDACTED, text)


def _safe(exc: BaseException) -> str:
    """Render an SDK exception for logging without leaking anything.

    Two hazards, both handled here. SDK exceptions carry a ``request`` object
    whose headers include the Authorization bearer token, so only the type name
    and first message line are ever used -- never ``repr()``. And a server
    error body can quote the offending key straight back at us, so whatever
    survives is passed through :func:`redact`.
    """
    try:
        message = str(exc).splitlines()[0] if str(exc) else ""
    except Exception:  # a pathological __str__ must not break error reporting
        message = ""
    name = type(exc).__name__
    return redact(f"{name}: {message}" if message else name)
