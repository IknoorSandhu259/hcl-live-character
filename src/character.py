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

Which calls carry a picture, and which do not
---------------------------------------------
:meth:`CharacterBrain.observe` and :meth:`CharacterBrain.locate` are the only
methods in the system that send an image, and they answer different questions.
``observe`` happens once, when the person asks the character to look, and its
whole output is a five-field record that ``scene_memory`` validates and stores.
``locate`` asks about one *named* target -- "is the mug here, and on which
side?" -- and its three-field answer is never stored: it is fresh evidence for
one goal, used twice within that goal and then discarded.

:meth:`CharacterBrain.respond` answers a spoken question and takes the retained
facts as *text*. It has no image parameter and no camera access, so a question
about the mug is answered from the note, never by looking again -- which is
what makes the answer survive the mug being moved or taken away.

The turn's response also carries a ``goal``: either nothing, or the single
``find_and_light(target)`` job. That is a separate contract from ``behavior``,
validated by ``goal.parse_goal``, and it is a *request* -- the robot does not
move for it until fresh perception says where to move.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from goal import (
    GOAL_KINDS,
    MAX_TARGET_CHARS,
    NO_GOAL,
    Goal,
    GoalError,
    TargetSighting,
    normalize_target,
    parse_goal,
    parse_sighting,
)
from scene_memory import (
    LOCATIONS,
    MAX_COLOR_CHARS,
    MAX_DETAIL_CHARS,
    MAX_NAME_CHARS,
    SceneObservation,
    parse_observation,
)

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

STT_MODEL = "gpt-4o-mini-transcribe"
CHARACTER_MODEL = "gpt-5.6-luna"
TTS_MODEL = "tts-1"

#: The model asked to look at one camera frame. The same Responses-API model as
#: the character voice by default -- one fewer moving part, and the reply it has
#: to produce is five short fields, not an essay. Overridable from the
#: environment so a deployment whose character model is text-only can point the
#: single vision call somewhere else without touching the code.
VISION_MODEL = os.environ.get("LAMP_VISION_MODEL", "").strip() or CHARACTER_MODEL

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
#: Looking is a deliberate, announced pause ("let me have a look"), so it gets a
#: little longer than a conversational turn -- an image upload is involved.
OBSERVE_TIMEOUT_SECONDS = 25.0
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
    "no lists. "
    "\n\n"
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
    "\n\n"
    "You do not remember earlier conversations and should not pretend to.\n"
    "You may, however, be given a block headed 'SCENE MEMORY' before the person's "
    "words. That is your own note about one object you looked at earlier through your "
    "camera. Treat it as the only thing you know about the scene. You are not looking "
    "now, so speak about it in the past tense ('it was a white mug, over on my left'). "
    "If the note is absent, or does not contain what they asked for, say plainly that "
    "you did not note that -- never guess a colour, a position or a detail, and never "
    "claim to be looking at something right now."
    "\n\n"
    "Finally, and entirely separately from 'behavior', set the 'goal' field. There is "
    "exactly one job you can take on: finding a named object in front of you right now and "
    "shining your light on it. Set kind to 'find_and_light' with the object as 'target' ONLY "
    "when the person clearly asks you to find, point at, light up or shine your light on one "
    "specific everyday object -- 'find my mug and shine your light on it'. The target must be "
    "the plain lowercase noun for that object ('mug', 'water bottle'): never a person, never a "
    "place, never an instruction, never a sentence. In every other case set kind to 'none' and "
    "target to an empty string; a question about something you looked at earlier is recall, "
    "not a goal, and is answered from your note. A goal does not replace your reply: still say "
    "your one or two sentences, and phrase them as something you are about to do ('let me have "
    "a look'), because you have not looked yet."
)

#: The goal-directed action contract, kept structurally apart from the
#: conversational one above. Two fields, one of which is an enum of two words:
#: there is no room in it for a gesture, a joint, an angle or a sequence, and a
#: second action would mean editing ``goal.GOAL_KINDS`` by hand.
GOAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": list(GOAL_KINDS),
            "description": (
                "'find_and_light' only when the person asks you to find a specific "
                "object in front of you now and shine your light on it. Otherwise 'none'."
            ),
        },
        "target": {
            "type": "string",
            "description": (
                "The plain lowercase noun for that object, at most "
                f"{MAX_TARGET_CHARS} characters. Empty string when kind is 'none'."
            ),
        },
    },
    "required": ["kind", "target"],
    "additionalProperties": False,
}

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
            "description": (
                "At most one physical gesture. Defaults to 'none', which is correct for "
                "most turns including greetings, ordinary questions and acknowledgement. "
                "'nod' only for an explicit request to move or a direct yes/agreement; "
                "'engage' only when the person is asking for your attention itself."
            ),
        },
        "goal": GOAL_SCHEMA,
    },
    "required": ["reply", "behavior", "goal"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class CharacterReply:
    """A validated turn: what to say, the single gesture, and any goal.

    ``behavior`` and ``goal`` are two separate contracts that happen to travel
    together. The gesture is ordinary conversational punctuation and is played
    immediately; the goal is a request to go and do something with the light,
    and nothing about it reaches the robot until fresh perception says it may.
    """

    reply: str
    behavior: str
    #: Defaults to no goal, so a response without the field -- an older one, or
    #: any of the goal-free tests -- is an ordinary turn rather than an error.
    goal: Goal = NO_GOAL


# --------------------------------------------------------------------------
# The observation contract
# --------------------------------------------------------------------------
#
# One frame in, five short fields out. The vision call cannot request a
# behaviour, cannot produce speech and cannot reach the robot: its entire
# output is a record that `scene_memory.parse_observation` validates before
# anything is stored.

OBSERVE_PROMPT = (
    "You are the eye of a small desk lamp robot. You are shown ONE frame from its "
    "camera, looking out across the desk in front of it.\n"
    "Pick the single most obvious object on the desk -- the one a person would mean if "
    "they said 'this thing here'. Ignore the person, their face and hands, the walls, "
    "the floor and the room.\n"
    "Report it in the given fields:\n"
    "  name    - a common noun for the object, one or two words ('mug', 'water bottle').\n"
    "  color   - its main visible colour, in plain words.\n"
    "  location- 'left', 'center' or 'right', meaning which side of the frame it is on; "
    "use 'unknown' only if you genuinely cannot tell.\n"
    "  detail  - ONE further short fact you are sure of (shape, material, a marking, "
    "what is next to it). Use an empty string rather than guessing.\n"
    "  confident - true only if you are sure there is one clear object and that name and "
    "colour are right. If the frame is dark, blurred, empty, or shows only a person, "
    "answer false and leave the other fields as short placeholders.\n"
    "Report only what is visible in this frame. Do not invent, and do not describe the "
    "person."
)

#: What the vision call is asked, alongside the image itself.
OBSERVE_REQUEST = "Look at this frame and describe the one clearest object on the desk."

OBSERVATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": f"Common noun for the object, at most {MAX_NAME_CHARS} characters.",
        },
        "color": {
            "type": "string",
            "description": f"Main visible colour, at most {MAX_COLOR_CHARS} characters.",
        },
        "location": {
            "type": "string",
            "enum": list(LOCATIONS),
            "description": "Which side of the frame the object is on.",
        },
        "detail": {
            "type": "string",
            "description": (
                "One further short fact you are sure of, or an empty string. "
                f"At most {MAX_DETAIL_CHARS} characters."
            ),
        },
        "confident": {
            "type": "boolean",
            "description": (
                "True only if one clear object was identified and the name and colour "
                "are right. False means nothing is remembered."
            ),
        },
    },
    "required": ["name", "color", "location", "detail", "confident"],
    "additionalProperties": False,
}

#: Five short fields; the cap is a backstop, not the control.
MAX_OBSERVATION_TOKENS = 200


# --------------------------------------------------------------------------
# The target lookup contract
# --------------------------------------------------------------------------
#
# A sibling of the observation above, and deliberately not a replacement for
# it. OBSERVE_PROMPT asks "what is the most obvious thing here?", which is the
# wrong question for "find my mug": the most obvious thing may not be the mug.
# This one asks about one named object and answers in three fields.
#
# Hour 4's observation path is untouched: different prompt, different schema,
# different parser, different memory (none -- a sighting is never stored).

LOCATE_PROMPT = (
    "You are the eye of a small desk lamp robot. You are shown ONE frame from its "
    "camera, looking out across the desk in front of it, and the name of ONE thing the "
    "robot has been asked to find. Nothing else in the frame matters.\n"
    "Answer only about that named target:\n"
    "  found     - true only if that specific object is actually visible in this frame. "
    "Do not accept a different object that is merely similar, and do not count a picture "
    "of one.\n"
    "  location  - 'left', 'center' or 'right', meaning which third of the frame the "
    "target is in. Use 'unknown' if you cannot tell, and whenever found is false.\n"
    "  confident - true only if you are sure of both answers. If the frame is dark, "
    "blurred or cluttered, or the target is only barely visible, answer false.\n"
    "The robot will turn its head and shine a light based on this answer, so a confident "
    "wrong answer is worse than an honest 'not found'. When in doubt, answer found=false, "
    "location='unknown', confident=false."
)

#: What the lookup call is asked, alongside the image and the target noun.
LOCATE_REQUEST = "Look at this frame and tell me whether you can see this thing, and where."

SIGHTING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {
            "type": "boolean",
            "description": "True only if the named target itself is visible in this frame.",
        },
        "location": {
            "type": "string",
            "enum": list(LOCATIONS),
            "description": (
                "Which third of the frame the target is in. 'unknown' when it is not "
                "visible or its side cannot be told."
            ),
        },
        "confident": {
            "type": "boolean",
            "description": (
                "True only if both other answers are sure. False means the robot will "
                "not move and will not light anything."
            ),
        },
    },
    "required": ["found", "location", "confident"],
    "additionalProperties": False,
}

#: Three short fields; the cap is a backstop, not the control.
MAX_SIGHTING_TOKENS = 100

#: Same budget as an observation -- it is the same kind of call with a picture
#: in it, and it happens twice inside one goal.
LOCATE_TIMEOUT_SECONDS = OBSERVE_TIMEOUT_SECONDS


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

    unexpected = set(payload) - {"reply", "behavior", "goal"}
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
    # The goal is validated by `goal.parse_goal`, which owns the two-word kind
    # enum and the bounded noun. A malformed goal fails the whole turn rather
    # than being dropped: "I could not read what you asked me to do" must not
    # quietly become "you asked me to do nothing".
    try:
        requested = parse_goal(payload.get("goal"))
    except GoalError as exc:
        raise CharacterError(f"character response carried an unusable goal: {exc}") from exc
    return CharacterReply(reply=reply.strip(), behavior=behavior, goal=requested)


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

    def respond(self, transcript: str, scene_note: str = "") -> CharacterReply:
        """Answer *transcript* in character and choose zero or one gesture.

        *scene_note* is the text rendered by :meth:`scene_memory.SceneMemory.
        prompt_note` -- the facts the character noted when it last looked, or
        "" if it has never looked. It is passed as text and nothing else: this
        method has no way to reach a camera, so recall cannot become a second
        observation however the question is phrased.
        """
        try:
            response = self._client.responses.create(
                model=CHARACTER_MODEL,
                instructions=SYSTEM_PROMPT,
                input=_compose_input(transcript, scene_note),
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

    # -- the one call that carries a picture -------------------------------

    def observe(self, jpeg_bytes: bytes) -> SceneObservation:
        """Look at exactly one frame and return a validated scene record.

        Called from a bounded, person-triggered observation event, never from a
        loop and never from the recall path. The frame is uploaded inline as a
        data URL; nothing is stored server-side by us, and the result the
        character keeps is the five validated fields, not the image.

        Raises :class:`CharacterError` if the call itself failed, and
        :class:`scene_memory.ObservationError` if the response was unusable or
        the model was not confident. Either way the caller stores nothing.
        """
        if not jpeg_bytes:
            raise CharacterError("no camera frame to look at")

        encoded = base64.b64encode(jpeg_bytes).decode("ascii")
        try:
            response = self._client.responses.create(
                model=VISION_MODEL,
                instructions=OBSERVE_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": OBSERVE_REQUEST},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{encoded}",
                            },
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
                max_output_tokens=MAX_OBSERVATION_TOKENS,
                timeout=OBSERVE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise CharacterError(f"looking at the scene failed: {_safe(exc)}") from exc

        raw = getattr(response, "output_text", None)
        if not isinstance(raw, str) or not raw.strip():
            raise CharacterError("the observation response was empty")
        # Validated locally regardless of the schema request above.
        return parse_observation(raw)

    def locate(self, jpeg_bytes: bytes, target: str) -> TargetSighting:
        """Look at one frame for one named *target* and report three fields.

        The sibling of :meth:`observe`, for the one case that record cannot
        serve: "is the thing they asked about in front of me *now*?". Called
        twice in a goal -- once to decide which way to turn, once after turning
        to decide whether the light may come on -- and each call gets its own
        freshly captured frame.

        *target* is re-validated here, immediately before it is written into a
        prompt, even though the spoken turn already validated it. It is a noun
        and nothing else: it is not an action, not a parameter, and it never
        reaches the robot.

        Raises :class:`CharacterError` if the call failed or the target was not
        bounded semantic text, and :class:`goal.GoalError` if the response was
        unusable. Either way the caller neither moves nor lights.
        """
        if not jpeg_bytes:
            raise CharacterError("no camera frame to look at")
        try:
            subject = normalize_target(target)
        except GoalError as exc:
            raise CharacterError(f"refusing to look for an unbounded target: {exc}") from exc

        encoded = base64.b64encode(jpeg_bytes).decode("ascii")
        try:
            response = self._client.responses.create(
                model=VISION_MODEL,
                instructions=LOCATE_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"{LOCATE_REQUEST}\nThe target is: {subject}",
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{encoded}",
                            },
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "target_sighting",
                        "strict": True,
                        "schema": SIGHTING_SCHEMA,
                    }
                },
                max_output_tokens=MAX_SIGHTING_TOKENS,
                timeout=LOCATE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise CharacterError(f"looking for the {subject} failed: {_safe(exc)}") from exc

        raw = getattr(response, "output_text", None)
        if not isinstance(raw, str) or not raw.strip():
            raise CharacterError("the target lookup response was empty")
        # Validated locally regardless of the schema request above.
        return parse_sighting(raw)

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


#: Label the person's words carry when a scene note precedes them, so the two
#: blocks cannot be read as one. The note itself is flattened to single-line
#: fields by ``scene_memory``, so it cannot forge this header.
SPOKEN_HEADER = "THE PERSON JUST SAID:"


def _compose_input(transcript: str, scene_note: str) -> str:
    """Build the turn's input: the note (when there is one) then the words.

    With no note this is exactly the transcript, unchanged -- an ordinary turn
    looks the same as it did before scene memory existed.
    """
    note = (scene_note or "").strip()
    if not note:
        return transcript
    return f"{note}\n\n{SPOKEN_HEADER}\n{transcript}"


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
