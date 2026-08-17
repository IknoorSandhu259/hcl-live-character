"""Small, inspectable scene memory for the Hour 4 object-recall flow.

Vision produces one semantic :class:`SceneObservation`; this module owns the
retained state. It deliberately stores no pixels, embeddings, tracks, or model
objects. Later language reasoning receives only the structured facts below, so
recall is visibly separated from live perception.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


UNKNOWN = "unknown"
MAX_FIELD_CHARS = 120


@dataclass(frozen=True)
class SceneObservation:
    """Useful facts retained about one visible desk object."""

    object_type: str
    color: str
    relative_location: str
    detail: str

    def __post_init__(self) -> None:
        for name in ("object_type", "color", "relative_location", "detail"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            if len(value) > MAX_FIELD_CHARS:
                raise ValueError(f"{name} exceeds {MAX_FIELD_CHARS} characters")
            object.__setattr__(self, name, value.strip())

    def summary(self) -> str:
        return (
            f"{self.color} {self.object_type}; location={self.relative_location}; "
            f"detail={self.detail}"
        )


class SceneMemory:
    """One-object, process-local memory.

    Re-observing replaces the prior object. That is intentional: the challenge
    asks for useful recall of at least one object, not persistent multi-object
    world modelling.
    """

    def __init__(self) -> None:
        self._observation: Optional[SceneObservation] = None

    @property
    def observation(self) -> Optional[SceneObservation]:
        return self._observation

    def remember(self, observation: SceneObservation) -> None:
        if not isinstance(observation, SceneObservation):
            raise TypeError("SceneMemory only accepts SceneObservation values")
        self._observation = observation

    def clear(self) -> None:
        self._observation = None

    def context(self) -> Optional[str]:
        """Return bounded facts for later language reasoning, or None."""
        observation = self._observation
        if observation is None:
            return None
        return (
            "Retained scene memory from an earlier camera observation:\n"
            f"- object: {observation.object_type}\n"
            f"- visible color: {observation.color}\n"
            f"- rough relative location: {observation.relative_location}\n"
            f"- other visible detail: {observation.detail}\n"
            "This is memory, not a live camera view. Use only these retained facts "
            "when answering questions about the observed object."
        )


class MemoryAwareBrain:
    """Adapter that reuses the existing voice path while injecting local memory.

    It has the same transcribe/respond/synthesize surface as CharacterBrain, so
    VoiceTurn does not need a parallel recall implementation.
    """

    def __init__(self, brain, memory: SceneMemory):
        self._brain = brain
        self._memory = memory

    def transcribe(self, wav_bytes: bytes) -> str:
        return self._brain.transcribe(wav_bytes)

    def respond(self, transcript: str):
        return self._brain.respond(transcript, scene_context=self._memory.context())

    def synthesize(self, reply: str) -> bytes:
        return self._brain.synthesize(reply)
