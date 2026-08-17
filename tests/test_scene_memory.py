"""Hermetic tests for bounded observation -> retained memory -> later recall context."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from character import CharacterBrain, CharacterError, parse_scene_observation  # noqa: E402
from scene_memory import MemoryAwareBrain, SceneMemory, SceneObservation  # noqa: E402


class _Endpoint:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        return type("Response", (), {"output_text": output})()


class _Audio:
    pass


class _Client:
    def __init__(self, outputs):
        self.responses = _Endpoint(outputs)
        self.audio = _Audio()


def test_scene_memory_is_one_object_structured_and_inspectable():
    memory = SceneMemory()
    assert memory.context() is None

    observation = SceneObservation(
        object_type="mug",
        color="blue",
        relative_location="left side of the desk",
        detail="white handle",
    )
    memory.remember(observation)

    assert memory.observation is observation
    context = memory.context()
    assert "object: mug" in context
    assert "visible color: blue" in context
    assert "left side of the desk" in context
    assert "white handle" in context
    assert "live camera" in context


def test_reobservation_replaces_old_object_instead_of_building_world_model():
    memory = SceneMemory()
    memory.remember(SceneObservation("mug", "blue", "left", "handle visible"))
    memory.remember(SceneObservation("notebook", "red", "center", "closed"))

    assert memory.observation.object_type == "notebook"
    assert "mug" not in memory.context()


def test_visual_observation_is_schema_validated_locally():
    raw = json.dumps(
        {
            "object_type": "mug",
            "color": "blue",
            "relative_location": "left",
            "detail": "white handle",
        }
    )
    observation = parse_scene_observation(raw)
    assert observation == SceneObservation("mug", "blue", "left", "white handle")

    with pytest.raises(CharacterError, match="fields invalid"):
        parse_scene_observation(json.dumps({"object_type": "mug", "color": "blue"}))


def test_observe_sends_one_image_and_returns_semantic_facts_only():
    payload = json.dumps(
        {
            "object_type": "mug",
            "color": "blue",
            "relative_location": "left",
            "detail": "white handle",
        }
    )
    client = _Client([payload])
    brain = CharacterBrain(client=client)

    observation = brain.observe(b"jpeg bytes")

    assert observation.object_type == "mug"
    call = client.responses.calls[0]
    assert call["model"] == "gpt-5.6-luna"
    content = call["input"][0]["content"]
    images = [part for part in content if part["type"] == "input_image"]
    assert len(images) == 1
    assert images[0]["image_url"].startswith("data:image/jpeg;base64,")
    assert call["text"]["format"]["schema"]["additionalProperties"] is False


def test_later_recall_receives_memory_text_not_an_image():
    response = json.dumps({"reply": "It was blue and on the left.", "behavior": "none"})
    client = _Client([response])
    base = CharacterBrain(client=client)
    memory = SceneMemory()
    memory.remember(SceneObservation("mug", "blue", "left", "white handle"))
    brain = MemoryAwareBrain(base, memory)

    result = brain.respond("What color was the mug and where was it?")

    assert result.reply == "It was blue and on the left."
    call = client.responses.calls[0]
    assert isinstance(call["input"], str)
    assert "What color was the mug" in call["input"]
    assert "visible color: blue" in call["input"]
    assert "rough relative location: left" in call["input"]
    assert "input_image" not in call["input"]
