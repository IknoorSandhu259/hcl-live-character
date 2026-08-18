"""The two ways the demo can be pointed at OpenAI, and the line between them.

Normal development uses a real key and talks to api.openai.com directly. An HCL
reviewer instead gets a temporary token and the URL of the optional relay in
``reviewer_proxy/``, selected purely with environment variables::

    OPENAI_API_KEY=<reviewer token>   OPENAI_BASE_URL=https://<host>/v1

``src/character.py`` is identical in both modes -- there is no proxy branch in
it -- because the official SDK reads ``OPENAI_BASE_URL`` itself. These tests pin
that behaviour so a future SDK bump could not silently send a reviewer token to
the real API. Constructing a client makes no request, so nothing here spends
quota.

    python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import character  # noqa: E402

pytest.importorskip("openai")

REAL_KEY = "sk-proj-not-a-real-key-0123456789"
REVIEWER_TOKEN = "hclrev_3f9a2c7e1b4d"
PROXY_BASE_URL = "https://hcl-reviewer-proxy.example.vercel.app/v1"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Neither variable leaks in from the developer's own shell."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def test_direct_mode_still_talks_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", REAL_KEY)

    client = character.build_client()

    assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1"
    assert client.api_key == REAL_KEY


def test_direct_mode_still_requires_a_key():
    with pytest.raises(character.MissingCredentialsError):
        character.require_api_key()


def test_reviewer_mode_is_selected_by_environment_alone(monkeypatch):
    """Two variables, no code change, and no real key anywhere in the process."""
    monkeypatch.setenv("OPENAI_API_KEY", REVIEWER_TOKEN)
    monkeypatch.setenv("OPENAI_BASE_URL", PROXY_BASE_URL)

    client = character.build_client()

    assert str(client.base_url).rstrip("/") == PROXY_BASE_URL
    assert client.api_key == REVIEWER_TOKEN
    assert not client.api_key.startswith("sk-")


@pytest.mark.parametrize(
    "path", ["/responses", "/audio/transcriptions", "/audio/speech"]
)
def test_reviewer_mode_hits_exactly_the_three_proxied_paths(monkeypatch, path):
    """A fourth OpenAI call would 404 at the proxy, not quietly widen its reach."""
    monkeypatch.setenv("OPENAI_API_KEY", REVIEWER_TOKEN)
    monkeypatch.setenv("OPENAI_BASE_URL", PROXY_BASE_URL)

    client = character.build_client()

    assert str(client._prepare_url(path)) == f"{PROXY_BASE_URL}{path}"


def test_the_models_character_asks_for_are_the_proxy_allowlist():
    """A rename here must be mirrored in reviewer_proxy/lib/proxy.js."""
    assert character.STT_MODEL == "gpt-4o-mini-transcribe"
    assert character.CHARACTER_MODEL == "gpt-5.6-luna"
    assert character.TTS_MODEL == "tts-1"
    assert character.TTS_VOICE == "alloy"
    # The proxy's per-request ceilings are derived from these.
    assert character.MAX_OUTPUT_TOKENS <= 1_000
    assert character.MAX_REPLY_CHARS <= 500


def test_a_reviewer_token_is_redacted_like_a_real_key(monkeypatch):
    """`redact` reads OPENAI_API_KEY, whatever kind of credential it holds."""
    monkeypatch.setenv("OPENAI_API_KEY", REVIEWER_TOKEN)

    error = character.CharacterError(f"proxy refused {REVIEWER_TOKEN}")

    assert REVIEWER_TOKEN not in str(error)
    assert "<redacted>" in str(error)
