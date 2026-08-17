"""Audio boundary: laptop microphone in, laptop speaker out.

This module is the *only* place that touches audio hardware. Everything above
it (``character``, ``voice_turn``) deals in plain ``bytes`` holding a complete
RIFF/WAV file, which is exactly what the OpenAI audio endpoints accept and
return. That keeps the whole voice path testable with fakes: a unit test
supplies bytes and asserts on bytes, and never opens a device.

Design notes
------------
* **WAV, not compressed audio.** Mono PCM16 is what the STT endpoint wants and
  ``tts-1`` will hand back on request, and Python's stdlib ``wave`` module
  reads and writes it. No ffmpeg, no codec wheels.
* **Bounded push-to-talk, not VAD.** ``MicrophoneRecorder.record()`` captures a
  fixed :data:`RECORD_SECONDS` window and returns. Voice-activity detection is
  a whole tuning problem of its own and buys nothing for a demo where the
  person deliberately presses a key first.
* **sounddevice**, because it is a thin PortAudio binding with one wheel per
  platform (ALSA/PulseAudio on the Ubuntu target, WASAPI on Windows,
  CoreAudio on macOS) and needs no server, no daemon and no plugin config.
  It is imported lazily so that importing this module -- as the tests do --
  never probes the audio subsystem.
* **Playback is split into prepare + play.** Everything that can fail while
  turning WAV bytes into something the sound card will accept -- container
  parsing, sample width, NumPy conversion, loading PortAudio, asking the
  output device whether it supports this rate -- happens in
  :meth:`SpeakerPlayer.prepare`, which the caller runs *before* it moves the
  robot. ``play()`` is then only the handoff to PortAudio. See ``voice_turn``
  for why that ordering matters.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Any, Protocol

# --------------------------------------------------------------------------
# Capture format
# --------------------------------------------------------------------------

#: Mono. Stereo would double the upload for no transcription benefit.
CHANNELS = 1

#: 16 kHz is the speech-recognition standard rate and the lowest rate that
#: keeps consonants intelligible. 4 s of it is ~128 kB -- a fast upload. This
#: is a *preference*: a device that refuses it is recorded at its own native
#: rate instead (see MicrophoneRecorder), because the STT endpoint accepts any
#: sane rate and resampling locally would mean pulling in a DSP dependency.
PREFERRED_SAMPLE_RATE = 16_000

#: Backwards-compatible alias; several tests and callers read this name.
SAMPLE_RATE = PREFERRED_SAMPLE_RATE

#: 16-bit signed PCM: the only width ``wave`` + PortAudio agree on trivially.
SAMPLE_WIDTH = 2

#: Length of one push-to-talk utterance. Long enough for a sentence or a
#: question, short enough that a turn stays snappy and the bound is obvious to
#: the person speaking.
RECORD_SECONDS = 4.0

#: Rates to try for capture, in order, if the preferred one is refused. These
#: are the native rates integrated laptop codecs actually expose on macOS and
#: on ALSA-without-PulseAudio Linux boxes.
FALLBACK_SAMPLE_RATES = (48_000, 44_100)


class AudioError(RuntimeError):
    """Raised when audio hardware is unavailable or a buffer is unusable.

    Every failure in this module surfaces as this one type, so the turn
    orchestrator has a single thing to catch.
    """


# --------------------------------------------------------------------------
# Testable boundaries
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedPlayback:
    """Decoded, validated audio that the output device has already accepted.

    Holding one of these is the evidence ``voice_turn`` needs that speaking is
    very likely to succeed. It is not a guarantee -- a device can still be
    unplugged between prepare and play -- but every failure mode we can check
    cheaply has been checked by the time this exists.
    """

    #: Interleaved int16 samples, shaped (frames, channels). Typed loosely so
    #: this module's public surface does not force a NumPy import on callers.
    samples: Any
    sample_rate: int
    channels: int
    #: Wall-clock length, used by the caller to know when the lamp stops talking.
    seconds: float


class Recorder(Protocol):
    """Anything that can produce one bounded utterance as WAV bytes."""

    def record(self) -> bytes: ...


class Player(Protocol):
    """Anything that can validate WAV bytes and then play them."""

    def prepare(self, wav_bytes: bytes) -> PreparedPlayback: ...

    def play(self, prepared: PreparedPlayback) -> None: ...


# --------------------------------------------------------------------------
# WAV helpers (stdlib only)
# --------------------------------------------------------------------------


def encode_wav(
    pcm: bytes, sample_rate: int = PREFERRED_SAMPLE_RATE, channels: int = CHANNELS
) -> bytes:
    """Wrap raw PCM16 frames in a RIFF/WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def decode_wav(wav_bytes: bytes) -> tuple[bytes, int, int, int]:
    """Return ``(pcm, sample_rate, channels, sample_width)`` from WAV bytes.

    Deliberately catches broadly: a truncated or non-RIFF buffer reaches
    ``wave`` as anything from ``wave.Error`` to ``EOFError`` to ``struct.error``
    depending on *where* it is malformed, and all of them mean the same thing
    to the caller.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            pcm = handle.readframes(handle.getnframes())
    except Exception as exc:
        raise AudioError(f"not a readable WAV stream: {exc}") from exc

    if channels < 1 or rate < 1:
        raise AudioError(f"WAV header is nonsensical: {channels} channels at {rate} Hz")
    if width != SAMPLE_WIDTH:
        raise AudioError(f"expected 16-bit PCM audio, got {width * 8}-bit")
    if len(pcm) < channels * width:
        raise AudioError("WAV stream contains no audio frames")
    if len(pcm) % (channels * width):
        raise AudioError("WAV stream ends mid-frame")
    return pcm, rate, channels, width


def _sounddevice():
    """Import sounddevice on first use, with an actionable error if it fails.

    On a fresh Ubuntu box the usual failure is a missing PortAudio shared
    library rather than a missing wheel, so say so explicitly.
    """
    try:
        import sounddevice  # noqa: PLC0415  (deliberately lazy)
    except OSError as exc:  # PortAudio itself missing / unloadable
        raise AudioError(
            f"PortAudio could not be loaded ({exc}). On Ubuntu 24.04 install it with "
            "'sudo apt install libportaudio2'."
        ) from exc
    except Exception as exc:
        raise AudioError(
            f"the 'sounddevice' package is not usable ({exc}); "
            "install it from requirements.txt."
        ) from exc
    return sounddevice


def _numpy():
    """Import NumPy on first use. Already present transitively via OpenCV."""
    try:
        import numpy  # noqa: PLC0415  (deliberately lazy)
    except Exception as exc:
        raise AudioError(f"NumPy is required for audio conversion but unusable: {exc}") from exc
    return numpy


# --------------------------------------------------------------------------
# Real devices
# --------------------------------------------------------------------------


class MicrophoneRecorder:
    """Captures a fixed-length mono utterance from the default input device.

    Tries :data:`PREFERRED_SAMPLE_RATE` first and falls back to rates that
    integrated codecs commonly expose natively. Whatever rate wins is written
    into the WAV header, so the transcription endpoint is always told the
    truth; nothing downstream assumes 16 kHz.
    """

    def __init__(
        self, seconds: float = RECORD_SECONDS, sample_rate: int = PREFERRED_SAMPLE_RATE
    ):
        self.seconds = float(seconds)
        self.preferred_rate = int(sample_rate)
        #: Sticky: once a rate works we stop re-probing on every turn.
        self._rate: int | None = None

    def _candidate_rates(self, sd) -> list[int]:
        if self._rate is not None:
            return [self._rate]
        rates = [self.preferred_rate, *FALLBACK_SAMPLE_RATES]
        try:
            native = int(sd.query_devices(kind="input")["default_samplerate"])
        except Exception:
            # Not being able to ask is not fatal; the fixed candidates remain.
            native = 0
        if native:
            rates.append(native)
        seen: list[int] = []
        for rate in rates:
            if rate > 0 and rate not in seen:
                seen.append(rate)
        return seen

    def record(self) -> bytes:
        sd = _sounddevice()
        failures: list[str] = []
        for rate in self._candidate_rates(sd):
            try:
                recording = sd.rec(
                    int(self.seconds * rate),
                    samplerate=rate,
                    channels=CHANNELS,
                    dtype="int16",
                    blocking=True,
                )
            except Exception as exc:  # PortAudioError and friends
                failures.append(f"{rate} Hz: {exc}")
                continue
            self._rate = rate
            try:
                return encode_wav(recording.tobytes(), rate, CHANNELS)
            except Exception as exc:
                raise AudioError(f"could not encode the recording as WAV: {exc}") from exc

        raise AudioError(
            "could not record from the default microphone ("
            + "; ".join(failures)
            + "). Check that an input device exists and that this terminal has "
            "microphone permission."
        )


class SpeakerPlayer:
    """Plays a WAV buffer through the default output device.

    Split in two on purpose: see the module docstring and ``voice_turn``.
    """

    def prepare(self, wav_bytes: bytes) -> PreparedPlayback:
        """Decode and validate audio, and confirm the device will accept it.

        Everything that can go wrong short of the device physically vanishing
        goes wrong here, before the caller commits to moving the robot.
        """
        if not wav_bytes:
            raise AudioError("no audio to play")

        pcm, sample_rate, channels, _width = decode_wav(wav_bytes)
        np = _numpy()
        try:
            samples = np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels)
        except Exception as exc:
            raise AudioError(f"could not interpret the decoded audio: {exc}") from exc

        sd = _sounddevice()
        try:
            # Asks PortAudio, not the OS mixer: this is the check that catches
            # a device exposing only 44.1/48 kHz being handed 24 kHz speech.
            sd.check_output_settings(
                samplerate=sample_rate, channels=channels, dtype="int16"
            )
        except Exception as exc:
            raise AudioError(
                f"the default output device will not accept {sample_rate} Hz "
                f"{channels}-channel audio ({exc})."
            ) from exc

        return PreparedPlayback(
            samples=samples,
            sample_rate=sample_rate,
            channels=channels,
            seconds=len(samples) / float(sample_rate),
        )

    def play(self, prepared: PreparedPlayback) -> None:
        """Hand the prepared samples to PortAudio and return immediately.

        Non-blocking on purpose: the caller is the engagement loop, which must
        keep reading camera frames while the character is talking. PortAudio
        drains the buffer on its own thread.
        """
        sd = _sounddevice()
        try:
            sd.play(prepared.samples, samplerate=prepared.sample_rate, blocking=False)
        except Exception as exc:
            raise AudioError(f"could not play audio on the default speaker: {exc}") from exc
