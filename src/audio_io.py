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
  turning WAV bytes into something playable -- container parsing, sample
  width, NumPy conversion, loading PortAudio, asking the output device whether
  it supports this rate, locating a system player -- happens in
  ``prepare()``, which the caller runs *before* it moves the robot.
  ``play()`` is then only the handoff. See ``voice_turn`` for why that
  ordering matters.

Two playback backends
---------------------
:class:`SpeakerPlayer` picks one at construction:

* :class:`SystemCommandPlayer` hands the WAV to the platform's own player
  (``afplay`` on macOS, ``paplay``/``aplay`` on Linux) as a **separate
  process**. Preferred wherever such a binary exists.
* :class:`SounddevicePlayer` streams the samples in-process through PortAudio.
  Used when no system player is available (notably Windows).

The split exists because in-process PortAudio playback and a PyBullet **GUI**
in the same interpreter fight over the GIL: sounddevice's stream callback is
Python code invoked from the CoreAudio/ALSA real-time thread, and it has to
take the GIL to run. PyBullet's GUI renderer holds the GIL in long bursts, the
callback misses its deadline, and the output underruns -- audible as heavy
static. The effect is absent headless (``--headless``) and absent when the
same WAV is played by any separate process, which is exactly what handing the
file to ``afplay`` restores. A separate process has its own interpreter lock,
so no amount of rendering in ours can starve its audio thread.
"""

from __future__ import annotations

import atexit
import io
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

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
    """Validated audio that the chosen backend has already accepted.

    Holding one of these is the evidence ``voice_turn`` needs that speaking is
    very likely to succeed. It is not a guarantee -- a device can still be
    unplugged between prepare and play -- but every failure mode we can check
    cheaply has been checked by the time this exists.

    Exactly one of ``samples`` / ``path`` is populated, depending on backend.
    """

    sample_rate: int
    channels: int
    #: Wall-clock length, used by the caller to know when the lamp stops talking.
    seconds: float
    #: Interleaved int16 samples, shaped (frames, channels). Typed loosely so
    #: this module's public surface does not force a NumPy import on callers.
    samples: Any = None
    #: Path to a WAV file on disk, for the system-player backend.
    path: Optional[str] = None


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


class SounddevicePlayer:
    """Streams samples through PortAudio, in this process.

    The portable fallback. Correct everywhere, but shares the interpreter --
    and therefore the GIL -- with whatever else is running, which is why it is
    not the first choice when a PyBullet GUI is on screen. See the module
    docstring.
    """

    def prepare(self, wav_bytes: bytes) -> PreparedPlayback:
        """Decode and validate audio, and confirm the device will accept it."""
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
        keep reading camera frames while the character is talking.

        ``latency="high"`` asks PortAudio for the largest buffer it considers
        reasonable. It does not remove the GIL contention described in the
        module docstring, but it widens the deadline the stream callback has to
        meet, which is the most this backend can do about it from inside the
        same interpreter.
        """
        sd = _sounddevice()
        try:
            sd.play(
                prepared.samples,
                samplerate=prepared.sample_rate,
                blocking=False,
                latency="high",
            )
        except Exception as exc:
            raise AudioError(f"could not play audio on the default speaker: {exc}") from exc


#: System players to look for, per platform, best first. Each entry is the
#: argv prefix; the WAV path is appended. All of these take a plain RIFF/WAV
#: file and exit when it finishes.
SYSTEM_PLAYERS: dict[str, Sequence[Sequence[str]]] = {
    "darwin": (("afplay",),),
    # paplay covers PulseAudio and PipeWire's pulse shim, which is what a
    # stock Ubuntu 24.04 desktop runs; aplay is the bare-ALSA fallback.
    "linux": (("paplay",), ("aplay", "-q")),
}


def find_system_player(platform: Optional[str] = None) -> Optional[list[str]]:
    """Return the argv prefix of an available system player, or None."""
    for candidate in SYSTEM_PLAYERS.get(platform or sys.platform, ()):
        binary = shutil.which(candidate[0])
        if binary:
            return [binary, *candidate[1:]]
    return None


class SystemCommandPlayer:
    """Plays a WAV by handing the file to the platform's own player process.

    The point is the process boundary, not the binary: a separate process has
    its own GIL, so PyBullet's GUI renderer cannot starve its audio thread.
    Playback stays non-blocking -- we spawn and return, exactly as the
    in-process backend does.
    """

    def __init__(self, command: Optional[Sequence[str]] = None):
        resolved = list(command) if command else find_system_player()
        if not resolved:
            raise AudioError(f"no system audio player found for platform {sys.platform!r}")
        self.command = resolved
        self._directory = tempfile.mkdtemp(prefix="lamp-tts-")
        self._counter = 0
        #: The file from the previous turn, deleted once the next one starts.
        self._previous: Optional[str] = None
        atexit.register(self._cleanup)

    def prepare(self, wav_bytes: bytes) -> PreparedPlayback:
        """Validate the audio and stage it on disk for the player process."""
        if not wav_bytes:
            raise AudioError("no audio to play")

        # Same validation as the in-process backend: a malformed or non-PCM16
        # buffer must fail here, before the robot moves, not inside afplay.
        pcm, sample_rate, channels, width = decode_wav(wav_bytes)

        self._discard_previous()
        self._counter += 1
        path = os.path.join(self._directory, f"reply-{self._counter}.wav")
        try:
            with open(path, "wb") as handle:
                handle.write(wav_bytes)
        except OSError as exc:
            raise AudioError(f"could not stage audio for playback: {exc}") from exc

        self._previous = path
        frames = len(pcm) // (channels * width)
        return PreparedPlayback(
            sample_rate=sample_rate,
            channels=channels,
            seconds=frames / float(sample_rate),
            path=path,
        )

    def play(self, prepared: PreparedPlayback) -> None:
        """Spawn the player and return immediately."""
        if not prepared.path:
            raise AudioError("this backend needs audio staged on disk")
        try:
            subprocess.Popen(  # noqa: S603 - fixed argv, resolved binary, no shell
                [*self.command, prepared.path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            raise AudioError(
                f"could not start {self.command[0]!r} to play the reply: {exc}"
            ) from exc

    # -- housekeeping ------------------------------------------------------

    def _discard_previous(self) -> None:
        """Delete the last turn's file. Safe: a new turn cannot start while
        the previous reply is still playing (VoiceSession.mark_speaking)."""
        if self._previous:
            try:
                os.unlink(self._previous)
            except OSError:
                pass  # already gone, or still held; the temp dir sweeps it up
            self._previous = None

    def _cleanup(self) -> None:
        shutil.rmtree(self._directory, ignore_errors=True)


#: Set LAMP_AUDIO_BACKEND=sounddevice to force the in-process path, or
#: =system to insist on the subprocess one. Unset picks the best available.
BACKEND_ENV_VAR = "LAMP_AUDIO_BACKEND"


def SpeakerPlayer(backend: Optional[str] = None):  # noqa: N802 - kept as a name
    """Build the best playback backend for this machine.

    Prefers :class:`SystemCommandPlayer` wherever a system player exists
    (macOS, and any Ubuntu box with pulse/alsa utils), because it is immune to
    the GUI/GIL contention described in the module docstring. Falls back to
    :class:`SounddevicePlayer` otherwise, so Windows and stripped-down Linux
    installs keep working with no extra system packages.

    Callable as a class was, so existing wiring and tests are unaffected.
    """
    choice = (backend or os.environ.get(BACKEND_ENV_VAR, "")).strip().lower()
    if choice == "sounddevice":
        return SounddevicePlayer()
    if choice == "system":
        return SystemCommandPlayer()
    if choice:
        raise AudioError(
            f"{BACKEND_ENV_VAR}={choice!r} is not a known backend; "
            "use 'system' or 'sounddevice'."
        )
    if find_system_player():
        return SystemCommandPlayer()
    return SounddevicePlayer()
