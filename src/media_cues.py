"""The character's two non-verbal sounds, generated locally and deterministically.

Hour 6 adds the last two modalities the challenge asks for -- a sound effect and
music -- and this module is the whole of them:

    engagement_ack_wav()   a short acknowledgement SFX, played with the
                           IDLE -> ENGAGED motion
    goal_success_wav()     a brief musical flourish, played once a goal has
                           verified its target and turned the light on

That is the entire public vocabulary, and it is deliberately two literal
functions rather than a registry. There is no name lookup, no cue table, no
filename and no path: nothing the language model says can select a sound,
because there is no parameter through which a selection could be expressed.
Both cues are written by this file at runtime, so the repository carries no
audio asset and no licence question with it.

Why synthesised rather than shipped
-----------------------------------
* **No new dependency and no new failure mode.** ``math`` and ``struct`` build
  the samples; :func:`audio_io.encode_wav` -- already the capture path's WAV
  writer -- wraps them in the same mono PCM16 container the speaker backends
  accept. The cues therefore travel down the *existing* ``prepare``/``play``
  interface as ordinary WAV bytes, exactly like a spoken reply does.
* **Deterministic.** No randomness anywhere, so a cue is byte-identical on
  every run and a test can assert on its shape rather than on a tolerance.
* **Bounded by construction.** :data:`MAX_CUE_SECONDS` is checked while the
  buffer is built, so neither cue can ever become long enough to hold the
  shared speaker -- and therefore the microphone -- shut for a surprising time.

Telling the two apart
---------------------
They have to be *audibly* different kinds of sound, not merely different
lengths, or the demo shows one modality twice. So:

* the acknowledgement is a single unpitched-sounding upward chirp lasting
  :data:`ENGAGEMENT_ACK_SECONDS` -- a fraction of a second, percussive, with a
  fast decay. There is no melody in it because there is only one gliding tone.
* the success flourish is :data:`GOAL_SUCCESS_SECONDS` of a six-note major
  arpeggio, each note separately struck and given a harmonic. It is
  unmistakably a tune.
"""

from __future__ import annotations

import math
import struct
from functools import lru_cache
from typing import Iterable, Iterator, Sequence, Tuple

from audio_io import CHANNELS, encode_wav

#: Cues are generated at the capture rate the rest of the audio path already
#: uses, so nothing in the pipeline has to resample or re-negotiate a device.
CUE_SAMPLE_RATE = 16_000

#: Hard ceiling on either cue. A decorative sound holds the shared speaker --
#: and so the microphone -- for its whole length, so "short" is a correctness
#: property, not a matter of taste.
MAX_CUE_SECONDS = 4.0

#: The acknowledgement is meant to read as a blip next to the engage motion.
ENGAGEMENT_ACK_SECONDS = 0.28

#: The chirp sweeps upward across this range: high enough to sit clear of
#: speech, and moving fast enough that no pitch is heard, only a flick.
_ACK_START_HZ = 1400.0
_ACK_END_HZ = 2600.0

#: (frequency in Hz, seconds). A C-major arpeggio up to the octave and back to
#: it -- six struck notes, which is what makes this a tune rather than a beep.
GOAL_SUCCESS_NOTES: Sequence[Tuple[float, float]] = (
    (523.25, 0.36),   # C5
    (659.25, 0.36),   # E5
    (783.99, 0.36),   # G5
    (1046.50, 0.42),  # C6
    (783.99, 0.30),   # G5
    (1046.50, 0.60),  # C6
)

#: Total length of the flourish: comfortably longer than the acknowledgement,
#: which is the property that keeps the two cues from being confused.
GOAL_SUCCESS_SECONDS = sum(seconds for _frequency, seconds in GOAL_SUCCESS_NOTES)

#: Peak amplitude as a fraction of full scale. Well below 1.0: these play over
#: a laptop speaker next to a person, and headroom costs nothing.
_PEAK = 0.35

#: Full-scale value for PCM16.
_FULL_SCALE = 32767


def _pcm16(samples: Iterable[float]) -> bytes:
    """Turn a stream of floats in [-1, 1] into little-endian PCM16 frames.

    Clamps rather than wraps: a sample that overshot would otherwise come back
    as a sample of the opposite sign, which is heard as a click.
    """
    values = [
        int(round(max(-1.0, min(1.0, sample)) * _PEAK * _FULL_SCALE))
        for sample in samples
    ]
    # Explicit little-endian, so the bytes do not depend on the host's order.
    return struct.pack(f"<{len(values)}h", *values)


def _chirp(seconds: float, start_hz: float, end_hz: float, rate: int) -> Iterator[float]:
    """One upward sweep with a fast exponential decay: a percussive blip.

    The phase is integrated rather than computed from ``sin(2*pi*f*t)`` with a
    moving ``f``, because the latter is not phase-continuous while the
    frequency changes and produces an audible tear part-way through the sweep.
    """
    total = max(1, int(seconds * rate))
    attack = max(1, int(0.002 * rate))
    phase = 0.0
    for index in range(total):
        fraction = index / total
        frequency = start_hz + (end_hz - start_hz) * fraction
        phase += 2.0 * math.pi * frequency / rate
        envelope = min(1.0, index / attack) * math.exp(-9.0 * fraction)
        yield envelope * math.sin(phase)


def _note(frequency: float, seconds: float, rate: int) -> Iterator[float]:
    """One struck note: a sine plus a quiet octave, under a decaying envelope.

    The envelope is what makes six notes sound like six notes -- without the
    decay between them the arpeggio runs together into one chord-shaped drone.
    """
    total = max(1, int(seconds * rate))
    attack = max(1, int(0.008 * rate))
    for index in range(total):
        position = index / rate
        envelope = min(1.0, index / attack) * math.exp(-3.2 * index / total)
        tone = math.sin(2.0 * math.pi * frequency * position)
        octave = 0.28 * math.sin(4.0 * math.pi * frequency * position)
        yield envelope * (tone + octave) / 1.28


def _cue(pcm: bytes, rate: int) -> bytes:
    """Bound-check one generated cue and wrap it in the shared WAV container."""
    seconds = len(pcm) / float(2 * CHANNELS * rate)
    if not 0.0 < seconds <= MAX_CUE_SECONDS:
        # Not an assert: this is the bound that keeps a decorative sound from
        # holding the speaker (and the microphone) shut, and it must survive
        # `python -O`.
        raise RuntimeError(
            f"generated cue is {seconds:.2f}s, outside (0, {MAX_CUE_SECONDS}]"
        )
    return encode_wav(pcm, rate, CHANNELS)


@lru_cache(maxsize=1)
def engagement_ack_wav() -> bytes:
    """The acknowledgement SFX played with the IDLE -> ENGAGED motion.

    Cached: it is byte-identical every time, and the live loop asks for it on a
    control-thread tick where a few milliseconds of arithmetic is not free.
    """
    return _cue(
        _pcm16(_chirp(ENGAGEMENT_ACK_SECONDS, _ACK_START_HZ, _ACK_END_HZ, CUE_SAMPLE_RATE)),
        CUE_SAMPLE_RATE,
    )


@lru_cache(maxsize=1)
def goal_success_wav() -> bytes:
    """The musical flourish played once a verified goal has lit its target."""
    samples: list[float] = []
    for frequency, seconds in GOAL_SUCCESS_NOTES:
        samples.extend(_note(frequency, seconds, CUE_SAMPLE_RATE))
    return _cue(_pcm16(samples), CUE_SAMPLE_RATE)
