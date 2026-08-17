# Live character lamp — setup and run

A single aware character built around the supplied fictional five-DOF lamp
robot: it watches the laptop camera for someone paying attention, acknowledges
them with **motion and a sound effect**, holds a spoken conversation through the
laptop microphone and speaker, remembers one object it was shown, and can be
asked out loud to find that object and shine its light on it. The **bulb is
reserved for a successfully verified goal** — it is not part of the greeting,
and music plays only when such a goal completes.

The challenge brief is [`CHALLENGE.md`](CHALLENGE.md); the supplied deliverables
and evaluation criteria are in [`SUBMISSION.md`](SUBMISSION.md). Measurement
procedures and the place to record the numbers are in
[`MEASUREMENTS.md`](MEASUREMENTS.md).

## Requirements

Target: a clean **Ubuntu 24.04 LTS** laptop, four CPU cores, 8 GB RAM, no
discrete GPU and no CUDA. Development and the demonstration may happen on
another OS; everything below is written for the Ubuntu target.

Hardware and services:

- Integrated or USB **camera** (OpenCV device index 0 by default).
- **Microphone** and **speaker** on the standard Linux audio stack
  (PipeWire/PulseAudio on a stock 24.04 desktop, or bare ALSA).
- **Wi-Fi / internet**, because speech-to-text, character reasoning,
  text-to-speech and the two vision calls are cloud calls.
- An **`OPENAI_API_KEY`**. Only [`src/character.py`](src/character.py) talks to
  OpenAI; nothing else in the tree makes a network call.

Python packages are declared in [`requirements.txt`](requirements.txt) (runtime)
and [`requirements-dev.txt`](requirements-dev.txt) (adds pytest). They are the
single source of truth for versions — this README does not restate them.

System packages:

```bash
sudo apt update
sudo apt install python3-venv python3-dev build-essential libportaudio2 pulseaudio-utils
```

- `python3-venv` — Ubuntu 24.04's default Python is 3.12, and it splits `venv`
  into its own package.
- `python3-dev` and `build-essential` — the Python headers and the C/C++
  toolchain. PyBullet may have to compile from source when no wheel matching
  this Python version and platform is available, and without these that install
  fails. They cost nothing when a usable wheel *is* published.
- `libportaudio2` — the shared library behind `sounddevice`, used to record.
  The wheel does not bundle it; without it recording fails with an explicit
  "PortAudio could not be loaded" message.
- `pulseaudio-utils` — provides `paplay`. Playback prefers a **separate player
  process** over in-process PortAudio, because the PyBullet GUI renderer and
  sounddevice's stream callback contend for the GIL and the reply comes out as
  static (see the module docstring in [`src/audio_io.py`](src/audio_io.py)).
  Rather than rely on what a given Ubuntu image happens to include, the command
  above installs `pulseaudio-utils` explicitly, so `paplay` is present by
  construction.

Optional, only if you prefer bare ALSA or cannot use `pulseaudio-utils`:

```bash
sudo apt install alsa-utils        # provides aplay
```

The player is chosen at startup: the code looks for `paplay` first, then
`aplay`, and falls back to in-process PortAudio/`sounddevice` when neither is
found. That fallback is a real code path rather than a guess, but it has **not
been exercised on Ubuntu 24.04 hardware** — see *Known limitations and
validation status* below.

No ROS, no Gazebo, no Docker, no GPU runtime, no system-wide install.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

There are two requirement files, and they are not interchangeable:

| File | Contents | Use it when |
|---|---|---|
| `requirements.txt` | Runtime only: pybullet, opencv-python, openai, sounddevice. | You want to **run** the demo. |
| `requirements-dev.txt` | Everything in `requirements.txt`, **plus pytest**. | You want to **run the test suite**. |

`requirements.txt` does **not** install pytest, so `python -m pytest` fails with
"No module named pytest" after a runtime-only install. To run the tests:

```bash
python -m pip install -r requirements-dev.txt
```

That is a superset — it installs the runtime packages too, so it is also a fine
single command for a machine that will both run and test.

Provide the credential through the environment, not a file:

```bash
export OPENAI_API_KEY="sk-..."
```

Nothing in the tree reads a `.env`, and the key is redacted from any error
message the demo prints. Keep it out of the repository.

## Run

The complete interaction — engagement, spoken turns, scene memory and the
goal-directed action — is one entrypoint:

```bash
python src/engagement_demo.py
```

Options (all optional):

| Flag | Effect |
|---|---|
| `--camera-index N` | Use another webcam. Default `0`. |
| `--no-preview` | Hide the OpenCV camera window. The PyBullet window stays. |
| `--headless` | Run PyBullet with no window (`DIRECT`). No lamp GUI, so no lamp-window keys. |
| `--no-voice` | Camera and lamp only: no microphone, no speaker, no API key needed. |

`--no-voice` is a **diagnostic**, not the submitted interaction. It exercises
the camera, the engagement policy and the lamp body and nothing else — no
speech, no memory, no goal. Use it to prove the hardware half works before
spending API calls.

### Controls

- **Engage** — look towards the laptop camera and hold it for about a second.
  The lamp sits up and plays its acknowledgement sound effect — **motion plus
  SFX, no light** — and the log prints `IDLE -> ENGAGED`. Turn away or leave and
  after a couple of seconds it settles back down: `ENGAGED -> IDLE`.
- **Talk** (while ENGAGED) — the accepted key depends on which window has focus,
  because each input path is read differently:

  | Focus | Keys that start a turn |
  |---|---|
  | This terminal | **Enter**, or **`t`** followed by Enter |
  | PyBullet lamp window | **Enter**, **Space**, or **`t`** |
  | Camera preview window | **`t`** or **Enter** |

  Space is accepted **only** in the lamp window; the terminal and preview paths
  do not read it. It records one bounded 4-second utterance, answers through the
  speaker, and may play one named gesture.
- **Observe** (while ENGAGED) — press **`o`** in the lamp window or the camera
  preview. One frame goes to the vision call and what comes back is stored in
  local scene memory. This is the only keypress that sends a picture anywhere.
  It is deliberately not a spoken command.
- **Recall** — later, ask about that object out loud. The answer comes from the
  stored note; no camera is involved, so it still works after the object has
  been moved or taken away.
- **Goal** (while ENGAGED) — say something like *"find my mug and shine your
  light on it"*. The character takes a fresh frame, finds the target, turns
  towards the side it is on, takes a **second** fresh frame after turning, and
  lights the bulb only if the target is still confidently on the side it aimed
  at. Otherwise the light stays off and it says so.
- **Quit** — **`q`** or **Esc** in the camera preview, or **Ctrl-C** in the
  terminal. Ctrl-C is the only way out with `--no-preview`. Either way the light
  is put out on the way down.

There is also a body-only motion demo with no camera and no network:

```bash
python src/demo.py              # neutral -> engage -> left -> right -> nod -> neutral
python src/demo.py --headless
```

### Optional environment variables

- `LAMP_AUDIO_BACKEND=system|sounddevice` — force a playback backend instead of
  letting it choose. Useful for isolating audio problems.
- `LAMP_VISION_MODEL=...` — override the model used for the observation and
  goal-localisation calls only.

## Validation

Run these in order on a fresh Ubuntu box; each one clears a different class of
problem.

```bash
python -m pip install -r requirements-dev.txt   # pytest is NOT in requirements.txt
python -m pytest
```
The whole behavioural suite. No camera, no microphone, no network, no API key —
every collaborator is faked. This proves the logic: engagement hysteresis, the
model-to-action allowlist, the two-look goal semantics, the fail-closed light,
recall separation, camera freshness and speaker ordering. If this fails, nothing
below is worth trying.

```bash
python src/engagement_demo.py --no-voice
```
Proves the local half on real hardware: the camera opens at the requested index,
faces are detected, `IDLE -> ENGAGED` and `ENGAGED -> IDLE` fire as you look at
and away from the lens, and PyBullet renders and moves the lamp. Needs no key
and no network.

```bash
python src/engagement_demo.py
```
The full interaction, and the only command that exercises the cloud path. It
needs `OPENAI_API_KEY` and working internet; it also needs a real microphone and
speaker, which `--no-voice` never touches. A failure here that `--no-voice`
passes is an audio-device, credential or network problem, not a logic one.

## Data flow and the cloud boundary

```text
camera / microphone  ->  character orchestrator  ->  scene memory + reasoning
                                 |
                                 v
                       semantic action intent
                                 |
                                 v
                    LampController  ->  PyBullet lamp (motion + light)
```

What leaves the machine, and when:

- **Microphone audio** — one bounded 4-second WAV per push-to-talk, sent to
  speech-to-text.
- **Transcript plus the local scene-memory note as text** — sent to the language
  call, which returns a reply, one behaviour word and at most one goal.
- **Reply text** — sent to text-to-speech; the audio comes back and is played
  through the laptop speaker.
- **One bounded camera frame** — only on an explicit `o` press, for the
  observation.
- **Fresh bounded camera frames** — during a goal, for the first and second
  localisation looks of the named target.
- **Recall sends nothing new.** A later question about a remembered object is
  answered from the retained text; no image is re-sent.
- **Joint commands never leave the process.** Everything physical goes through
  `LampController`, which owns joint discovery, limits and the named behaviours.
  The model chooses among a fixed vocabulary of named behaviours and one goal;
  it never produces a joint value.

Frames are downscaled to 640 px wide and JPEG-compressed with a hard byte cap
before any of this, so an upload cannot grow without bound.

## Documentation status

[`SUBMISSION.md`](SUBMISSION.md) is the **supplied** submission requirements and
evaluation-criteria document. It is **not** the technical note.

The required ≤2-page technical note — the architecture and data-flow diagram,
the protocol/model-to-action/simulation/deployment design choices, the four
measurements and the limitations — is **still pending**. It is written in the
final submission pass, after the measurements in
[`MEASUREMENTS.md`](MEASUREMENTS.md) have actually been collected on real
hardware, so that its numbers are measured rather than asserted.

## Known limitations and validation status

The behavioural logic is covered by the test suite, which runs entirely on fakes.
What that suite cannot cover, and what therefore remains unvalidated:

- **The Ubuntu 24.04 target has not been validated on real target hardware.**
  The apt list, the PyBullet install path and everything below are reasoned from
  the code and the packages' documented contents, not observed on a 24.04 box.
- **Linux audio needs real-hardware validation.** Which backend is selected,
  and whether the in-process PortAudio fallback is acceptable under the PyBullet
  GUI, is a property of the machine and has not been tested on the target.
- **Camera index 0 may not be the right webcam.** USB cameras frequently enumerate
  at index 1 or higher on Linux; use `--camera-index N`. There is no auto-probe.
- **Default microphone and output device selection is machine-dependent.** The
  code takes the system defaults and does not offer device selection.
- **Engagement is an approximate frontal-face / forward-facing-presence proxy,
  not eye-gaze estimation.** A Haar cascade with hysteresis. It cannot tell where
  the eyes are pointing.
- **Engagement behaviour depends on camera placement and lighting**, so results
  from one desk do not transfer to another.
- **CPU and RSS on Ubuntu cannot be inferred from macOS or Windows** and must be
  measured on the target.
- **PyBullet GUI behaviour and resource use are platform-dependent**; the
  headless comparison in `MEASUREMENTS.md` exists to quantify that difference
  rather than to assume it.
