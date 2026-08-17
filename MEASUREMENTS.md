# Measurement procedure

The four numbers [`SUBMISSION.md`](SUBMISSION.md) asks for — engagement
reliability, response latency, CPU and memory — and exactly how to collect them.

These are **small empirical demo trials on one machine**, not benchmarks. Each
section says what it does and does not establish. The result tables below are
deliberately empty: nothing is filled in until it has actually been run, and a
row must name the machine it was run on.

Setup is in [`README.md`](README.md). Every trial assumes the venv is active.

---

## 0. Record the context first

Every table below is meaningless without the machine. Capture it once per run:

```bash
lsb_release -d                      # distribution and version
uname -srm                          # kernel and architecture
nproc                               # CPU core count
free -m | awk '/Mem:/ {print $2" MB total"}'
python --version
python -c "import cv2, pybullet; print('opencv', cv2.__version__)"
```

And note by hand, because no command reports them:

- PyBullet **GUI or `--headless`**
- camera active (yes/no) and which index
- voice enabled (full run) or `--no-voice`
- idle vs. mid-interaction
- laptop on mains or battery (governors throttle differently)

On macOS, `sysctl -n hw.ncpu` and `sw_vers` replace `nproc` and `lsb_release`.

---

## 1. Engagement reliability

**What this measures:** whether the deliberate act of turning to face the lamp
produces `IDLE -> ENGAGED`, and whether deliberately turning away produces
`ENGAGED -> IDLE`, within a few seconds, for one person at one desk.

**What it does not measure:** gaze. The perception layer is an OpenCV frontal
Haar-cascade proxy for *forward-facing presence*, with hysteresis on top — about
0.8 s of facing to engage, about 2.5 s of absence to disengage. It cannot tell
where the eyes are pointing, and a face turned far enough in profile stops being
detected at all, which is precisely why "look away" works as a disengagement
cue. No general claim about attention estimation is being made, and the
perception algorithm is **not** to be tuned for this trial.

### Procedure

```bash
python src/engagement_demo.py --no-voice
```

`--no-voice` on purpose: engagement is entirely local, and removing the speech
path removes network variance and API cost from a trial that does not need
either. The preview window is left on so the detector's own view is visible.

Sit at a normal working distance under the room's ordinary lighting. Then, ten
times:

1. Start in `IDLE`, facing away from the laptop.
2. Turn to face the camera and hold. Wait for `IDLE -> ENGAGED`.
3. Turn clearly away (or leave the frame). Wait for `ENGAGED -> IDLE`.
4. Note anything that fired when you did not ask for it.

Count an attempt as successful if the transition happens within ~5 s of the
deliberate act. Count anything else as a failure and say which kind. Keep the
terminal log — the timestamped `IDLE -> ENGAGED` / `ENGAGED -> IDLE` lines are
the raw record, and they now carry milliseconds.

Log the run to a file and keep it with the results:

```bash
python src/engagement_demo.py --no-voice 2>&1 | tee engagement-trial.log
```

### Results

| Field | Value |
|---|---|
| Machine / OS | _not yet collected_ |
| Camera | |
| Lighting / distance | |
| Successful engages | / 10 |
| Successful disengages | / 10 |
| False engages (nobody attending) | |
| False disengages (attending throughout) | |
| Notes | |

---

## 2. Spoken response latency

**What this measures:** a **user-facing end-to-end latency proxy** —

```text
recording-worker launch  ->  successful reply playback launch / handoff
```

read from the two log lines `listening...` and `said:`.

**Be precise about both endpoints:**

- `listening...` is printed on the control thread immediately *after* the
  background turn has been started, so the microphone may actually open a few
  milliseconds **before** this timestamp. The stamp is a close proxy for the
  start of recording, not an exact capture of it.
- `said:` is printed *after* the turn has successfully committed — that is,
  after the gesture (if any) has played and the prepared audio has been handed
  to the player. It marks the **handoff**, not the first audible sample. It does
  not include whatever latency the player process or the audio device adds
  before sound physically leaves the speaker.

So this is an end-to-end proxy for what a person waits through, **not** an
acoustic first-audible-sample measurement. Say so when reporting it.

**What the interval contains,** so the number is not misread as model latency:

- a **fixed 4.0-second bounded recording** — there is no voice-activity
  detection, so the microphone always captures the full window before anything
  is sent. This floor is a design choice, it dominates a fast turn, and it must
  be reported alongside the total rather than silently subtracted;
- speech-to-text;
- the character reasoning call;
- text-to-speech;
- audio decoding and the output-device check;
- **the named gesture, when the model chose one** — `nod` or `engage` play on
  the control thread before playback starts, so a turn with `behavior=nod` has
  that motion inside the measured interval and a turn with `behavior=none` does
  not. Record the behaviour alongside each value so the two are distinguishable.

### Procedure

Run the full demo and use the log's own timestamps — no extra tooling:

```bash
python src/engagement_demo.py 2>&1 | tee latency-trial.log
```

Engage, then take roughly five representative spoken turns — ordinary short
questions, including at least one scene-memory recall. For each turn, take the
two lines and subtract the timestamps.

> **Illustrative format only — these are invented lines, not a measurement.**
> No timing in this document has been observed.
>
> ```text
> [14:03:27.412] listening...
> [14:03:34.907] said:  "..."  behavior=nod
> ```
>
> The arithmetic for such a pair would be 34.907 − 27.412 = 7.495 s. The real
> values go in the table below, once real turns have been run.

Timestamps carry milliseconds specifically so this subtraction is credible; at
whole-second resolution the rounding either side would be a large fraction of a
short turn.

**Exclude unsuccessful turns from the summary.** A turn that logged
`turn failed:` or `discarded a stale turn` never committed playback, so it has
no `said:` line and no latency to report. Count those separately and state how
many occurred — the mean/min/max are over successful turns only, and a run with
several failures says something the latency figure alone does not.

### Results

Successful turns only:

| Turn | What was said | behavior | listening... | said: | Δ (s) |
|---:|---|---|---|---|---:|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

| | seconds |
|---|---:|
| mean | _not yet collected_ |
| min | |
| max | |

Failed or stale turns during the run (excluded from the summary above): _not yet
recorded._

Machine, network and whether the run was on Wi-Fi: _not yet recorded._

---

## 3. CPU and memory

**What this measures:** what the single demo process costs on the target class
of machine, idle and during a representative interaction.

Standard process tools only — adding a Python monitoring dependency to measure
the thing it is running inside would change the measurement.

### Procedure

Start the demo in one terminal:

```bash
python src/engagement_demo.py
```

In a second terminal, find it and sample it:

```bash
pgrep -af 'python .*engagement_demo.py'
```

#### Process context and memory — `ps`

`ps` is used for PID confirmation, a one-shot RSS snapshot and general context.
`RSS` is resident set size in **kilobytes**; divide by 1024 for MB.

```bash
PID=$(pgrep -f engagement_demo.py | head -1)
ps -o pid,%cpu,%mem,rss,etime,comm -p "$PID"
```

**Do not use this `%CPU` column as the CPU measurement, and do not sample it in
a loop.** `ps %cpu` is the average over the process's entire lifetime, not over
the moment it was sampled. Reading it repeatedly does not produce per-second
samples — it produces a slowly-moving lifetime average that still has process
startup baked into it, and it cannot distinguish an idle window from an active
one at all.

#### CPU — `top`

`top` reports CPU over the interval between refreshes, which is the quantity
actually wanted:

```bash
top -b -d 1 -n 31 -p "$PID"
```

- `-b` batch mode, so the output can be piped or saved.
- `-d 1` one second between refreshes.
- `-n 31` thirty-one refreshes.
- **Discard the first refresh.** Its `%CPU` is computed against process start,
  so it carries the same lifetime-average problem as `ps`.
- The **remaining 30 refreshes** are the ~30-second sample. Summarise their
  `%CPU` for the condition under test (mean, and min/max if they spread widely).

`%CPU` here is percent of a **single** core, so on the 4-core target 400 % is
the ceiling and values above 100 % are normal for a multi-threaded process.

Save the sample alongside the results:

```bash
top -b -d 1 -n 31 -p "$PID" > cpu-gui-idle.log
```

Take a separate ~30 s sample for each condition:

- **Idle** — engaged or not, but no spoken turn, no observation, no goal in
  flight. The perception loop, PyBullet and the GUI only.
- **Active** — during a representative stretch: engage, one spoken turn, one
  observation, one find-and-light goal.

Repeat with `--headless`. The point of that pair is to **measure how much of the
total the PyBullet GUI accounts for** — the renderer is a presentation choice
rather than a requirement, so its share is worth knowing separately. Which
condition costs more is a question for the data; do not assume an answer before
the samples exist.

Peak RSS for a whole run, where useful:

```bash
/usr/bin/time -v python src/engagement_demo.py --no-voice
# read "Maximum resident set size (kbytes)"
```

### Results

`%CPU` is the mean of the 30 retained `top` refreshes, in percent of one core.
`RSS` is from `ps`, converted to MB.

| Condition | mean %CPU (of one core) | RSS (MB) |
|---|---:|---:|
| GUI, camera, voice — idle | _not yet collected_ | |
| GUI, camera, voice — active | | |
| `--headless`, camera, voice — idle | | |
| `--no-voice`, GUI — idle | | |

Context (from §0): _not yet recorded._

---

## Status

**No measurements in this file have been collected yet.** Every table is a
placeholder, and the only numbers appearing anywhere above are inside a block
explicitly labelled as an invented formatting example. The real numbers must come
from a real run on a stated machine, and the Ubuntu 24.04 rows must come from
actual Ubuntu 24.04 hardware — or a VM with the camera and audio devices passed
through — rather than being extrapolated from a development machine running
another OS.

Once collected, these four measurements feed the ≤2-page technical note required
by [`SUBMISSION.md`](SUBMISSION.md). That note has not been written yet; it is
the final submission pass, and it is deliberately deferred until the evidence
here is real.
