# Technical note — live character lamp

One aware character around the supplied five-DOF lamp: it notices someone, greets
them with motion and an SFX, holds spoken turns, remembers one object it was
shown, and can be asked to find that object and light it. Setup:
[`README.md`](README.md). Evidence: [`MEASUREMENTS.md`](MEASUREMENTS.md).

## Architecture and data flow

```text
camera + microphone
   |
   v                    bounded cloud requests (audio, text, one JPEG)
local perception / orchestrator / scene memory --> [restricted proxy] --> OpenAI
   |
   v   validated semantic intent
   |     behavior: none | nod | engage
   |     goal:     none | find_and_light(target)
   v
LampController (joint discovery, limits, named behaviours, light)
   |
   v
PyBullet motion + light
```

The proxy (reviewer mode only) sits on the cloud-request branch and cannot
control the robot: an optional evaluation transport, with direct OpenAI operation
unchanged by it.

## Protocol, perception and memory

Only `src/character.py` talks to OpenAI; it never imports PyBullet and has no
concept of a joint. Responses are validated locally despite the strict schema
requested — server-side enforcement is a convenience, not a trust boundary. The
two contracts above are the whole vocabulary, dispatched through a literal dict
of zero-argument `LampController` calls: no `getattr`, no `eval`, no tool
registry, no field that could carry an angle or a pose. The worst a confused
model can do is pick the wrong word from a short list of limit-clamped gestures.

Engagement is an OpenCV frontal-face Haar cascade used as an approximate
forward-facing-presence proxy — **not** gaze estimation — with hysteresis.
Pressing `o` sends one frame to the vision call and stores the validated fields
in a one-slot record; recall passes that record as *text* and has no camera in
it, so answers survive the object being moved. A spoken `find_and_light(target)`
takes a fresh frame, locates the target, turns toward `left`/`center`/`right`,
then takes a **second** fresh frame; the light comes on only when
`second.found and second.confident and second.location == expected_location`
(the side the first look reported); otherwise it stays off and the character says
so. The camera does not move with the head, so the second look is a consistency
check, not a centring check.

## Cloud / local split

**Local:** perception and hysteresis, orchestration, memory, semantic-action
validation, the goal state machine, `LampController`, all joint commands,
PyBullet, light control, media playback.

**Leaves the machine, only on request:** one 4 s recording per push-to-talk;
transcript plus retained memory text; reply text for synthesis; one JPEG on an
explicit `o`; the fresh JPEGs a goal needs — capped at 640 px and 400 kB, never
streamed. Hosted models are strongest at language and object identity, exactly
where a 4-core, no-GPU laptop is weakest; every decision that moves the robot or
lights the bulb stays local.

## Simulation, deployment and tradeoffs

PyBullet loads the supplied URDF fixed-base at 240 Hz with position control,
velocity scaling and smoothstep interpolation. Two loader artefacts are fixed in
code, not by editing the model: inertia-less semantic frame links are zeroed
(PyBullet's default 1 kg each would eat the shoulder's effort budget), and "lit"
is a recolour, since PyBullet has no light sources.

Deployment is a venv plus `requirements.txt` — no ROS, Gazebo, Docker or GPU.
Playback uses a separate player process rather than in-process PortAudio, whose
callback loses the GIL to the GUI renderer. Ubuntu 24.04 ARM64 was validated in a
4-core/8-GB VM (build, tests, headless runtime, all 5 joints); it exposes no
peripherals, so those were checked on macOS.

Key tradeoffs: **named behaviours, not joint-space model output** — costs range,
buys a boundary that cannot be talked past; **fixed 4 s push-to-talk, not VAD** —
costs latency, buys a predictable bound; **coarse left/center/right, not visual
servoing** — vision and motion share one vocabulary, and `unknown` is absent from
the orientation table, so "not sure" becomes "don't move" structurally.

## Measurements

Physical M3 Mac, full GUI + camera + voice demo. Small trials on one machine, not
benchmarks; full method and caveats in [`MEASUREMENTS.md`](MEASUREMENTS.md).

| | Result |
|---|---|
| Engagement, 10 cycles | 10/10 engage, 10/10 disengage, 0 false transitions |
| Spoken turn, end-to-end | mean 9.046 s (7.992–11.760), 5/5 successful |
| CPU | 122.1 % idle, 127.0 % active, of one core |
| Memory | active RSS ≈ 326 MB |

Latency is the user-facing proxy (recording launch → playback handoff), includes
the fixed 4 s window, and is not model-only latency. CPU is percent of one core.
No idle RSS was retained; Ubuntu CPU/RSS was not measured.

## Known limitations

Frontal-face detection is not gaze estimation, and lighting and camera placement
affect it — under poorer lighting than the trial's, engagement was informally seen
to drop while the user still faced the camera. The fixed 4 s recording dominates
latency; cloud connectivity is required; camera index and audio devices are
machine-dependent; no physical Ubuntu peripheral validation was possible.
Simulation is not hardware — contact, payload dynamics and real actuators are not
exercised. Localisation is coarse: no SLAM, no tracking, no manipulation, one
orientation attempt per goal. The reviewer proxy is evaluation convenience, not
robot control.
