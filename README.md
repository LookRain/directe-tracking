# directe-tracking

Climber tracking for climbing video: find the climber you tapped and keep them, frame to frame, through belayers and
people walking past, a neighbour on the next route, chalk clouds, occlusion, falls and a hand-held camera. It runs
on an iPhone's Neural Engine, several times faster than real time (about 6x on an iPhone 15 Pro Max).

This is the tracker inside Directe, an iPhone app that turns a climbing video into a vertical video that follows
the climber. The app itself (framing, rest detection, export, UI) is not part of this repository.

## What's here

| Path | What |
|---|---|
| `Sources/TrackerCore` | Pure Swift. `TargetTracker`: who the climber is, tick to tick, with Locked / Uncertain / Lost states; the tag at the tap; per-climber profiles (colour by body region, re-ID embeddings, a gallery of other viewpoints, banked negatives); fall, ground and tie-in rules; camera-motion compensation. |
| `Sources/Perception` | Core ML person detection (YOLO26 pose, optional masks), re-ID embeddings (OSNet), colour features, "is the wall moving here" motion evidence and whole-image camera shift. Turns a frame into the tracker's candidates. |
| `python/directe_tracking` | The reference implementation in Python (NumPy, OpenCV, Ultralytics, boxmot). The Swift modules are bit-exact ports of it. This is where the tracker is developed and evaluated on recorded clips. |
| `tools/export_coreml.py` | Exports the upstream weights to the Core ML models Perception loads (FP16, raw heads, NMS in Swift; multifunction models for the three input shapes). |
| `tools/coreml_bench.py` | Times the models on a Mac's Neural Engine. |
| `tools/golden/tracker_golden.py` | Replays the Python tracker on synthetic scenes and writes the fixtures the Swift tests replay bit-exactly. |

## How it tracks

Every tick (10 per second of video) Perception finds the people in a wide view of the frame and in a 2x focus window
around where the climber should be, and describes each one: box, pose keypoints, colour per body region and a re-ID
embedding. `TargetTracker` scores each candidate against the climber's profile (appearance, position against a
motion prediction, size) and either keeps the lock, goes Uncertain (holds the prediction, widens its search, learns
nothing), or Lost (searches the whole frame and only re-locks on a confident match). Things that make gym footage
hard each have a rule:

- **Belayers and people on the ground:** once the climber is up the wall, nobody at ground level is them (the
  ground line moves with the camera).
- **Look-alikes on the next route:** a jump guard refuses sudden moves to someone else, rivals are banked as
  negatives, and the tie-in phase watches who actually leaves the ground.
- **Falls:** a fast drop is followed (size and acceleration checks) instead of being taken for a swap.
- **Hand-held video:** phase correlation measures the camera's own motion between ticks and the tracker moves its
  prediction with it; when the shift is unknown it infers it from the people.
- **Missed detections:** a second chance looks for a person where the wall is moving.

## Use it from Swift

```swift
// Package.swift
.package(url: "https://github.com/LookRain/directe-tracking", branch: "main"),
// target dependencies
.product(name: "TrackerCore", package: "directe-tracking"),
.product(name: "Perception", package: "directe-tracking"),
```

```swift
import Perception  // re-exports TrackerCore

let models = try PerceptionModels(.mid, directory: modelsFolder)  // compiled models, see "Models"
let perceiver = Perceiver(models: models)

// Tag: perceive ~1.5 s of frames from the tap, then start the tracker on the person at the tap.
let (tracker, _, _) = try TargetTracker.start(window: window, point: tap, frameHeight: H, config: .combo)

// Then per tick: camera motion, where to look, perceive, step.
tracker.camera(shift: shift, conf: conf)
let predicted = tracker.predictedBox(at: t)
let candidates = try perceiver.perceive(frame, focus: Perceiver.focusWindow(around: predicted, width: W, height: H))
let point = tracker.step(t: t, view: candidates)   // .locked / .uncertain / .lost, and the climber's box
```

`TargetTracker`'s documentation (Sources/TrackerCore/TargetTracker.swift) gives the full order of calls.
Frames are 32BGRA `CVPixelBuffer`s in upright source pixels.

## Models

No weights are included. `tools/export_coreml.py` downloads them from upstream and exports them:

- **Pose and masks:** [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) (`yolo26n/m-pose`,
  `yolo26n/m-seg`), licensed AGPL-3.0 by Ultralytics.
- **Re-ID:** OSNet from [boxmot](https://github.com/mikel-brostrom/boxmot) / torchreid (`osnet_x0_25_msmt17`,
  `osnet_ain_x1_0_msmt17`, `osnet_ibn_x1_0_msdc`). The weights were trained on research datasets (MSMT17 and
  others) with their own terms.

Check those licences before you ship the weights in a product.

```sh
uv run --project python --extra reid --extra coreml python tools/export_coreml.py
```

`$DIRECTE_DATA` (default `./data`) is where models, cached detections and runs go.

## Tests

```sh
swift test                                       # TrackerCore and Perception
uv run --project python python -m directe_tracking.tracker      # the Python tracker's selftest
uv run --project python python tools/golden/tracker_golden.py   # regenerate the synthetic fixtures
```

The fixtures here are synthetic scenes. The same harness also replays cases recorded on real footage and the
Perception goldens (which need the compiled models and decoded frames); those stay with whoever filmed the footage.
Point the tests at them with `DIRECTE_GOLDEN=/path/to/fixtures DIRECTE_DATA=/path/to/data swift test`.

## Python

```sh
uv sync --project python --extra reid
```

Clips are listed in a `footage.toml` (see `footage.example.toml`; `$DIRECTE_FOOTAGE` to put it elsewhere), videos
under `$DIRECTE_DATA/footage/raw`. Hand-checked identity labels (`labels.py`) are what tracker changes are judged on:
every few seconds, which box is the climber, or none.

## Contributing

Issues and pull requests are welcome. Contributions need the [Contributor Licence Agreement](CLA.md): it lets the
project keep shipping inside the Directe app while this repository stays AGPL. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

[GNU Affero General Public License v3.0](LICENSE). Copyright © 2026 LookRain.
