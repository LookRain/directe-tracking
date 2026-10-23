"""Model sets. "full" measures best-case accuracy in the lab; "lite" mirrors what an iPhone would run.

Every command reads and writes caches for the selected pipeline, so the same tracker, labels and
benchmark score either set. Select with DIRECTE_PIPELINE.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Pipeline:
    name: str
    pose: str          # Ultralytics pose weights
    pose_imgsz: int
    seg: str           # person masks; on the phone Apple Vision's person instance masks
    seg_imgsz: int
    reid: str          # boxmot re-ID weights

    @property
    def id_suffix(self) -> str:
        return "" if self.name == "full" else f"_{self.name}"


PIPELINES = {
    "full": Pipeline("full", "yolo26m-pose.pt", 1280, "yolo26m-seg.pt", 1280, "osnet_ain_x1_0_msmt17.pt"),
    "lite": Pipeline("lite", "yolo26n-pose.pt", 640, "yolo26n-seg.pt", 640, "osnet_x0_25_msmt17.pt"),
    # lite detectors + the full re-ID model (0.61 vs 0.23 ms on the Neural Engine): lite re-ID
    # barely separates people (outside review F3)
    "lite_ain": Pipeline("lite_ain", "yolo26n-pose.pt", 640, "yolo26n-seg.pt", 640, "osnet_ain_x1_0_msmt17.pt"),
    # app size is no constraint: lab-size pose and mask models at phone resolution (the focus window
    # is 640 px), with the re-ID model that separates people best when masks are good
    "mid": Pipeline("mid", "yolo26m-pose.pt", 640, "yolo26m-seg.pt", 640, "osnet_ain_x1_0_msmt17.pt"),
    "small": Pipeline("small", "yolo26s-pose.pt", 960, "yolo26n-seg.pt", 640, "osnet_x0_25_msmt17.pt"),
}


def current() -> Pipeline:
    name = os.environ.get("DIRECTE_PIPELINE", "full")
    if name not in PIPELINES:
        raise SystemExit(f"unknown pipeline {name}; known: {', '.join(PIPELINES)}")
    return PIPELINES[name]
