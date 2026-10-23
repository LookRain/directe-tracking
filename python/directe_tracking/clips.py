"""Clip registry and data-root paths. Large files never leave the data root.

$DIRECTE_FOOTAGE: the clip registry (footage.toml; default ./footage.toml), one table per clip (see
footage.example.toml). $DIRECTE_DATA: the data root (default ./data): footage/raw/<file>, runs/, models/, cache/.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

FOOTAGE = Path(os.environ.get("DIRECTE_FOOTAGE", "footage.toml"))


def data_root() -> Path:
    root = Path(os.environ.get("DIRECTE_DATA", "data"))
    if not root.is_dir():
        raise SystemExit(f"data root {root} not found (set DIRECTE_DATA)")
    return root


@dataclass
class Clip:
    id: str
    file: str
    desc: str
    target: str
    seed: tuple[float, float, float]
    fov_long: float
    handheld: bool
    register: str = "tap"  # "tap": operator taps at `seed`; "pose": the climber holds the L pose

    @property
    def path(self) -> Path:
        return data_root() / "footage" / "raw" / self.file

    @property
    def run_dir(self) -> Path:
        d = data_root() / "runs" / self.id
        d.mkdir(parents=True, exist_ok=True)
        return d


def load(clip_id: str) -> Clip:
    reg = tomllib.loads(FOOTAGE.read_text())
    if clip_id not in reg:
        raise SystemExit(f"unknown clip {clip_id}; known: {', '.join(reg)}")
    c = reg[clip_id]
    return Clip(clip_id, c["file"], c["desc"], c["target"], tuple(c.get("seed", (0, 0.5, 0.5))),
                c["fov_long"], c["handheld"], c.get("register", "tap"))


def all_ids() -> list[str]:
    return list(tomllib.loads(FOOTAGE.read_text()))
