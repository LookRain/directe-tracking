"""Frame sampling and H.264 writing (via ffmpeg)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def sample(path: Path, fps: float, start: float = 0.0, end: float | None = None
           ) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, seconds, BGR frame) at roughly `fps`, display-oriented."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    idx = int(round(start * src_fps))
    next_t = start
    try:
        while True:
            t = idx / src_fps
            if end is not None and t > end:
                return
            if t + 1e-6 >= next_t:
                ok, frame = cap.read()
                if not ok:
                    return
                yield idx, t, frame
                next_t += 1.0 / fps
            elif not cap.grab():
                return
            idx += 1
    finally:
        cap.release()


def frames_at(path: Path, indices) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, frame) for the given increasing source frame indices."""
    indices = [int(i) for i in indices]
    if not indices:
        return
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    pos = indices[0]
    try:
        for want in indices:
            while pos < want:
                if not cap.grab():
                    return
                pos += 1
            ok, frame = cap.read()
            if not ok:
                return
            pos += 1
            yield want, frame
    finally:
        cap.release()


def info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    out = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": frame.shape[1],
        "height": frame.shape[0],
    }
    cap.release()
    out["duration"] = out["frames"] / out["fps"]
    return out


class Writer:
    """Pipe BGR frames into ffmpeg; output is H.264 that QuickTime and browsers play."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        w, h = size
        self.size = (w - w % 2, h - h % 2)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{self.size[0]}x{self.size[1]}", "-r", f"{fps}", "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = frame[: self.size[1], : self.size[0]]
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
