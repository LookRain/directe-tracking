"""Pinhole geometry for the simulator: a virtual camera rotating inside a wide source frame.

A gimbal rotates the phone about (roughly) its optical centre, so the view it would see is a pure
rotation of the wide shot: x_view ~ K_view · Rᵀ · K_src⁻¹ · x_src. That's exact for rotation, not
an approximation by cropping.

Angles: yaw > 0 looks right, pitch > 0 looks up. Pan is the outer axis, tilt the inner one.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class Lens:
    def __init__(self, size: tuple[int, int], fov_long_deg: float):
        self.W, self.H = size
        self.f = (max(self.W, self.H) / 2) / math.tan(math.radians(fov_long_deg) / 2)
        self.K = np.array([[self.f, 0, self.W / 2], [0, self.f, self.H / 2], [0, 0, 1]], dtype=np.float64)
        self.Kinv = np.linalg.inv(self.K)
        self.fov = (math.degrees(2 * math.atan(self.W / 2 / self.f)),
                    math.degrees(2 * math.atan(self.H / 2 / self.f)))

    def angles(self, uv: np.ndarray) -> np.ndarray:
        """Pixel(s) [..., 2] -> (azimuth, elevation) in degrees, in this lens's frame."""
        uv = np.asarray(uv, dtype=np.float64)
        x = (uv[..., 0] - self.W / 2) / self.f
        y = (uv[..., 1] - self.H / 2) / self.f
        az = np.degrees(np.arctan2(x, 1.0))
        el = np.degrees(np.arctan2(-y, np.sqrt(1.0 + x * x)))
        return np.stack([az, el], axis=-1)


def rotation(yaw: float, pitch: float) -> np.ndarray:
    y, p = math.radians(yaw), math.radians(pitch)
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rp = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return Ry @ Rp


def _apply(Hm: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.concatenate([pts, np.ones((*pts.shape[:-1], 1))], axis=-1) @ Hm.T
    w = p[..., 2:3]
    return p[..., :2] / np.where(np.abs(w) < 1e-9, 1e-9, w), w[..., 0] > 0


def _corners(boxes: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([np.stack([x0, y0], 1), np.stack([x1, y0], 1), np.stack([x1, y1], 1),
                     np.stack([x0, y1], 1)], 1)


class VirtualCamera:
    def __init__(self, src: Lens, out_size: tuple[int, int], zoom: float):
        self.src = src
        fov_long_src = 2 * math.degrees(math.atan(max(src.W, src.H) / 2 / src.f))
        fov_long = 2 * math.degrees(math.atan(math.tan(math.radians(fov_long_src) / 2) / zoom))
        self.view = Lens(out_size, fov_long)
        self.zoom = zoom

    def src_to_view(self, yaw: float, pitch: float) -> np.ndarray:
        return self.view.K @ rotation(yaw, pitch).T @ self.src.Kinv

    def view_to_src(self, yaw: float, pitch: float) -> np.ndarray:
        return self.src.K @ rotation(yaw, pitch) @ self.view.Kinv

    def render(self, frame: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
        return cv2.warpPerspective(frame, self.src_to_view(yaw, pitch), (self.view.W, self.view.H),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def boxes_to_view(self, boxes: np.ndarray, yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
        """World boxes -> (view boxes, fraction of each box inside the view)."""
        if len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        pts, front = _apply(self.src_to_view(yaw, pitch), _corners(boxes))
        vb = np.concatenate([pts.min(axis=1), pts.max(axis=1)], axis=1)
        cx0 = np.clip(vb[:, 0], 0, self.view.W); cx1 = np.clip(vb[:, 2], 0, self.view.W)
        cy0 = np.clip(vb[:, 1], 0, self.view.H); cy1 = np.clip(vb[:, 3], 0, self.view.H)
        area = np.maximum((vb[:, 2] - vb[:, 0]) * (vb[:, 3] - vb[:, 1]), 1e-6)
        vis = (cx1 - cx0) * (cy1 - cy0) / area
        vis[~front.all(axis=1)] = 0.0
        return vb, vis

    def boxes_to_world(self, vboxes: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
        if len(vboxes) == 0:
            return np.zeros((0, 4))
        pts, _ = _apply(self.view_to_src(yaw, pitch), _corners(vboxes))
        return np.concatenate([pts.min(axis=1), pts.max(axis=1)], axis=1)

    def outline_in_world(self, yaw: float, pitch: float) -> np.ndarray:
        W, H = self.view.W, self.view.H
        edge = np.concatenate([np.stack([np.linspace(0, W, 9), np.zeros(9)], 1),
                               np.stack([np.full(9, W), np.linspace(0, H, 9)], 1),
                               np.stack([np.linspace(W, 0, 9), np.full(9, H)], 1),
                               np.stack([np.zeros(9), np.linspace(H, 0, 9)], 1)])
        pts, _ = _apply(self.view_to_src(yaw, pitch), edge)
        return pts

    def framing_offset(self, y_frac: float) -> float:
        """Pitch offset (deg) that puts the aim point at y_frac of the view height."""
        return math.degrees(math.atan((y_frac - 0.5) * self.view.H / self.view.f))
