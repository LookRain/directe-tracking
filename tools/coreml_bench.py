"""Export the lite (and full, for comparison) models to Core ML and time them on this Mac.

Apple-silicon Macs have the same Neural Engine design as recent iPhones, so per-inference times
here are a reasonable stand-in for an iPhone 15 Pro / 16 Pro (roughly similar, not identical).

  uv run --project python --extra reid --extra coreml python tools/coreml_bench.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from directe_tracking.clips import data_root  # noqa: E402

def _patch_coremltools() -> None:
    """coremltools 9 calls int()/bool() on one-element arrays, which numpy >= 2.5 rejects."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops

    def _cast(context, node, dtype, dtype_name):
        x = ops._get_inputs(context, node, expected=1)[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")
        if x.can_be_folded_to_const():
            v = x.val
            res = x if isinstance(v, dtype) else mb.const(val=dtype(np.asarray(v).reshape(-1)[0].item()), name=node.name)
        elif len(x.shape) > 0:
            res = mb.cast(x=mb.squeeze(x=x, name=node.name + "_item"), dtype=dtype_name, name=node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    ops._cast = _cast


_patch_coremltools()
OUT = data_root() / "models" / "coreml"
OUT.mkdir(parents=True, exist_ok=True)


def export_yolo(weights: str, imgsz: int) -> Path:
    from ultralytics import YOLO

    target = OUT / f"{Path(weights).stem}_{imgsz}.mlpackage"
    if not target.exists():
        path = YOLO(str(data_root() / "models" / weights)).export(format="coreml", imgsz=imgsz, half=True, nms=False)
        os.replace(path, target)
    return target


def export_reid(weights: str) -> Path:
    import coremltools as ct
    import torch
    from boxmot.reid.core.runtime import ReID

    target = OUT / f"{Path(weights).stem}.mlpackage"
    if target.exists():
        return target
    backend = ReID(weights=data_root() / "models" / weights, device="cpu").model
    net = backend.model.eval()

    class Fixed(torch.nn.Module):
        """OSNet's head computes its pooling size dynamically, which Core ML can't convert;
        a fixed-shape mean over the feature map is the same operation for a 256x128 input."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            f = self.m.featuremaps(x).mean(dim=(2, 3))
            return self.m.fc(f) if getattr(self.m, "fc", None) is not None else f

    x = torch.rand(1, 3, 256, 128)
    wrapped = Fixed(net).eval()
    with torch.no_grad():
        ref = net(x)
        got = wrapped(x)
    assert torch.allclose(ref, got, atol=1e-4), float((ref - got).abs().max())
    traced = torch.jit.trace(wrapped, x)
    ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape)], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14)
    ml.save(str(target))
    return target


def time_model(path: Path, units: str, n: int = 60) -> dict:
    import coremltools as ct
    from PIL import Image

    cu = {"ALL": ct.ComputeUnit.ALL, "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE, "CPU_ONLY": ct.ComputeUnit.CPU_ONLY}[units]
    m = ct.models.MLModel(str(path), compute_units=cu)
    spec = m.get_spec().description.input[0]
    if spec.type.WhichOneof("Type") == "imageType":
        w, h = spec.type.imageType.width, spec.type.imageType.height
        inp = {spec.name: Image.fromarray(np.random.randint(0, 255, (h, w, 3), np.uint8))}
    else:
        shape = tuple(spec.type.multiArrayType.shape)
        inp = {spec.name: np.random.rand(*shape).astype(np.float32)}
    for _ in range(5):
        m.predict(inp)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        m.predict(inp)
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": round(statistics.median(ts), 2), "p90_ms": round(float(np.percentile(ts, 90)), 2)}


def main() -> None:
    models = {
        "lite pose (yolo26n, 640)": export_yolo("yolo26n-pose.pt", 640),
        "lite masks (yolo26n-seg, 640)": export_yolo("yolo26n-seg.pt", 640),
        "lite re-ID (osnet x0.25)": export_reid("osnet_x0_25_msmt17.pt"),
        "full pose (yolo26m, 1280)": export_yolo("yolo26m-pose.pt", 1280),
        "full re-ID (osnet-ain x1.0)": export_reid("osnet_ain_x1_0_msmt17.pt"),
    }
    results = {}
    for name, path in models.items():
        results[name] = {u: time_model(path, u) for u in ("CPU_AND_NE", "CPU_ONLY")}
        print(f"{name:32s} Neural Engine {results[name]['CPU_AND_NE']['median_ms']:7.1f} ms   "
              f"CPU only {results[name]['CPU_ONLY']['median_ms']:7.1f} ms")
    (data_root() / "runs" / "coreml_timing.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
