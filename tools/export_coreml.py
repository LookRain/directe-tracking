"""Core ML models for Perception (Sources/Perception).

  uv run --project python --extra reid --extra coreml python tools/export_coreml.py [--force] [--check]

Downloads the upstream weights (Ultralytics YOLO26, boxmot's OSNet; check their licences before you ship them),
writes $DIRECTE_DATA/models/coreml/<name>.mlpackage and compiles each one into
$DIRECTE_DATA/models/coreml/app/<name>.mlmodelc, the folder `PerceptionModels(_:directory:)` loads from.
Names: <weights stem>_<H>x<W>, or _640 for 640x640.

  pose  yolo26n-pose, yolo26m-pose  at 640x640 (focus window, second chance) and at the two
        rectangles Ultralytics letterboxes a 16:9 frame into (`rect`, stride 32): 640x384 for a
        portrait frame, 384x640 for a landscape one (wide pass; other aspect ratios use 640x640)
  masks yolo26n-seg, yolo26m-seg    at 640x640
  re-ID osnet_x0_25_msmt17, osnet_ain_x1_0_msmt17 (masked crops), osnet_ibn_x1_0_msdc (plain crops) at 256x128

Each pose model's three input shapes are also merged into one multifunction model, <stem>_shapes (functions
s640x640, s640x384, s384x640): the same weights stored once (yolo26m pose 3 x 41 MB -> 41 MB), same outputs.
models/coreml/bundle gets the small set (ModelSet.lite: yolo26n pose + seg, OSNet x0.25, ~12 MB), what an app
can ship inside itself; the rest can be downloaded on demand.

YOLO26 has two heads: the NMS-free one-to-one head (what `export(nms=False)` bakes in, end2end=True,
top 300 rows of x0 y0 x1 y1 conf class ...) and the one-to-many head that needs NMS. The Python tracker runs
Ultralytics' predictor with its default nms=None, which (ultralytics 8.4) means end2end=False: the
one-to-many head + NMS (IoU 0.7). The two heads disagree noticeably (nano pose: box IoU down to 0.5,
conf off by up to 0.3 on the same input), so this exports that head: raw outputs, NMS in Swift.
Outputs (renamed here; N anchors = 8400 at 640x640, 5040 at 640x384):
  pose   detections [1, 56, N]: cx cy w h (input pixels), person score, 17 x (x, y, visibility)
  seg    detections [1, 116, N]: cx cy w h, 80 class scores, 32 mask coefficients
         protos     [1, 32, 160, 160]: mask prototypes (mask logit = coefficients . protos)
Image input "image" (RGB, scaled 1/255; feed a BGRA pixel buffer of exactly the input size).

OSNet: boxmot's preprocessing is resize to 128x256 (cv2 INTER_LINEAR), BGR->RGB, /255, ImageNet
mean/std. The exported model takes an RGB image "image" (128x256) and does the /255 and mean/std
itself; output "embedding" [1, 512] is not normalised (Swift L2-normalises, as boxmot does).
The head's dynamic pooling is replaced by a fixed mean (tools/coreml_bench.py).

FP16 everywhere: --check compares each model with the PyTorch reference on real frames.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coreml_bench import _patch_coremltools  # noqa: E402,F401  (importing it patches coremltools' _cast)

from directe_tracking.clips import data_root  # noqa: E402

OUT = data_root() / "models" / "coreml"
APP = OUT / "app"

POSE = ["yolo26n-pose.pt", "yolo26m-pose.pt"]
SEG = ["yolo26n-seg.pt", "yolo26m-seg.pt"]
REID = ["osnet_x0_25_msmt17.pt", "osnet_ain_x1_0_msmt17.pt", "osnet_ibn_x1_0_msdc.pt"]
POSE_SHAPES = [(640, 640), (640, 384), (384, 640)]  # (H, W)
BUNDLE = OUT / "bundle"
BUNDLED = ["yolo26n-pose_shapes", "yolo26n-seg_640", "osnet_x0_25_msmt17"]  # ModelSet.lite


def yolo_name(weights: str, h: int, w: int) -> str:
    return f"{Path(weights).stem}_{h}" if h == w else f"{Path(weights).stem}_{h}x{w}"


def _rename_outputs(path: Path, names: list[str]) -> None:
    import coremltools as ct

    m = ct.models.MLModel(str(path), skip_model_load=True)
    spec = m.get_spec()
    outs = [o.name for o in spec.description.output]
    if outs == names:
        return
    # seg: the [1,300,38] rows first, then protos; order by rank so the names land right
    ranked = sorted(outs, key=lambda n: len(next(o for o in spec.description.output if o.name == n).type.multiArrayType.shape))
    for old, new in zip(ranked, names):
        ct.utils.rename_feature(spec, old, new)
    m2 = ct.models.MLModel(spec, weights_dir=m.weights_dir, skip_model_load=True)
    m2.user_defined_metadata.update(dict(m.user_defined_metadata))
    tmp = path.with_name(path.stem + "_tmp.mlpackage")
    m2.save(str(tmp))
    shutil.rmtree(path)
    os.replace(tmp, path)


def export_yolo(weights: str, h: int, w: int, force: bool) -> Path:
    from ultralytics import YOLO

    target = OUT / f"{yolo_name(weights, h, w)}.mlpackage"
    if target.exists() and not force:
        import coremltools as ct

        meta = ct.models.MLModel(str(target), skip_model_load=True).user_defined_metadata
        if meta.get("end2end") == "False":
            _rename_outputs(target, ["detections"] if "pose" in weights else ["detections", "protos"])
            return target
    src = YOLO(str(data_root() / "models" / weights)).export(format="coreml", imgsz=[h, w], half=True)
    if target.exists():
        shutil.rmtree(target)
    os.replace(src, target)
    _rename_outputs(target, ["detections"] if "pose" in weights else ["detections", "protos"])
    return target


def merge_shapes(weights: str, force: bool) -> Path:
    """The pose model's input shapes as one multifunction model: functions s<H>x<W>, weights stored once."""
    from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

    target = OUT / f"{Path(weights).stem}_shapes.mlpackage"
    parts = [OUT / f"{yolo_name(weights, h, w)}.mlpackage" for h, w in POSE_SHAPES]
    if target.exists() and not force and all(target.stat().st_mtime >= p.stat().st_mtime for p in parts):
        return target
    desc = MultiFunctionDescriptor()
    for (h, w), part in zip(POSE_SHAPES, parts):
        desc.add_function(str(part), src_function_name="main", target_function_name=f"s{h}x{w}")
    desc.default_function_name = "s640x640"
    if target.exists():
        shutil.rmtree(target)
    save_multifunction(desc, str(target))
    return target


def export_reid(weights: str, force: bool) -> Path:
    import coremltools as ct
    import torch
    from boxmot.reid.core.runtime import ReID

    target = OUT / f"{Path(weights).stem}.mlpackage"
    if target.exists() and not force:
        m = ct.models.MLModel(str(target), skip_model_load=True)
        if m.get_spec().description.input[0].type.WhichOneof("Type") == "imageType":
            return target
    net = ReID(weights=data_root() / "models" / weights, device="cpu").model.model.eval()

    class Fixed(torch.nn.Module):
        """RGB 0..1 in; ImageNet normalisation, OSNet features, fixed-shape mean pool, fc."""

        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            f = self.m.featuremaps((x - self.mean) / self.std).mean(dim=(2, 3))
            return self.m.fc(f) if getattr(self.m, "fc", None) is not None else f

    x = torch.rand(1, 3, 256, 128)
    wrapped = Fixed(net).eval()
    with torch.no_grad():
        ref = net((x - wrapped.mean) / wrapped.std)
        got = wrapped(x)
    assert torch.allclose(ref, got, atol=1e-4), float((ref - got).abs().max())
    traced = torch.jit.trace(wrapped, x)
    ml = ct.convert(traced, inputs=[ct.ImageType(name="image", shape=x.shape, scale=1 / 255.0,
                                                 color_layout=ct.colorlayout.RGB)],
                    outputs=[ct.TensorType(name="embedding")], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS17)
    ml.user_defined_metadata.update({"input": "RGB 128x256, crop resized with cv2 INTER_LINEAR",
                                     "output": "embedding, L2-normalise", "weights": weights})
    if target.exists():
        shutil.rmtree(target)
    ml.save(str(target))
    return target


def compile_app(pkg: Path) -> Path:
    APP.mkdir(parents=True, exist_ok=True)
    out = APP / (pkg.stem + ".mlmodelc")
    if out.exists() and out.stat().st_mtime >= pkg.stat().st_mtime:
        return out
    if out.exists():
        shutil.rmtree(out)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(APP)], check=True, capture_output=True)
    return out


def _size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


# ---------------------------------------------------------------- accuracy check (FP16 vs PyTorch)

def _letterbox(img: np.ndarray, h: int, w: int) -> tuple[np.ndarray, float, int, int]:
    import cv2

    r = min(h / img.shape[0], w / img.shape[1])
    nw, nh = max(round(img.shape[1] * r), 1), max(round(img.shape[0] * r), 1)
    out = np.full((h, w, 3), 114, np.uint8)
    left, top = round((w - nw) / 2 - 0.1), round((h - nh) / 2 - 0.1)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, left, top


def check(frames: list[np.ndarray]) -> dict:
    """Core ML (FP16, Neural Engine) vs PyTorch (FP32) on the same letterboxed inputs."""
    import coremltools as ct
    import cv2
    import torch
    from PIL import Image
    from boxmot.reid.core.runtime import ReID
    from ultralytics import YOLO
    from ultralytics.utils import nms

    from directe_tracking.tracker import iou

    report = {}
    for weights in POSE + SEG:
        for h, w in (POSE_SHAPES if weights in POSE else [(640, 640)]):
            name = yolo_name(weights, h, w)
            ml = ct.models.MLModel(str(OUT / f"{name}.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
            yolo = YOLO(str(data_root() / "models" / weights))
            ious, kd, dconf, unmatched, n = [], [], [], 0, 0
            for f in frames:
                if (h > w) != (f.shape[0] > f.shape[1]) and h != w:
                    continue
                lb, *_ = _letterbox(f, h, w)
                ref = yolo.predict(lb, imgsz=[h, w], conf=0.2, classes=[0], device="mps", verbose=False)[0]
                rb = ref.boxes.xyxy.cpu().numpy()
                rc = ref.boxes.conf.cpu().numpy()
                raw = ml.predict({"image": Image.fromarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))})["detections"]
                out = nms.non_max_suppression(torch.from_numpy(raw), 0.2, 0.7, classes=[0],
                                              nc=len(yolo.names))[0].numpy()
                gb, gc = out[:, :4], out[:, 4]
                for b, c, i in zip(rb, rc, range(len(rb))):
                    n += 1
                    if not len(gb):
                        unmatched += 1
                        continue
                    ov = iou(b, gb)
                    j = int(np.argmax(ov))
                    if ov[j] < 0.5:
                        unmatched += 1
                        continue
                    ious.append(ov[j]); dconf.append(abs(c - gc[j]))
                    if weights in POSE:
                        rk = ref.keypoints.data[i].cpu().numpy()
                        gk = out[j, 6:].reshape(17, 3)
                        vis = rk[:, 2] > 0.35
                        if vis.any():
                            kd.append(float(np.linalg.norm(rk[vis, :2] - gk[vis, :2], axis=1).max()))
            report[name] = {"dets": n, "unmatched": unmatched,
                            "iou_min": round(float(np.min(ious)), 4) if ious else None,
                            "iou_median": round(float(np.median(ious)), 4) if ious else None,
                            "conf_absdiff_max": round(float(np.max(dconf)), 4) if dconf else None,
                            "kpt_px_max_median": round(float(np.median(kd)), 2) if kd else None}
            print(name, report[name], flush=True)
    rng = np.random.default_rng(0)
    for weights in REID:
        name = Path(weights).stem
        ml = ct.models.MLModel(str(OUT / f"{name}.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
        reid = ReID(weights=data_root() / "models" / weights, device="cpu")
        cos = []
        for f in frames:
            for _ in range(4):
                x0, y0 = int(rng.integers(0, f.shape[1] - 200)), int(rng.integers(0, f.shape[0] - 400))
                crop = f[y0:y0 + int(rng.integers(60, 400)), x0:x0 + int(rng.integers(30, 200))]
                ref = reid([crop])[0]
                rgb = cv2.cvtColor(cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                e = ml.predict({"image": Image.fromarray(rgb)})["embedding"].reshape(-1)
                e = e / np.linalg.norm(e)
                cos.append(float(ref @ e))
        report[name] = {"crops": len(cos), "cos_min": round(min(cos), 5), "cos_median": round(float(np.median(cos)), 5)}
        print(name, report[name], flush=True)
    del torch
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-export even if the .mlpackage exists")
    ap.add_argument("--check", action="store_true", help="compare FP16 Core ML with PyTorch on golden frames")
    args = ap.parse_args()
    pkgs = []
    for wts in POSE:
        pkgs += [export_yolo(wts, h, w, args.force) for h, w in POSE_SHAPES]
        pkgs.append(merge_shapes(wts, args.force))
    for wts in SEG:
        pkgs.append(export_yolo(wts, 640, 640, args.force))
    for wts in REID:
        pkgs.append(export_reid(wts, args.force))
    sizes = {}
    for p in pkgs:
        c = compile_app(p)
        sizes[p.stem] = {"mlpackage_mb": round(_size_mb(p), 1), "mlmodelc_mb": round(_size_mb(c), 1)}
        print(f"{p.stem:28s} {sizes[p.stem]['mlmodelc_mb']:6.1f} MB  {c}")
    (data_root() / "runs" / "coreml_app_models.json").write_text(json.dumps(sizes, indent=1))
    # the small set, for an app to bundle
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)
    for name in BUNDLED:
        shutil.copytree(APP / f"{name}.mlmodelc", BUNDLE / f"{name}.mlmodelc")
    print(f"bundle: {', '.join(BUNDLED)}, {_size_mb(BUNDLE):.1f} MB in {BUNDLE}")
    if args.check:
        import cv2

        golden = data_root() / "ios_golden" / "perception"
        frames = [cv2.imread(str(p)) for p in sorted(golden.glob("*.png"))]
        if not frames:
            raise SystemExit(f"no frames in {golden}; put golden frames (PNG, BGR as cv2 writes them) there first")
        (data_root() / "runs" / "coreml_fp16_check.json").write_text(json.dumps(check(frames), indent=1))


if __name__ == "__main__":
    main()
