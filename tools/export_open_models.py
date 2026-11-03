"""Core ML models for Perception from openly licensed weights (Apache-2.0, trained on COCO only), in the same input and
output layouts as the YOLO26 models tools/export_coreml.py makes, so Perception runs either set unchanged.

  $VENV/bin/python tools/export_open_models.py [--force] [--check]

$VENV: Python 3.11 with torch 2.5, mmengine 0.10.7, mmcv-lite 2.1 (its compiled ops replaced by a pure-PyTorch
nms/batched_nms: nothing else here needs them), mmdet 3.3, mmpose 1.3.2, coremltools 8.3. Weights (OpenMMLab's
download server) in $DIRECTE_DATA/models/external/safe:

  rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth, rtmo-m_16xb16-600e_coco-640x640-6f4e0306_20231211.pth (mmpose)
  rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth (mmdet; CSPNeXt backbone ImageNet-pretrained)

Writes $DIRECTE_DATA/models/coreml/open/<name>.mlpackage, compiled into models/coreml/app (next to the YOLO26 ones; the
names differ) and models/coreml/bundle-open (ModelSet.lite, what an app bundles):

  rtmo-m_shapes, rtmo-s_shapes   pose, functions s640x640 / s640x384 / s384x640 (weights stored once)
      input  "image": BGR 0...255 (RTMO's own input), letterboxed grey 114
      output "detections" [1, 56, N]: cx cy w h (input pixels), person score, 17 x (x, y, visibility)
  rtmo-s-boxseg_640, rtmdet-m-boxseg_640   the mask model's slot (second chance; masks for the small set)
      output "detections" [1, 4 + 1 + 32, N] (mask coefficients 1, 0, ...), "protos" [1, 32, 160, 160] (plane 0 ones):
      each person's mask is their box. Second chance uses the mask model only to find someone.

What differs from mmpose / mmdet's own pipelines, all with the same outputs where it matters:
  - YOLOX's Focus stem (four strided slices: Core ML runs them on the CPU at full resolution) is the equivalent 6x6
    stride-2 conv, and the SPP's 9 and 13 max pools are cascaded 5x5 ones (SPPF): every op on the Neural Engine
    (rtmo-m: 6.9 ms at 640 x 640 on an M4, vs 45 ms as converted).
  - RTMO's keypoints come from its keypoint-offset branch (trained for its proxy poses; mmpose decodes keypoints with
    the DCC head per person after NMS): every anchor in one pass, the layout above, NMS in Swift. On the
    detector-size moments they agree with yolo26m-pose's as well as DCC's do (OKS 0.674 vs 0.677).
  - RTMO-m's boxes are looser than YOLO26-m's (15% wider, 7% taller on the same people) and the tracker and Smart
    Zoom were tuned on YOLO26 boxes: half of that is taken off about the centre (BOX_SCALE). The whole of it made
    the crops too tight on a sprawled, falling climber (re-ID and colour lost them on a real lead fall).
Bench: docs/review/deep-dives/safe-models.md in the app repo.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import shutil
import subprocess
from pathlib import Path

import torch
from torch import nn

DATA = Path(os.environ.get("DIRECTE_DATA", "/Volumes/ROCKET-XTRM/Offload/climb-cam"))
W = DATA / "models" / "external" / "safe"
OUT = DATA / "models" / "coreml" / "open"
APP = DATA / "models" / "coreml" / "app"
BUNDLE = DATA / "models" / "coreml" / "bundle-open"
SHAPES = [(640, 640), (640, 384), (384, 640)]
POSE = {  # name: (mmpose config, checkpoint, box scale w, h)
    "rtmo-m": ("rtmo/coco/rtmo-m_16xb16-600e_coco-640x640.py", "rtmo-m_16xb16-600e_coco-640x640-6f4e0306_20231211.pth", (0.933, 0.965)),
    "rtmo-s": ("rtmo/coco/rtmo-s_8xb32-600e_coco-640x640.py", "rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth", (1.0, 1.0)),
}
DET = {"rtmdet-m": ("rtmdet/rtmdet_m_8xb32-300e_coco.py", "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth")}
# ModelSet.lite (Sources/Perception/Models.swift) and the multifunction OSNet x0.25 the YOLO26 export bundles
BUNDLED = ["rtmo-s_shapes", "rtmo-s-boxseg_640", "osnet_x0_25_msmt17"]


def configs(pkg: str) -> Path:
    return Path(importlib.import_module(pkg).__file__).parent / ".mim" / "configs"


class FocusConv(nn.Module):
    """Focus (space-to-depth, then a k x k conv) as one 2k x 2k stride-2 conv with the same output."""

    def __init__(self, focus):
        super().__init__()
        c = focus.conv.conv
        w = c.weight.data                                     # [O, 12, k, k]; input groups (dy, dx) = 00, 10, 01, 11
        o, _, k, _ = w.shape
        big = torch.zeros(o, 3, 2 * k, 2 * k)
        for g, (dy, dx) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
            big[:, :, dy::2, dx::2] = w[:, 3 * g:3 * g + 3]
        self.conv = nn.Conv2d(3, o, 2 * k, stride=2, padding=2 * (k // 2), bias=c.bias is not None)
        self.conv.weight.data = big
        if c.bias is not None:
            self.conv.bias.data = c.bias.data
        self.norm, self.act = focus.conv.bn, focus.conv.activate

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SPPF(nn.Module):
    """SPPBottleneck with pools 5/9/13 as three cascaded 5x5 pools (the same maxima)."""

    def __init__(self, spp):
        super().__init__()
        assert [p.kernel_size for p in spp.poolings] == [5, 9, 13]
        self.conv1, self.conv2 = spp.conv1, spp.conv2
        self.pool = nn.MaxPool2d(5, 1, 2)

    def forward(self, x):
        x = self.conv1(x)
        a = self.pool(x)
        b = self.pool(a)
        return self.conv2(torch.cat([x, a, b, self.pool(b)], 1))


def neural_engine_friendly(backbone):
    kinds = []
    for mod in ("mmpose.models.backbones.csp_darknet", "mmdet.models.backbones.csp_darknet"):
        try:
            m = importlib.import_module(mod)
            kinds.append((m.Focus, FocusConv))
            kinds.append((m.SPPBottleneck, SPPF))
        except ImportError:
            pass
    done: list[str] = []
    for name, m in list(backbone.named_modules()):
        if not name or any(name.startswith(d + ".") for d in done):
            continue
        for cls, repl in kinds:
            if isinstance(m, cls):
                parent = backbone.get_submodule(name.rsplit(".", 1)[0]) if "." in name else backbone
                setattr(parent, name.rsplit(".", 1)[-1], repl(m))
                done.append(name)
                break
    return backbone


def flat(xs):
    return torch.cat([x.permute(0, 2, 3, 1).flatten(1, 2) for x in xs], 1)


class RTMOPose(nn.Module):
    """[1, 56, N] over every anchor: boxes, score, keypoints from the offset branch."""

    def __init__(self, model, hw, box_scale):
        super().__init__()
        self.backbone, self.neck, head = neural_engine_friendly(model.backbone), model.neck, model.head
        self.head = head.head_module
        sizes = [(hw[0] // s, hw[1] // s) for s in head.featmap_strides]
        pri = head.prior_generator.grid_priors([torch.Size(s) for s in sizes], dtype=torch.float32, device="cpu")
        self.register_buffer("priors", torch.cat(pri))
        self.register_buffer("strides", torch.cat([torch.full((h * w,), float(s)) for (h, w), s in zip(sizes, head.featmap_strides)]))
        self.register_buffer("box_scale", torch.tensor(box_scale, dtype=torch.float32).view(1, 2))

    def forward(self, image):
        x, h = self.neck(self.backbone(image)), self.head
        cls, bbox, kreg, vis = [], [], [], []
        for i in range(len(x)):
            cf, rf = x[i].split(x[i].size(1) // 2, 1)
            cf, rf = h.conv_cls[i](cf), h.conv_pose[i](rf)
            cls.append(h.out_cls[i](cf)); bbox.append(h.out_bbox[i](rf))
            kreg.append(h.out_kpt_reg[i](rf)); vis.append(h.out_kpt_vis[i](rf))
        st = self.strides[:, None]
        b = flat(bbox)[0]
        xy = b[:, :2] * st + self.priors
        wh = b[:, 2:].exp() * st * self.box_scale
        kp = flat(kreg)[0].reshape(-1, 17, 2) * st[:, :, None] + self.priors[:, None, :]
        kv = flat(vis)[0].sigmoid()
        det = torch.cat([xy, wh, flat(cls)[0].sigmoid(), torch.cat([kp, kv[..., None]], 2).reshape(-1, 51)], 1)
        return det.t()[None]


def box_masks(image, det5):
    """YOLO26-seg's outputs with each mask its box: coefficients (1, 0, ...) over a first prototype plane of ones."""
    coef = torch.cat([torch.ones_like(det5[:, :1]), torch.zeros_like(det5[:, :1]).repeat(1, 31, 1)], 1)
    protos = (nn.functional.avg_pool2d(image[:, :1], 4) * 0 + 1).repeat(1, 32, 1, 1)
    return torch.cat([det5, coef], 1), protos


class RTMOBoxSeg(nn.Module):
    def __init__(self, pose: RTMOPose):
        super().__init__()
        self.pose = pose

    def forward(self, image):
        return box_masks(image, self.pose(image)[:, :5])


class RTMDetBoxSeg(nn.Module):
    """RTMDet's person class, BGR 0...255 in (its mean/std inside)."""

    def __init__(self, model, hw=(640, 640)):
        super().__init__()
        self.backbone, self.neck, self.head = neural_engine_friendly(model.backbone), model.neck, model.bbox_head
        pp = model.data_preprocessor
        self.register_buffer("mean", pp.mean.view(1, 3, 1, 1).clone())
        self.register_buffer("std", pp.std.view(1, 3, 1, 1).clone())
        strides = [s[0] for s in self.head.prior_generator.strides]
        pri = self.head.prior_generator.grid_priors([torch.Size((hw[0] // s, hw[1] // s)) for s in strides], dtype=torch.float32, device="cpu")
        self.register_buffer("priors", torch.cat([p[:, :2] for p in pri]))

    def forward(self, image):
        cls, dist = self.head(self.neck(self.backbone((image - self.mean) / self.std)))
        score = flat(cls)[0, :, :1].sigmoid()
        d = flat(dist)[0]                                     # l t r b, already times the stride
        x0y0, x1y1 = self.priors - d[:, :2], self.priors + d[:, 2:]
        return box_masks(image, torch.cat([(x0y0 + x1y1) / 2, x1y1 - x0y0, score], 1).t()[None])


def load_pose(name):
    from mmengine.config import Config
    from mmpose.apis import init_model
    cfg, ckpt, _ = POSE[name]
    return init_model(Config.fromfile(str(configs("mmpose") / "body_2d_keypoint" / cfg)), str(W / ckpt), device="cpu").eval()


def load_det(name):
    from mmdet.apis import init_detector
    from mmengine.config import Config
    cfg, ckpt = DET[name]
    return init_detector(Config.fromfile(str(configs("mmdet") / cfg)), str(W / ckpt), device="cpu").eval()


def convert(net: nn.Module, hw, outputs: list[str], path: Path, note: str) -> Path:
    import coremltools as ct
    ex = torch.rand(1, 3, *hw) * 255
    with torch.no_grad():
        traced = torch.jit.trace(net.eval(), ex)
    m = ct.convert(traced, inputs=[ct.ImageType(name="image", shape=ex.shape, scale=1.0, color_layout=ct.colorlayout.BGR)],
                   outputs=[ct.TensorType(name=o) for o in outputs], minimum_deployment_target=ct.target.iOS17,
                   compute_precision=ct.precision.FLOAT16, convert_to="mlprogram")
    m.short_description = note
    if path.exists():
        shutil.rmtree(path)
    m.save(str(path))
    return path


def merge_shapes(name: str) -> Path:
    from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction
    target = OUT / f"{name}_shapes.mlpackage"
    desc = MultiFunctionDescriptor()
    for h, w in SHAPES:
        desc.add_function(str(OUT / f"{name}_{h}x{w}.mlpackage"), src_function_name="main", target_function_name=f"s{h}x{w}")
    desc.default_function_name = "s640x640"
    if target.exists():
        shutil.rmtree(target)
    save_multifunction(desc, str(target))
    return target


def compile_app(pkg: Path) -> Path:
    APP.mkdir(parents=True, exist_ok=True)
    out = APP / (pkg.stem + ".mlmodelc")
    if out.exists():
        shutil.rmtree(out)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(APP)], check=True, capture_output=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pkgs = []
    for name, (_, _, bs) in POSE.items():
        target = OUT / f"{name}_shapes.mlpackage"
        if a.force or not target.exists():
            model = load_pose(name)
            for hw in SHAPES:
                convert(RTMOPose(copy.deepcopy(model), hw, bs), hw, ["detections"], OUT / f"{name}_{hw[0]}x{hw[1]}.mlpackage",
                        f"{name} (mmpose RTMO, Apache-2.0, COCO) in YOLO26-pose layout, {hw[0]}x{hw[1]}")
            merge_shapes(name)
        pkgs.append(target)
    seg = OUT / "rtmo-s-boxseg_640.mlpackage"
    if a.force or not seg.exists():
        convert(RTMOBoxSeg(RTMOPose(load_pose("rtmo-s"), (640, 640), POSE["rtmo-s"][2])), (640, 640), ["detections", "protos"], seg,
                "rtmo-s (mmpose RTMO, Apache-2.0, COCO) in YOLO26-seg layout, box-shaped masks")
    pkgs.append(seg)
    det = OUT / "rtmdet-m-boxseg_640.mlpackage"
    if a.force or not det.exists():
        convert(RTMDetBoxSeg(load_det("rtmdet-m")), (640, 640), ["detections", "protos"], det,
                "rtmdet-m (mmdet RTMDet, Apache-2.0, COCO) in YOLO26-seg layout, box-shaped masks, person only")
    pkgs.append(det)
    for p in pkgs:
        c = compile_app(p)
        print(f"{p.stem:24s} {sum(f.stat().st_size for f in c.rglob('*') if f.is_file()) / 1e6:6.1f} MB  {c}")
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)
    for name in BUNDLED:
        shutil.copytree(APP / f"{name}.mlmodelc", BUNDLE / f"{name}.mlmodelc")
    print(f"bundle: {', '.join(BUNDLED)} in {BUNDLE}")


if __name__ == "__main__":
    main()
