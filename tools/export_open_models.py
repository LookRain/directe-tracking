"""Core ML models for Perception from openly licensed weights (Apache-2.0, trained on COCO only), in the same input and
output layouts as the YOLO26 models tools/export_coreml.py makes, so Perception runs either set unchanged.

  $VENV/bin/python tools/export_open_models.py [--force] [--check]

$VENV: Python 3.11 with torch 2.5, mmengine 0.10.7, mmcv-lite 2.1 (its compiled ops replaced by a pure-PyTorch
nms/batched_nms: nothing else here needs them), mmdet 3.3, mmpose 1.3.2, coremltools 8.3. Weights (OpenMMLab's
download server) in $DIRECTE_DATA/models/external/safe:

  rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth, rtmo-m_16xb16-600e_coco-640x640-6f4e0306_20231211.pth (mmpose)
  rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth (mmdet RTMDet-Ins: the outline model; CSPNeXt backbone
  ImageNet-pretrained)

Writes $DIRECTE_DATA/models/coreml/open/<name>.mlpackage, compiled into models/coreml/app (next to the YOLO26 ones; the
names differ) and models/coreml/bundle-open-q8 (ModelSet.lite, what an app bundles):

  rtmo-m_shapes, rtmo-s_shapes   pose, functions s640x640 / s640x384 / s384x640 (weights stored once); rtmo-s_shapes
                                 and seg640, the model's box finder (ModelSet's "rtmo-m:seg640", "rtmo-s:seg640")
      input  "image": BGR 0...255 (RTMO's own input), letterboxed grey 114
      output "detections" [1, 56, N]: cx cy w h (input pixels), person score, 17 x (x, y, visibility)
  <rtmo>:seg640 (function of <rtmo>_shapes)   the mask model's slot for second chance
      output "detections" [1, 4 + 1 + 32, N] (mask coefficients 1, 0, ...), "protos" [1, 32, 160, 160] (plane 0 ones):
      each person's mask is their box. Second chance uses the mask model only to find someone.
  rtmdet-ins-m_640               the outline model (focus-window masks: Precise, finding far climbers)
      output "detections" [1, 4 + 1 + 48, 48]: the 48 best anchors, each coefficient one-hot, and "protos"
      [1, 48, 80, 80]: those anchors' own mask logits, so coefficients . prototypes is each mask exactly
Every model carries metadata nms_iou = 0.6 (Perception's nmsIoU; YOLO26 has none and keeps 0.7).

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

DATA = Path(os.environ.get("DIRECTE_DATA", "data"))  # as directe_tracking/clips.py
W = DATA / "models" / "external" / "safe"
OUT = DATA / "models" / "coreml" / "open"
APP = DATA / "models" / "coreml" / "app"
BUNDLE = DATA / "models" / "coreml" / "bundle-open-q8"
SHAPES = [(640, 640), (640, 384), (384, 640)]
# NMS overlap for every model here (YOLO26: 0.7). RTMO's neighbouring anchors fire together: at 0.7 one person came out
# as two boxes in 31% of looks on a synthetic scene (YOLO26 under 1%), which the tracker reads as someone beside the
# climber. At 0.6: Basic's live segments 74.8% -> 85.4% right, wrong person 9.3% -> 2.0%; Full unchanged.
NMS_IOU = "0.6"
POSE = {  # name: (mmpose config, checkpoint, box scale w, h)
    "rtmo-m": ("rtmo/coco/rtmo-m_16xb16-600e_coco-640x640.py", "rtmo-m_16xb16-600e_coco-640x640-6f4e0306_20231211.pth", (0.933, 0.965)),
    "rtmo-s": ("rtmo/coco/rtmo-s_8xb32-600e_coco-640x640.py", "rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth", (1.0, 1.0)),
}
DET = {"rtmdet-ins-m": ("rtmdet/rtmdet-ins_m_8xb32-300e_coco.py", "rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth")}
# ModelSet.lite (Sources/Perception/Models.swift) and the multifunction OSNet x0.25 the YOLO26 export bundles
# One-function models only: an iPhone (iOS 27.2) wouldn't load the 8-bit multifunction rtmo-m-q8_shapes for the Neural
# Engine ("`.functionName` must be nil unless the model type is ML Program"; fine on its CPU and on a Mac, build 112), so
# RTMO-m comes at 640 x 640 only and the outline model is second chance's mask model too.
BUNDLED = ["rtmo-m-q8_640", "rtmdet-ins-m-q8_640", "osnet_ain_x1_0_msmt17", "osnet_ibn_x1_0_msdc"]
# the 8-bit copies (ModelSet.lite): weights per-channel int8 (coremltools linear_quantize_weights), outputs within 0.4%
# of the 16-bit ones; the OSNets come from tools/export_coreml.py's packages
OSNETS = ["osnet_ain_x1_0_msmt17", "osnet_ibn_x1_0_msdc"]


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


class RTMDetInsSeg(nn.Module):
    """RTMDet-Ins (person class) in YOLO26-seg's layout, real outlines: the K best anchors' masks become K prototype
    planes and each detection's coefficients pick its own (one-hot), so YOLO26-seg's coefficients . prototypes gives
    each mask exactly. The dynamic convs run per instance as batched matmuls; prototypes at the mask head's stride 8.
    detections [1, 4 + 1 + K, K], protos [1, K, H/8, W/8]."""

    def __init__(self, model, hw=(640, 640), k=48):
        super().__init__()
        self.backbone, self.neck, self.head = neural_engine_friendly(model.backbone), model.neck, model.bbox_head
        self.k = k
        pp = model.data_preprocessor
        self.register_buffer("mean", pp.mean.view(1, 3, 1, 1).clone())
        self.register_buffer("std", pp.std.view(1, 3, 1, 1).clone())
        strides = [s[0] for s in self.head.prior_generator.strides]
        pri = self.head.prior_generator.grid_priors([torch.Size((hw[0] // s, hw[1] // s)) for s in strides], dtype=torch.float32,
                                                    device="cpu", with_stride=True)
        self.register_buffer("priors", torch.cat(pri))                # [N, 4]: x, y, stride x, stride y
        mh, mw = hw[0] // strides[0], hw[1] // strides[0]
        coord = self.head.prior_generator.single_level_grid_priors((mh, mw), level_idx=0, device="cpu")
        self.register_buffer("coord", coord.reshape(1, -1, 2))         # [1, HW, 2]
        self.mh, self.mw = mh, mw

    def forward(self, image):
        h = self.head
        cls, dist, kern, mask_feat = h(self.neck(self.backbone((image - self.mean) / self.std)))
        score = flat(cls)[0, :, 0].sigmoid()
        top, idx = score.topk(self.k)
        d = flat(dist)[0][idx]
        pri = self.priors[idx]
        x0y0, x1y1 = pri[:, :2] - d[:, :2], pri[:, :2] + d[:, 2:]
        kernels = flat(kern)[0][idx]                                  # [K, P]
        # relative coordinates, as RTMDetInsHead._mask_predict_by_feat_single
        rel = (pri[:, None, :2] - self.coord) / (pri[:, None, 2:3] * 8)  # [K, HW, 2]
        feat = mask_feat.reshape(1, mask_feat.shape[1], -1).expand(self.k, -1, -1)  # [K, C, HW]
        x = torch.cat([rel.permute(0, 2, 1), feat], 1)               # [K, C + 2, HW]
        o = 0
        ws = h.weight_nums
        bs = h.bias_nums
        n = len(ws)
        ch = h.dyconv_channels
        params_w, params_b = [], []
        for i in range(n):
            params_w.append(kernels[:, o:o + ws[i]]); o += ws[i]
        for i in range(n):
            params_b.append(kernels[:, o:o + bs[i]]); o += bs[i]
        for i in range(n):
            out = ch if i < n - 1 else 1
            w = params_w[i].reshape(self.k, out, -1)                   # [K, out, in]
            x = torch.bmm(w, x) + params_b[i].reshape(self.k, out, 1)
            if i < n - 1:
                x = torch.relu(x)
        protos = x.reshape(1, self.k, self.mh, self.mw)                # mask logits per instance
        coef = torch.eye(self.k, dtype=top.dtype)                      # [K coefficient rows, K detections]
        det = torch.cat([((x0y0 + x1y1) / 2).t(), (x1y1 - x0y0).t(), top[None], coef], 0)[None]
        return det, protos


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
    m.user_defined_metadata["nms_iou"] = NMS_IOU  # Perception reads it (nmsIoU)
    if path.exists():
        shutil.rmtree(path)
    m.save(str(path))
    return path


def merge_shapes(name: str, extra: dict[str, Path] | None = None) -> Path:
    """The pose shapes as functions s<H>x<W> of one model, plus `extra` functions (its box finder, seg640)."""
    from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction
    target = OUT / f"{name}_shapes.mlpackage"
    desc = MultiFunctionDescriptor()
    for h, w in SHAPES:
        desc.add_function(str(OUT / f"{name}_{h}x{w}.mlpackage"), src_function_name="main", target_function_name=f"s{h}x{w}")
    for fn, pkg in (extra or {}).items():
        desc.add_function(str(pkg), src_function_name="main", target_function_name=fn)
    desc.default_function_name = "s640x640"
    if target.exists():
        shutil.rmtree(target)
    save_multifunction(desc, str(target))
    # save_multifunction drops the parts' metadata: write it into the package's spec (the weights stay where they are)
    import json
    info = json.loads((target / "Manifest.json").read_text())
    spec_path = next(target / "Data" / e["path"] for e in info["itemInfoEntries"].values() if e["path"].endswith(".mlmodel"))
    import coremltools as ct
    spec = ct.utils.load_spec(str(target))
    spec.description.metadata.userDefined["nms_iou"] = NMS_IOU
    spec_path.write_bytes(spec.SerializeToString())
    assert ct.utils.load_spec(str(target)).description.metadata.userDefined["nms_iou"] == NMS_IOU
    return target


def int8(src: Path, dst: Path) -> Path:
    import coremltools as ct
    from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights
    cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                                                     granularity="per_channel", weight_threshold=2048))
    if dst.exists():
        shutil.rmtree(dst)
    linear_quantize_weights(ct.models.MLModel(str(src), skip_model_load=True), config=cfg).save(str(dst))
    return dst


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
    segs = {}
    for name, (_, _, bs) in POSE.items():  # each RTMO's box finder, a function of its _shapes model (no extra weights)
        segs[name] = OUT / f"{name}-boxseg_640.mlpackage"
        if a.force or not segs[name].exists():
            convert(RTMOBoxSeg(RTMOPose(load_pose(name), (640, 640), bs)), (640, 640), ["detections", "protos"], segs[name],
                    f"{name} (mmpose RTMO, Apache-2.0, COCO) in YOLO26-seg layout, box-shaped masks")
    for name, (_, _, bs) in POSE.items():
        target = OUT / f"{name}_shapes.mlpackage"
        if a.force or not target.exists():
            model = load_pose(name)
            for hw in SHAPES:
                convert(RTMOPose(copy.deepcopy(model), hw, bs), hw, ["detections"], OUT / f"{name}_{hw[0]}x{hw[1]}.mlpackage",
                        f"{name} (mmpose RTMO, Apache-2.0, COCO) in YOLO26-pose layout, {hw[0]}x{hw[1]}")
            merge_shapes(name, {"seg640": segs[name]})
        pkgs.append(target)
    # the built-in set: rtmo-m and the OSNets at 8 bits
    q8 = OUT / "rtmo-m-q8_shapes.mlpackage"
    if a.force or not q8.exists():
        for h, w in SHAPES:
            int8(OUT / f"rtmo-m_{h}x{w}.mlpackage", OUT / f"rtmo-m-q8_{h}x{w}.mlpackage")
        int8(segs["rtmo-m"], OUT / "rtmo-m-q8-boxseg_640.mlpackage")
        merge_shapes("rtmo-m-q8", {"seg640": OUT / "rtmo-m-q8-boxseg_640.mlpackage"})
    pkgs.append(q8)
    q8one = OUT / "rtmo-m-q8_640.mlpackage"  # the bundled one: 640 x 640 alone (BUNDLED)
    if q8one.exists():
        shutil.rmtree(q8one)
    shutil.copytree(OUT / "rtmo-m-q8_640x640.mlpackage", q8one)
    pkgs.append(q8one)
    for name in OSNETS:
        dst = OUT / f"{name}-q8.mlpackage"
        if a.force or not dst.exists():
            int8(DATA / "models" / "coreml" / f"{name}.mlpackage", dst)
        pkgs.append(dst)
    ins = OUT / "rtmdet-ins-m_640.mlpackage"
    if a.force or not ins.exists():
        convert(RTMDetInsSeg(load_det("rtmdet-ins-m")), (640, 640), ["detections", "protos"], ins,
                "rtmdet-ins-m (mmdet RTMDet-Ins, Apache-2.0, COCO) in YOLO26-seg layout: the 48 best anchors' masks as prototypes")
    pkgs.append(ins)
    insq = OUT / "rtmdet-ins-m-q8_640.mlpackage"
    if a.force or not insq.exists():
        int8(OUT / "rtmdet-ins-m_640.mlpackage", insq)
    pkgs.append(insq)
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
