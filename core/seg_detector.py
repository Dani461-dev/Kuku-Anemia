"""YOLO26-seg nail instance segmentation -> nail/skin boxes + per-pixel mask.

Tracks nails (each finger = one instance, mask + bounding box) with a trained
YOLO26n-seg model. The pixel-precise mask feeds straight into the feature
pipeline (core/pipeline.patient_features), replacing heuristic k-means/Otsu.

Skin box derivation (no SKIN class in the seg model):
    layout  - equal to nail box shifted by off_x * nail_width along +x,
              matching the MSU dataset layout statistics (skin_x-nail_x
              ~ 2.3x nail width, same y range). Consistent with how the
              Hb feature model was trained.
    mask_pca - proximal skin along the nail major axis (orientation aware,
              expensive; distal/proximal ambiguity remains).

Weights default: experiments/yolo26_seg/runs/seg26/weights/best.pt
(override via env ANEVIA_SEG26_WEIGHTS).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core import config as cfg
from core.detectors import DetectedFinger
from core.util import clip_box


@dataclass
class NailInstance:
    """One segmented nail: box [t,l,b,r] pixels, full-frame binary mask, score."""

    nail_box: list[int]
    mask: np.ndarray            # uint8, same HxW as the input image, 1=nail
    confidence: float
    finger_id: int = 0

    def skin_box(self, img_h: int, img_w: int, off_x: float = 2.3, method: str = "layout") -> list[int]:
        """Skin box derived from the nail mask region."""
        t, l, b, r = self.nail_box
        h, w = b - t, r - l
        if method == "layout":
            shift = int(round(off_x * w))
            return clip_box([t, l + shift, b, r + shift], img_h, img_w)
        # mask_pca: place skin at the proximal (wider) end along the major axis
        ys, xs = np.where(self.mask > 0)
        if len(ys) == 0:
            return clip_box([t, l, b, r], img_h, img_w)
        # covariance of nail pixels -> major axis = finger direction
        cov = np.cov(np.stack([xs, ys]))
        eigvals, eigvecs = np.linalg.eigh(cov)
        ax = eigvecs[:, np.argmax(eigvals)]           # unit major axis (x,y)
        proj = xs * ax[0] + ys * ax[1]
        direction = 1.0 if np.median(proj) < 0 else -1.0  # pick the hand side
        shift = direction * ax * 1.2 * h
        return clip_box([int(t + shift[1] - h / 2), int(l + shift[0] - w / 2),
                         int(b + shift[1] - h / 2), int(r + shift[0] - w / 2)], img_h, img_w)

    def to_detected_finger(self, img_h: int, img_w: int, off_x: float = 2.3,
                           method: str = "layout") -> DetectedFinger:
        skin = self.skin_box(img_h, img_w, off_x=off_x, method=method)
        return DetectedFinger(finger_id=self.finger_id, nail_box=list(self.nail_box),
                              skin_box=skin, confidence=self.confidence, source="seg26")


@dataclass
class SegResult:
    """Output of one image: instances + convenience selectors."""

    instances: list[NailInstance] = field(default_factory=list)
    img_h: int = 0
    img_w: int = 0

    def __bool__(self) -> bool:
        return len(self.instances) > 0

    def middle_finger(self, method: str = "layout") -> DetectedFinger | None:
        """Pick the instance whose nail-box centre is the vertical (y) median."""
        if not self.instances:
            return None
        ordered = sorted(self.instances,
                         key=lambda i: (i.nail_box[0] + i.nail_box[2]) / 2)
        mid = ordered[len(ordered) // 2]
        return mid.to_detected_finger(self.img_h, self.img_w, method=method)

    def as_detected_fingers(self, method: str = "layout") -> list[DetectedFinger]:
        return [i.to_detected_finger(self.img_h, self.img_w, method=method)
                for i in self.instances]


def _default_weights() -> Path:
    env = os.environ.get("ANEVIA_SEG26_WEIGHTS")
    if env:
        return Path(env)
    return cfg.PROJECT_ROOT / "experiments" / "yolo26_seg" / "runs" / "seg26" / "weights" / "best.pt"


class Yolo26SegDetector:
    """Instance-segmentation nail detector backed by YOLO26-seg."""

    name = "seg26"

    def __init__(self, weights: str | Path | None = None, conf: float = 0.30,
                 iou: float = 0.5, device: str = "cpu",
                 skin_off_x: float = 2.3, skin_method: str = "layout",
                 imgsz: int | None = 640, fallback: bool = False,
                 fallback_conf: float = 0.10, fallback_imgsz: int = 1280,
                 min_nails: int = 3, max_nails: int = 5,
                 min_fallback_conf: float = 0.25):
        from ultralytics import YOLO

        weights = Path(weights) if weights else _default_weights()
        if not weights.exists():
            raise FileNotFoundError(
                f"seg26 weights not found: {weights}. Train first via "
                "experiments/yolo26_seg/train_seg26.py."
            )
        self.conf = conf
        self.iou = iou
        self.skin_off_x = skin_off_x
        self.skin_method = skin_method
        self.imgsz = imgsz or 640          # pass utama = skala training (640)
        # dua-pass (aktif di runtime app via run_seg_pipeline):
        # pass utama conf 0.30@640 -> jika <3 kuku, ulang conf 0.10@1280
        # + dedupe + min-conf 0.25 + cap 5. Offline/eval: fallback=False
        # supaya perilaku threshold eksplisit tetap reproducible.
        self.fallback = fallback
        self.fallback_conf = fallback_conf
        self.fallback_imgsz = fallback_imgsz
        self.min_nails = min_nails
        self.max_nails = max_nails
        self.min_fallback_conf = min_fallback_conf
        self._model = YOLO(str(weights))
        self._device = device

    @staticmethod
    def _box_iou(a: list[int], b: list[int]) -> float:
        """IoU dua box format [t, l, b, r]."""
        it = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - it)
        return it / ua if ua > 0 else 0.0

    @staticmethod
    def _dedupe(instances: list, min_conf: float = 0.0) -> list:
        """Simpan deteksi terkuat; buang duplikat (IoU>0.5) & conf di bawah min_conf.

        Pada conf rendah (fallback) mask yang sama bisa terbelah jadi 2-3 instans
        (conf ~0.13-0.18) -> inilah sumber 'kuku 6 padahal 5'.
        """
        keep: list = []
        for inst in sorted(instances, key=lambda i: i.confidence, reverse=True):
            if inst.confidence < min_conf:
                continue
            if any(Yolo26SegDetector._box_iou(inst.nail_box, k.nail_box) > 0.5
                   for k in keep):
                continue
            keep.append(inst)
        return keep

    def _predict(self, img_rgb: np.ndarray, conf: float, imgsz: int):
        results = self._model.predict(
            img_rgb, conf=conf, iou=self.iou, verbose=False,
            device=self._device, retina_masks=True, imgsz=imgsz,
        )
        res = results[0]
        if res.masks is None:
            return []
        h, w = img_rgb.shape[:2]
        masks = res.masks.data
        if masks is None:
            return []
        masks = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
        boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else np.asarray(res.boxes.xyxy)
        confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else np.asarray(res.boxes.conf)
        out = []
        for i in range(masks.shape[0]):
            m = masks[i]
            if m.shape[:2] != (h, w):      # safety: resize to frame if needed
                import cv2
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            m_bin = (m > 0).astype(np.uint8)
            out.append(NailInstance(
                nail_box=[y1, x1, y2, x2], mask=m_bin,
                confidence=float(confs[i]), finger_id=i))
        return out

    def detect(self, img_rgb: np.ndarray) -> SegResult:
        """Returns nail instances (box + full-frame mask) for one RGB image."""
        h, w = img_rgb.shape[:2]
        insts = self._dedupe(self._predict(img_rgb, self.conf, self.imgsz))
        # fallback: foto susah (resolusi rendah / framing global) sering gagal
        # di pass utama -> ulangi dengan conf rendah + input lebih besar.
        if (self.fallback and len(insts) < self.min_nails
                and (self.fallback_conf != self.conf
                     or self.fallback_imgsz != self.imgsz)):
            alt = self._dedupe(
                self._predict(img_rgb, self.fallback_conf, self.fallback_imgsz),
                min_conf=self.min_fallback_conf)
            if len(alt) > len(insts):
                insts = alt
        insts = insts[: self.max_nails]
        return SegResult(instances=insts, img_h=h, img_w=w)

    def detect_boxes(self, img_rgb: np.ndarray) -> list[DetectedFinger]:
        """Compatibility with the NailSkinDetector contract (boxes only)."""
        result = self.detect(img_rgb)
        return result.as_detected_fingers(method=self.skin_method)

    def segment(self, img_rgb: np.ndarray) -> SegResult:
        """Alias of detect(); richer API naming for the runtime pipeline."""
        return self.detect(img_rgb)


def nail_skin_masks(inst: NailInstance, img_h: int, img_w: int,
                    off_x: float = 2.3) -> tuple[np.ndarray, np.ndarray]:
    """Return (full-frame nail mask, full-frame skin mask).

    Skin mask = the geometry-derived skin box (layout: shifted right by
    off_x*nail_width, same y-range as nail — matches MSU training layout),
    with the nail mask excluded. This mirrors exactly how the Hb feature model
    was trained (SKIN percentiles over the skin box), so seg masks feed the
    same feature distribution without re-deriving a different skin region.
    Both uint8 0/1.
    """
    mask = inst.mask
    t, l, b, r = inst.nail_box
    st, sl, sb, sr = inst.skin_box(img_h, img_w, off_x=off_x, method="layout")
    skin = np.zeros((img_h, img_w), np.uint8)
    skin[st:sb, sl:sr] = 1
    skin[mask > 0] = 0          # exclude the nail itself
    return mask, skin