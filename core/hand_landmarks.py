"""Nail & skin box derivation from MediaPipe hand landmarks.

MediaPipe HandLandmarker detects hands and returns 21 landmarks per hand.
There is no "nail detector" in MediaPipe, so nail boxes are derived
geometrically from the fingertip and the distal (DIP) joint of each finger,
and skin boxes are placed just proximal (towards the wrist) of the nail box.

All geometry is done in NORMALISED coordinates (0..1), then scaled to pixels.

Landmark ids (https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker):
  thumb=4, index=8, middle=12, ring=16, pinky=20
  DIP ids = fingertip_id - 1
"""

from __future__ import annotations

import math

import numpy as np

from core.config import CANDIDATE_FINGERS, HAND_LANDMARKER_MODEL

FINGER_NAMES = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}


class NailSkinBoxes:
    """nail_box / skin_box are [top, left, bottom, right] in PIXELS."""

    def __init__(self, finger_id: int, nail_box, skin_box, box_center_norm):
        self.finger_id = finger_id
        self.finger_name = FINGER_NAMES.get(finger_id, str(finger_id))
        self.nail_box = nail_box
        self.skin_box = skin_box
        self.box_center_norm = box_center_norm  # (x, y) normalized nail centre


class HandGeometry:
    """Tunable parameters mapping landmarks -> nail/skin boxes.

    All lengths are multiples of the fingertip->DIP segment length (s).
    """

    def __init__(
        self,
        off: float = 0.42,      # nail centre is this far inward from the tip
        nail_w: float = 1.35,   # nail box width  (across the finger)  = nail_w * s
        nail_l: float = 1.05,   # nail box length (along the finger)   = nail_l * s
        gap: float = 0.30,      # gap between nail and skin boxes       = gap * s
        skin_l: float = 0.95,   # skin box length                       = skin_l * s
        skin_w: float = 1.45,   # skin box width                        = skin_w * s
    ):
        self.off = off
        self.nail_w = nail_w
        self.nail_l = nail_l
        self.gap = gap
        self.skin_l = skin_l
        self.skin_w = skin_w

    def param_dict(self) -> dict:
        return {
            "off": self.off,
            "nail_w": self.nail_w,
            "nail_l": self.nail_l,
            "gap": self.gap,
            "skin_l": self.skin_l,
            "skin_w": self.skin_w,
        }


def _norm_rect_bbox(center_n, half_w_n: float, half_l_n: float, u, p):
    """Axis-aligned bbox of a rotated rect, in normalised coords.

    center_n: (x, y) rect centre (0..1)
    half_w_n: half width  (across the finger, along perpendicular p)
    half_l_n: half length (along the finger, along unit u)
    returns (top, left, bottom, right) in normalised coords, clipped to [0,1].
    """
    cx, cy = center_n
    xs, ys = [], []
    for a, b in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        xs.append(cx + a * p[0] * half_w_n + b * u[0] * half_l_n)
        ys.append(cy + a * p[1] * half_w_n + b * u[1] * half_l_n)
    top = max(0.0, min(ys))
    left = max(0.0, min(xs))
    bottom = min(1.0, max(ys))
    right = min(1.0, max(xs))
    return (top, left, bottom, right)


def boxes_for_finger(
    landmarks, finger_id: int, img_w: int, img_h: int, geo: HandGeometry
) -> NailSkinBoxes | None:
    """Derive nail/skin boxes for one finger. Returns None if unusable."""
    tip = landmarks[finger_id]
    dip = landmarks[finger_id - 1]
    s = math.hypot(tip.x - dip.x, tip.y - dip.y)
    if s < 1e-4 or not (0 < tip.x <= 1 and 0 < tip.y <= 1):
        return None
    ux, uy = (tip.x - dip.x) / s, (tip.y - dip.y) / s  # outward along finger
    px, py = -uy, ux  # perpendicular

    # ----- nail box -----
    nail_center_n = (tip.x - geo.off * s * ux, tip.y - geo.off * s * uy)
    nail_norm = _norm_rect_bbox(nail_center_n, geo.nail_w * s / 2, geo.nail_l * s / 2, (ux, uy), (px, py))

    # ----- skin box, further inward along the finger -----
    skin_off = geo.off + geo.nail_l / 2 + geo.gap + geo.skin_l / 2
    skin_center_n = (tip.x - skin_off * s * ux, tip.y - skin_off * s * uy)
    skin_norm = _norm_rect_bbox(skin_center_n, geo.skin_w * s / 2, geo.skin_l * s / 2, (ux, uy), (px, py))

    def to_px(box_norm):
        t, l, b, r = box_norm
        return [int(t * img_h), int(l * img_w), int(b * img_h), int(r * img_w)]

    nail_box = to_px(nail_norm)
    skin_box = to_px(skin_norm)
    if nail_box[2] - nail_box[0] < 2 or nail_box[3] - nail_box[1] < 2:
        return None
    return NailSkinBoxes(finger_id, nail_box, skin_box, nail_center_n)


class HandLandmarkExtractor:
    """Lazy wrapper around the MediaPipe HandLandmarker vision task."""

    def __init__(self, model_path=HAND_LANDMARKER_MODEL, num_hands: int = 1):
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=num_hands)
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, img_rgb: np.ndarray):
        """Return the 21 normalized landmarks of the best-scoring hand or None."""
        return self.detect_ranked(img_rgb)[0]

    def detect_ranked(self, img_rgb: np.ndarray):
        """Return (landmarks, score) of the best-scoring hand, or (None, 0.0)."""
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None, 0.0
        best, best_score = None, -1.0
        for hand_index, hand in enumerate(result.handedness):
            score = float(hand[0].score)
            if score > best_score:
                best_score = score
                best = result.hand_landmarks[hand_index]
        return best, best_score


def derive_all_boxes(landmarks, img_w: int, img_h: int, geo: HandGeometry, fingers=None) -> list[NailSkinBoxes]:
    fingers = fingers if fingers is not None else CANDIDATE_FINGERS
    return [b for f in fingers if (b := boxes_for_finger(landmarks, f, img_w, img_h, geo)) is not None]