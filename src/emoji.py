"""Emoji masks drawn with OpenCV (no image assets required) and alpha overlay."""
from functools import lru_cache

import cv2
import numpy as np

from .config import EMOTIONS

_S = 256  # canvas size
_YELLOW, _DARK, _WHITE = (60, 205, 255), (30, 40, 60), (255, 255, 255)
_FACE_COLOR = {"angry": (60, 80, 235), "disgust": (90, 200, 150), "fear": (245, 200, 170)}


def _pt(x, y):
    return int(x * _S), int(y * _S)


def _eyes(img, ry=0.045, y=0.40):
    for cx in (0.34, 0.66):
        cv2.ellipse(img, _pt(cx, y), _pt(0.045, ry), 0, 0, 360, _DARK, -1, cv2.LINE_AA)


def _brow(img, x1, y1, x2, y2):
    cv2.line(img, _pt(x1, y1), _pt(x2, y2), _DARK, 9, cv2.LINE_AA)


def _arc(img, cy, ry, start, end, rx=0.2):
    cv2.ellipse(img, _pt(0.5, cy), _pt(rx, ry), 0, start, end, _DARK, 9, cv2.LINE_AA)


@lru_cache(maxsize=None)
def emoji(emotion: str) -> np.ndarray:
    """Return a 256x256 BGRA emoji for the given emotion name."""
    img = np.zeros((_S, _S, 3), np.uint8)
    mask = np.zeros((_S, _S), np.uint8)
    cv2.circle(mask, _pt(0.5, 0.5), int(0.47 * _S), 255, -1, cv2.LINE_AA)
    cv2.circle(img, _pt(0.5, 0.5), int(0.47 * _S), _FACE_COLOR.get(emotion, _YELLOW), -1, cv2.LINE_AA)

    if emotion == "happy":
        _eyes(img)
        cv2.ellipse(img, _pt(0.5, 0.58), _pt(0.24, 0.2), 0, 0, 180, _DARK, -1, cv2.LINE_AA)
        cv2.ellipse(img, _pt(0.5, 0.58), _pt(0.2, 0.06), 0, 0, 180, _WHITE, -1, cv2.LINE_AA)
    elif emotion == "sad":
        _eyes(img)
        _brow(img, 0.24, 0.31, 0.40, 0.26); _brow(img, 0.60, 0.26, 0.76, 0.31)
        _arc(img, 0.76, 0.10, 200, 340)
        cv2.ellipse(img, _pt(0.30, 0.54), _pt(0.03, 0.06), 0, 0, 360, (235, 170, 60), -1, cv2.LINE_AA)
    elif emotion == "angry":
        _eyes(img, y=0.43)
        _brow(img, 0.22, 0.27, 0.42, 0.36); _brow(img, 0.58, 0.36, 0.78, 0.27)
        _arc(img, 0.78, 0.09, 200, 340)
    elif emotion == "surprise":
        _eyes(img, ry=0.07, y=0.38)
        _brow(img, 0.25, 0.22, 0.42, 0.20); _brow(img, 0.58, 0.20, 0.75, 0.22)
        cv2.ellipse(img, _pt(0.5, 0.70), _pt(0.10, 0.14), 0, 0, 360, _DARK, -1, cv2.LINE_AA)
    elif emotion == "fear":
        _eyes(img, ry=0.07, y=0.40)
        _brow(img, 0.24, 0.28, 0.40, 0.22); _brow(img, 0.60, 0.22, 0.76, 0.28)
        cv2.ellipse(img, _pt(0.5, 0.72), _pt(0.17, 0.09), 0, 0, 360, _DARK, -1, cv2.LINE_AA)
        cv2.ellipse(img, _pt(0.5, 0.69), _pt(0.13, 0.03), 0, 0, 360, _WHITE, -1, cv2.LINE_AA)
    elif emotion == "disgust":
        _eyes(img, ry=0.02, y=0.42)
        _brow(img, 0.22, 0.30, 0.42, 0.35); _brow(img, 0.58, 0.35, 0.78, 0.30)
        pts = np.array([_pt(0.30, 0.72), _pt(0.40, 0.66), _pt(0.50, 0.72), _pt(0.60, 0.66), _pt(0.70, 0.72)])
        cv2.polylines(img, [pts], False, _DARK, 9, cv2.LINE_AA)
    else:  # neutral
        _eyes(img)
        cv2.line(img, _pt(0.32, 0.70), _pt(0.68, 0.70), _DARK, 9, cv2.LINE_AA)

    return np.dstack([img, mask])


def overlay_emoji(frame: np.ndarray, emotion: str, box, scale: float = 1.15) -> None:
    """Alpha-blend the emoji for `emotion` over the face box (x, y, w, h) in place."""
    x, y, w, h = box
    size = int(max(w, h) * scale)
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = cx - size // 2, cy - size // 2
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + size, frame.shape[1]), min(y0 + size, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    em = cv2.resize(emoji(emotion), (size, size), interpolation=cv2.INTER_AREA)
    em = em[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    alpha = em[..., 3:4].astype(np.float32) / 255.0
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (alpha * em[..., :3] + (1 - alpha) * roi).astype(np.uint8)


if __name__ == "__main__":  # python -m src.emoji -> writes a preview sheet
    sheet = np.full((_S, _S * len(EMOTIONS), 3), 255, np.uint8)
    for i, e in enumerate(EMOTIONS):
        overlay_emoji(sheet, e, (i * _S, 0, _S, _S), scale=0.95)
    cv2.imwrite("emoji_preview.png", sheet)
    print("Wrote emoji_preview.png")
