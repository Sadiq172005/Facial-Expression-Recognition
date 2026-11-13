"""Inference pipeline: Haar cascade face detection -> 48x48 ROI -> CNN / SVM."""
from __future__ import annotations

import cv2
import joblib
import numpy as np
import tensorflow as tf

from .config import EMOTIONS, HAAR_PATH, IMG_SIZE, NUM_CLASSES
from .model import feature_extractor


def _cascade_path() -> str:
    if HAAR_PATH.exists():
        return str(HAAR_PATH)
    return cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceDetector:
    """Haar cascade detector that runs on a downscaled frame for speed."""

    def __init__(self, detect_width: int = 320, scale_factor: float = 1.2, min_neighbors: int = 5):
        self.cascade = cv2.CascadeClassifier(_cascade_path())
        if self.cascade.empty():
            raise RuntimeError("Could not load the Haar cascade XML file")
        self.detect_width = detect_width
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        ratio = gray.shape[1] / self.detect_width
        small = gray
        if ratio > 1:
            small = cv2.resize(gray, (self.detect_width, int(gray.shape[0] / ratio)),
                               interpolation=cv2.INTER_AREA)
        else:
            ratio = 1.0
        faces = self.cascade.detectMultiScale(
            cv2.equalizeHist(small), scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors, minSize=(30, 30))
        return [tuple(int(v * ratio) for v in f) for f in faces]


class EmotionClassifier:
    """Classifies 48x48 face crops with the CNN, the SVM head, or both averaged."""

    def __init__(self, model_path: str, svm_path: str | None = None, mode: str = "cnn"):
        if mode in ("svm", "ensemble") and not svm_path:
            raise ValueError(f"mode='{mode}' needs --svm")
        self.mode = mode
        model = tf.keras.models.load_model(model_path)
        self.svm = joblib.load(svm_path) if svm_path and mode != "cnn" else None
        # One traced graph returning both probabilities and embeddings.
        both = tf.keras.Model(model.input, [model.output, feature_extractor(model).output])
        self._forward = tf.function(
            lambda x: both(x, training=False),
            input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 1], tf.float32)],
        )
        self._forward(tf.zeros([1, IMG_SIZE, IMG_SIZE, 1]))  # warm-up

    @staticmethod
    def crop(gray: np.ndarray, box) -> np.ndarray:
        x, y, w, h = box
        roi = gray[max(y, 0):y + h, max(x, 0):x + w]
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        return (roi.astype("float32") / 255.0 - 0.5) * 2.0

    def predict_proba(self, gray: np.ndarray, boxes) -> np.ndarray:
        """Return an (n_faces, 7) probability array."""
        if not len(boxes):
            return np.zeros((0, NUM_CLASSES), np.float32)
        batch = np.stack([self.crop(gray, b) for b in boxes])[..., None]
        cnn_prob, feats = (t.numpy() for t in self._forward(tf.constant(batch)))
        if self.mode == "cnn":
            return cnn_prob
        svm_prob = np.zeros_like(cnn_prob)
        svm_prob[:, self.svm.classes_] = self.svm.predict_proba(feats)
        return svm_prob if self.mode == "svm" else (cnn_prob + svm_prob) / 2.0


class ProbabilitySmoother:
    """Exponential moving average of probabilities per tracked face (reduces flicker)."""

    def __init__(self, alpha: float = 0.5, max_dist: float = 80.0):
        self.alpha, self.max_dist = alpha, max_dist
        self.tracks: list[tuple[np.ndarray, np.ndarray]] = []  # (center, probs)

    def update(self, boxes, probs: np.ndarray) -> np.ndarray:
        new_tracks, out = [], []
        for (x, y, w, h), p in zip(boxes, probs):
            c = np.array([x + w / 2, y + h / 2])
            if self.tracks:
                d = [np.linalg.norm(c - tc) for tc, _ in self.tracks]
                j = int(np.argmin(d))
                if d[j] < self.max_dist:
                    p = self.alpha * p + (1 - self.alpha) * self.tracks[j][1]
            new_tracks.append((c, p))
            out.append(p)
        self.tracks = new_tracks
        return np.array(out).reshape(-1, NUM_CLASSES)


def label_of(prob: np.ndarray) -> tuple[str, float]:
    i = int(np.argmax(prob))
    return EMOTIONS[i], float(prob[i])
