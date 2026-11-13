"""Dataset loaders for FER2013 and CK+ and the tf.data input pipeline.

Supported layouts
-----------------
FER2013
  * fer2013.csv with columns: emotion, pixels, Usage
  * folder release:  <root>/train/<emotion>/*.png and <root>/test/<emotion>/*.png
CK+
  * CK+48 folder release: <root>/<emotion>/S010_006_00000013.png
    (the subject id "S010" is used for a subject-independent split)
  * ckextended.csv with columns: emotion, pixels, Usage (label 7 = contempt, skipped)

Every loader returns uint8 grayscale arrays of shape (N, 48, 48) and int labels.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import FOLDER_TO_CLASS, IMG_SIZE, NUM_CLASSES, SEED

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


# --------------------------------------------------------------------------- #
# Low-level readers
# --------------------------------------------------------------------------- #
def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if img.shape != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


def _read_class_folders(root: Path):
    """Read <root>/<emotion>/* and return images, labels, file stems."""
    xs, ys, stems = [], [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = FOLDER_TO_CLASS.get(folder.name.lower())
        if cls is None:  # e.g. "contempt"
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                xs.append(_read_gray(f))
                ys.append(cls)
                stems.append(f.stem)
    if not xs:
        raise ValueError(f"No emotion folders with images found in {root}")
    return np.stack(xs), np.array(ys, dtype=np.int64), stems


def _read_pixel_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["emotion"] < NUM_CLASSES].reset_index(drop=True)
    x = np.stack(
        [np.array(p.split(), dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE) for p in df["pixels"]]
    )
    y = df["emotion"].to_numpy(dtype=np.int64)
    usage = df["Usage"].str.strip().to_numpy() if "Usage" in df.columns else None
    return x, y, usage


def _find_csv(path: Path) -> Path | None:
    if path.is_file() and path.suffix == ".csv":
        return path
    if path.is_dir():
        csvs = sorted(path.glob("*.csv"))
        if csvs:
            return csvs[0]
    return None


# --------------------------------------------------------------------------- #
# Public loaders -> dict(train=(x, y), val=(x, y), test=(x, y))
# --------------------------------------------------------------------------- #
def load_fer2013(path: str | Path) -> dict:
    path = Path(path)
    csv_path = _find_csv(path)
    if csv_path is not None:
        x, y, usage = _read_pixel_csv(csv_path)
        if usage is None:
            raise ValueError("fer2013.csv must contain a 'Usage' column")
        tr, va, te = usage == "Training", usage == "PublicTest", usage == "PrivateTest"
        return {"train": (x[tr], y[tr]), "val": (x[va], y[va]), "test": (x[te], y[te])}

    # Folder release only ships train/ and test/; carve a validation set from train.
    x_tr, y_tr, _ = _read_class_folders(path / "train")
    x_te, y_te, _ = _read_class_folders(path / "test")
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=SEED
    )
    return {"train": (x_tr, y_tr), "val": (x_va, y_va), "test": (x_te, y_te)}


def load_ckplus(path: str | Path, test_size: float = 0.2, val_size: float = 0.1) -> dict:
    """Load CK+ with a deterministic split.

    For the CK+48 folder release the split is subject-independent: all frames of
    one person land in the same partition, so the test score is not inflated by
    near-duplicate frames of people seen during training.
    """
    path = Path(path)
    csv_path = _find_csv(path)
    if csv_path is not None:
        x, y, usage = _read_pixel_csv(csv_path)
        if usage is not None and {"Training", "PrivateTest"} <= set(usage):
            tr, te = usage == "Training", usage == "PrivateTest"
            va = usage == "PublicTest"
            return {"train": (x[tr], y[tr]), "val": (x[va], y[va]), "test": (x[te], y[te])}
        groups = np.arange(len(y))  # no subject ids available
    else:
        x, y, stems = _read_class_folders(path)
        subjects = [s.split("_")[0] for s in stems]
        groups = np.unique(subjects, return_inverse=True)[1]

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    trval_idx, te_idx = next(outer.split(x, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=SEED)
    tr_rel, va_rel = next(inner.split(x[trval_idx], y[trval_idx], groups[trval_idx]))
    tr_idx, va_idx = trval_idx[tr_rel], trval_idx[va_rel]
    return {
        "train": (x[tr_idx], y[tr_idx]),
        "val": (x[va_idx], y[va_idx]),
        "test": (x[te_idx], y[te_idx]),
    }


# --------------------------------------------------------------------------- #
# Preprocessing and tf.data pipeline
# --------------------------------------------------------------------------- #
def preprocess(x: np.ndarray) -> np.ndarray:
    """uint8 (N, 48, 48) -> float32 (N, 48, 48, 1) scaled to [-1, 1]."""
    x = x.astype("float32") / 255.0
    x = (x - 0.5) * 2.0
    return x[..., None]


def build_augmenter() -> tf.keras.Sequential:
    """Augmentation applied on raw [0, 255] pixels (before normalisation)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.2, value_range=(0, 255)),
        ],
        name="augment",
    )


def _normalise(x):
    return (x / 255.0 - 0.5) * 2.0


def make_dataset(x, y, batch_size: int, training: bool) -> tf.data.Dataset:
    raw = x.astype("float32")[..., None]  # keep [0, 255] until after augmentation
    ds = tf.data.Dataset.from_tensor_slices((raw, tf.one_hot(y, NUM_CLASSES)))
    if training:
        aug = build_augmenter()
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True).batch(batch_size)
        ds = ds.map(lambda a, b: (_normalise(aug(a, training=True)), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size).map(lambda a, b: (_normalise(a), b))
    return ds.prefetch(tf.data.AUTOTUNE)


def class_weights(y: np.ndarray) -> dict:
    """Balanced class weights; classes absent from y get weight 0."""
    counts = np.bincount(y, minlength=NUM_CLASSES).astype("float64")
    present = counts > 0
    w = np.zeros(NUM_CLASSES)
    w[present] = counts.sum() / (present.sum() * counts[present])
    return {i: float(v) for i, v in enumerate(w)}
