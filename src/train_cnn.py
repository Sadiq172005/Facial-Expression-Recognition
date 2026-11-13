"""Train mini-XCEPTION on FER2013, CK+, or both.

Recommended two-stage recipe
----------------------------
1) Pre-train on FER2013 (large, noisy, in-the-wild faces):
     python -m src.train_cnn --fer data/fer2013 --epochs 100 --out models/fer_base.keras
2) Joint fine-tune on FER2013 + CK+ (CK+ oversampled) starting from stage 1:
     python -m src.train_cnn --fer data/fer2013 --ck data/CK+48 \
         --init models/fer_base.keras --lr 3e-4 --epochs 60 --out models/fer_ck.keras
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import EMOTIONS, MODELS_DIR, SEED
from .data import class_weights, load_ckplus, load_fer2013, make_dataset, preprocess
from .model import build_mini_xception


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fer", help="FER2013 csv file or folder")
    p.add_argument("--ck", help="CK+ folder (CK+48) or csv file")
    p.add_argument("--init", help="Weights/model to start from (.keras)")
    p.add_argument("--out", default=str(MODELS_DIR / "mini_xception.keras"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    p.add_argument("--ck-oversample", type=int, default=0,
                   help="Times to repeat CK+ train data when mixing with FER2013 (0 = auto)")
    return p.parse_args()


def main():
    args = parse_args()
    if not (args.fer or args.ck):
        raise SystemExit("Provide --fer and/or --ck")
    tf.keras.utils.set_random_seed(SEED)

    fer = load_fer2013(args.fer) if args.fer else None
    ck = load_ckplus(args.ck) if args.ck else None

    parts_x, parts_y = [], []
    if fer:
        parts_x.append(fer["train"][0]); parts_y.append(fer["train"][1])
    if ck:
        rep = 1
        if fer:  # make CK+ roughly a quarter of every epoch
            rep = args.ck_oversample or max(1, len(fer["train"][1]) // (3 * len(ck["train"][1])))
        parts_x += [ck["train"][0]] * rep
        parts_y += [ck["train"][1]] * rep
    x_tr, y_tr = np.concatenate(parts_x), np.concatenate(parts_y)

    # Model selection follows the smaller, cleaner CK+ validation set when available.
    x_va, y_va = ck["val"] if ck else fer["val"]
    print(f"train={len(y_tr)}  val={len(y_va)}  class counts={np.bincount(y_tr, minlength=7)}")

    model = build_mini_xception()
    if args.init:
        model.load_weights(args.init)
        print(f"Loaded initial weights from {args.init}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(out), monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=max(3, args.patience // 3), min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience, verbose=1),
        tf.keras.callbacks.CSVLogger(str(out.with_suffix(".log.csv"))),
    ]
    model.fit(
        make_dataset(x_tr, y_tr, args.batch_size, training=True),
        validation_data=make_dataset(x_va, y_va, args.batch_size, training=False),
        epochs=args.epochs,
        class_weight=class_weights(y_tr),
        callbacks=callbacks,
        verbose=2,
    )

    best = tf.keras.models.load_model(out)
    metrics = {}
    for name, ds in (("fer2013", fer), ("ckplus", ck)):
        if ds is None:
            continue
        x_te, y_te = ds["test"]
        pred = best.predict(preprocess(x_te), batch_size=256, verbose=0).argmax(1)
        metrics[f"{name}_test_accuracy"] = float((pred == y_te).mean())
        metrics[f"{name}_test_size"] = int(len(y_te))
    metrics["classes"] = EMOTIONS
    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Best model saved to {out}")


if __name__ == "__main__":
    main()
