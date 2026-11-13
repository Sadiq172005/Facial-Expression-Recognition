"""Train an RBF-SVM on the 128-d embeddings of a trained mini-XCEPTION.

The CNN acts as the feature extractor and the SVM as the final classifier:
    python -m src.train_svm --model models/fer_ck.keras --fer data/fer2013 --ck data/CK+48
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import SEED
from .data import load_ckplus, load_fer2013, preprocess
from .model import feature_extractor


def embed(extractor, x, flip=False):
    x = preprocess(x)
    if flip:
        x = x[:, :, ::-1, :]
    return extractor.predict(x, batch_size=256, verbose=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Trained mini-XCEPTION (.keras)")
    p.add_argument("--fer")
    p.add_argument("--ck")
    p.add_argument("--out", help="Output .joblib (default: next to the model)")
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--max-fer", type=int, default=15000,
                   help="Cap on FER2013 training samples used for the SVM (keeps fitting fast)")
    args = p.parse_args()
    if not (args.fer or args.ck):
        raise SystemExit("Provide --fer and/or --ck")

    extractor = feature_extractor(tf.keras.models.load_model(args.model))
    fer = load_fer2013(args.fer) if args.fer else None
    ck = load_ckplus(args.ck) if args.ck else None
    rng = np.random.default_rng(SEED)

    feats, labels = [], []
    for ds, cap in ((fer, args.max_fer), (ck, None)):
        if ds is None:
            continue
        x = np.concatenate([ds["train"][0], ds["val"][0]])
        y = np.concatenate([ds["train"][1], ds["val"][1]])
        if cap and len(y) > cap:
            idx = rng.choice(len(y), cap, replace=False)
            x, y = x[idx], y[idx]
        feats += [embed(extractor, x), embed(extractor, x, flip=True)]
        labels += [y, y]
    f_tr, y_tr = np.concatenate(feats), np.concatenate(labels)

    print(f"Fitting SVM on {len(y_tr)} embeddings of dim {f_tr.shape[1]} ...")
    svm = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(  # calibrated probabilities for the real-time overlay
            SVC(kernel="rbf", C=args.C, gamma="scale", class_weight="balanced", random_state=SEED),
            ensemble=False, cv=5),
    )
    svm.fit(f_tr, y_tr)

    metrics = {}
    for name, ds in (("fer2013", fer), ("ckplus", ck)):
        if ds is None:
            continue
        x_te, y_te = ds["test"]
        metrics[f"{name}_test_accuracy"] = float((svm.predict(embed(extractor, x_te)) == y_te).mean())
    out = Path(args.out) if args.out else Path(args.model).with_suffix(".svm.joblib")
    joblib.dump(svm, out)
    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"SVM saved to {out}")


if __name__ == "__main__":
    main()
