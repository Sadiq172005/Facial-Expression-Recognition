"""Evaluate the CNN (and optionally the SVM head) on the held-out test sets.

    python -m src.evaluate --model models/fer_ck.keras --svm models/fer_ck.svm.joblib \
        --fer data/fer2013 --ck data/CK+48
Writes a classification report and a confusion matrix image per dataset.
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import EMOTIONS, NUM_CLASSES
from .data import load_ckplus, load_fer2013, preprocess
from .model import feature_extractor


def plot_confusion(cm, labels, title, path):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--svm")
    p.add_argument("--fer")
    p.add_argument("--ck")
    p.add_argument("--out-dir", default="reports")
    args = p.parse_args()

    model = tf.keras.models.load_model(args.model)
    svm = joblib.load(args.svm) if args.svm else None
    extractor = feature_extractor(model) if svm else None
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for name, loader, path in (("fer2013", load_fer2013, args.fer), ("ckplus", load_ckplus, args.ck)):
        if not path:
            continue
        x, y = loader(path)["test"]
        xp = preprocess(x)
        cnn_prob = model.predict(xp, batch_size=256, verbose=0)
        preds = {"cnn": cnn_prob.argmax(1)}
        if svm:
            svm_prob = np.zeros_like(cnn_prob)
            svm_prob[:, svm.classes_] = svm.predict_proba(extractor.predict(xp, batch_size=256, verbose=0))
            preds["svm"] = svm_prob.argmax(1)
            preds["cnn+svm"] = (cnn_prob + svm_prob).argmax(1)

        present = sorted(set(y) | set(np.concatenate(list(preds.values()))))
        names = [EMOTIONS[i] for i in present]
        lines = [f"=== {name} test set ({len(y)} images) ==="]
        for head, pred in preds.items():
            lines.append(f"\n[{head}] accuracy = {accuracy_score(y, pred):.4f}")
            lines.append(classification_report(y, pred, labels=present, target_names=names,
                                               digits=3, zero_division=0))
            plot_confusion(confusion_matrix(y, pred, labels=present), names,
                           f"{name} - {head}", out_dir / f"confusion_{name}_{head.replace('+', '_')}.png")
        report = "\n".join(lines)
        print(report)
        (out_dir / f"report_{name}.txt").write_text(report)


if __name__ == "__main__":
    main()
