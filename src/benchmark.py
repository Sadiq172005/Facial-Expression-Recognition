"""Measure end-to-end pipeline speed (Haar detection + classification + emoji overlay).

    python -m src.benchmark --model models/fer_ck.keras --source 0          # webcam
    python -m src.benchmark --model models/fer_ck.keras --image face.jpg    # one frame repeated
Camera capture time is excluded so the number reflects processing speed only.
"""
import argparse
import time

import cv2
import numpy as np

from .inference import EmotionClassifier, FaceDetector
from .realtime import annotate


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--svm")
    p.add_argument("--classifier", choices=["cnn", "svm", "ensemble"], default="cnn")
    p.add_argument("--source", default="0")
    p.add_argument("--image")
    p.add_argument("--frames", type=int, default=300)
    args = p.parse_args()

    detector = FaceDetector()
    classifier = EmotionClassifier(args.model, args.svm, args.classifier)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"Cannot read {args.image}")
        frames = [img] * args.frames
    else:
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
        frames = []
        while len(frames) < args.frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            raise SystemExit("No frames captured")

    times, with_face = [], 0
    for f in frames:
        f = f.copy()
        t0 = time.perf_counter()
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        boxes = detector.detect(gray)
        probs = classifier.predict_proba(gray, boxes)
        annotate(f, boxes, probs)
        times.append(time.perf_counter() - t0)
        with_face += bool(len(boxes))
    t = np.array(times[5:] or times) * 1000
    h, w = frames[0].shape[:2]
    print(f"frames={len(frames)} ({w}x{h}), frames with a face={with_face}, classifier={args.classifier}")
    print(f"mean {t.mean():.1f} ms | p95 {np.percentile(t, 95):.1f} ms | {1000 / t.mean():.1f} FPS")


if __name__ == "__main__":
    main()
