"""Real-time facial expression recognition with emoji overlay.

Webcam:        python -m src.realtime --model models/fer_ck.keras
Video file:    python -m src.realtime --model models/fer_ck.keras --source clip.mp4
Single image:  python -m src.realtime --model models/fer_ck.keras --image face.jpg --save out.jpg
SVM head:      add  --svm models/fer_ck.svm.joblib --classifier svm   (or ensemble)

Keys: q quit | e toggle emoji mask | s save snapshot
"""
import argparse
import time

import cv2

from .emoji import overlay_emoji
from .inference import EmotionClassifier, FaceDetector, ProbabilitySmoother, label_of


def annotate(frame, boxes, probs, show_emoji=True):
    for box, prob in zip(boxes, probs):
        x, y, w, h = box
        emotion, conf = label_of(prob)
        if show_emoji:
            overlay_emoji(frame, emotion, box)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 220, 60), 2)
        text = f"{emotion} {conf * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = max(y - 12, th + 6)
        cv2.rectangle(frame, (x, ty - th - 6), (x + tw + 8, ty + 6), (30, 30, 30), -1)
        cv2.putText(frame, text, (x + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--svm")
    p.add_argument("--classifier", choices=["cnn", "svm", "ensemble"], default="cnn")
    p.add_argument("--source", default="0", help="Webcam index or video path")
    p.add_argument("--image", help="Run on a single image instead of a stream")
    p.add_argument("--save", help="Output image/video path")
    p.add_argument("--no-emoji", action="store_true", help="Draw boxes instead of emoji masks")
    p.add_argument("--no-display", action="store_true", help="Do not open a window (headless)")
    return p.parse_args()


def main():
    args = parse_args()
    detector = FaceDetector()
    classifier = EmotionClassifier(args.model, args.svm, args.classifier)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Cannot read {args.image}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = detector.detect(gray)
        probs = classifier.predict_proba(gray, boxes)
        annotate(frame, boxes, probs, not args.no_emoji)
        for b, p in zip(boxes, probs):
            print(b, label_of(p))
        if args.save:
            cv2.imwrite(args.save, frame)
        if not args.no_display:
            cv2.imshow("Facial Expression Recognition", frame); cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {args.source}")
    writer, smoother = None, ProbabilitySmoother()
    show_emoji, fps, n_frames, t_start = not args.no_emoji, 0.0, 0, time.perf_counter()

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = detector.detect(gray)
        probs = smoother.update(boxes, classifier.predict_proba(gray, boxes))
        annotate(frame, boxes, probs, show_emoji)

        dt = time.perf_counter() - t0
        fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        n_frames += 1

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("Facial Expression Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e"):
                show_emoji = not show_emoji
            if key == ord("s"):
                cv2.imwrite(f"snapshot_{int(time.time())}.png", frame)

    elapsed = time.perf_counter() - t_start
    print(f"Processed {n_frames} frames, average {n_frames / max(elapsed, 1e-9):.1f} FPS")
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
