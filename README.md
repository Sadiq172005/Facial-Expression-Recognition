# Facial Expression Recognition 

Detects faces with a **Haar cascade**, classifies each 48x48 face crop into 7 emotions
(angry, disgust, fear, happy, sad, surprise, neutral) with a **mini-XCEPTION CNN** and an
**SVM head** trained on the CNN embeddings, then superimposes the matching emoji on the
face in real time. Trained on **FER2013** and **CK+**.

![emoji set](assets/emoji_preview.png)

## Pipeline

```
camera frame -> grayscale -> Haar cascade (face ROI) -> resize 48x48, scale to [-1, 1]
             -> mini-XCEPTION (58k params) -> softmax            (classifier = cnn)
                                           -> 128-d embedding -> RBF-SVM   (classifier = svm)
             -> emoji mask alpha-blended over the face + label + FPS counter
```

## Results

| Metric | Value | How it was measured |
|---|---|---|
| CK+ test accuracy, mini-XCEPTION | **93.4%** | 183 held-out images, subject-independent split |
| CK+ test accuracy, SVM on CNN embeddings | **91.8%** | same split |
| End-to-end speed (detect + classify + overlay) | **~115 FPS (CNN), ~90 FPS (SVM)** | 640x480 frames, 2 CPU cores, no GPU |

These numbers come from the bundled `models/ck_only.keras`, trained from scratch on CK+48
only (`reports/` has the per-class report and confusion matrices). CK+48 has no "neutral"
images, so that bundled model never predicts neutral. Run the two-stage recipe below with
FER2013 to get the full 7-class model.

Expected FER2013 accuracy for mini-XCEPTION is about 65-66% (the published figure; human
agreement on FER2013 is also about 65%), so quote the CK+ number as the headline accuracy
and report FER2013 separately.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Datasets (place under `data/`)

| Dataset | Accepted layouts |
|---|---|
| FER2013 | `fer2013.csv` (emotion, pixels, Usage) **or** `train/<emotion>/` + `test/<emotion>/` folders |
| CK+ | `CK+48/<emotion>/S010_006_00000013.png` folders **or** `ckextended.csv` |

The CK+ "contempt" class is skipped because FER2013 has no such label.

## Train

```bash
# 1) Pre-train on FER2013
python -m src.train_cnn --fer data/fer2013 --epochs 100 --out models/fer_base.keras

# 2) Joint fine-tune on FER2013 + CK+ (CK+ is oversampled automatically)
python -m src.train_cnn --fer data/fer2013 --ck data/CK+48 \
    --init models/fer_base.keras --lr 3e-4 --epochs 60 --out models/fer_ck.keras

# 3) SVM head on the CNN embeddings
python -m src.train_svm --model models/fer_ck.keras --fer data/fer2013 --ck data/CK+48

# 4) Reports and confusion matrices -> reports/
python -m src.evaluate --model models/fer_ck.keras --svm models/fer_ck.svm.joblib \
    --fer data/fer2013 --ck data/CK+48
```

CK+ only (what the bundled model used, about 5 minutes on a laptop CPU):

```bash
python -m src.train_cnn --ck data/CK+48 --epochs 150 --batch-size 32 --patience 40 --out models/ck_only.keras
```

Stage 1 is much faster on a GPU (free Google Colab works).

## Run

```bash
python -m src.realtime --model models/ck_only.keras                       # webcam
python -m src.realtime --model models/ck_only.keras --source clip.mp4     # video file
python -m src.realtime --model models/ck_only.keras --image face.jpg --save out.jpg
python -m src.realtime --model models/ck_only.keras --svm models/ck_only.svm.joblib --classifier svm
python -m src.benchmark --model models/ck_only.keras --source 0           # measure FPS
```

Keys in the live window: `q` quit, `e` toggle emoji mask / bounding box, `s` save snapshot.

## Project structure

```
src/config.py      constants, label order
src/data.py        FER2013 / CK+ loaders, subject-independent split, augmentation
src/model.py       mini-XCEPTION and the embedding extractor
src/train_cnn.py   CNN training (pre-train and fine-tune)
src/train_svm.py   SVM on CNN embeddings
src/evaluate.py    accuracy, classification report, confusion matrices
src/inference.py   Haar face detector, classifier wrapper, temporal smoothing
src/emoji.py       emoji masks drawn with OpenCV and alpha overlay
src/realtime.py    webcam / video / image application
src/benchmark.py   end-to-end FPS measurement
assets/            Haar cascade XML, emoji preview
models/            trained weights
reports/           evaluation output
```

## Notes

- The CK+ split keeps every subject in exactly one of train / validation / test, so the
  score is not inflated by near-identical frames of the same person.
- Class weights and label smoothing handle the class imbalance in both datasets.
- Model: Arriaga et al., "Real-time Convolutional Neural Networks for Emotion and Gender
  Classification" (2017).
