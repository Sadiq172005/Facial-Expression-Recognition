"""Shared constants for the facial expression recognition system."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
HAAR_PATH = ASSETS_DIR / "haarcascade_frontalface_default.xml"

IMG_SIZE = 48
# FER2013 label order (index == class id in fer2013.csv)
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(EMOTIONS)

# Folder names used by the different dataset releases -> our class id.
# "contempt" (CK+ only) has no FER2013 counterpart and is skipped.
FOLDER_TO_CLASS = {
    "angry": 0, "anger": 0,
    "disgust": 1, "disgusted": 1,
    "fear": 2, "fearful": 2, "scared": 2,
    "happy": 3, "happiness": 3,
    "sad": 4, "sadness": 4,
    "surprise": 5, "surprised": 5,
    "neutral": 6,
}

SEED = 42
