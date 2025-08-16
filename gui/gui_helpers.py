"""GUI helpers for loading models and running predictions.

This module provides small utilities used by the Tkinter GUI, including
model loading with graceful fallbacks and a thin wrapper for Keras emotion
models to present a `predict_labels` interface compatible with the rest of
the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.model_utils import (
    load_sklearn_model,
    load_keras_model,
)


DEFAULT_EMOTIONS: List[str] = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprised",
]


def parse_emotion_classes(text: str | None) -> List[str]:
    if not text:
        return list(DEFAULT_EMOTIONS)
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]


class KerasEmotionBundle:
    """Adapter for a Keras multi-class classifier to expose predict_labels()."""

    def __init__(self, keras_model: object, classes: List[str]):
        if not hasattr(keras_model, "predict"):
            raise TypeError("keras_model must expose a .predict method")
        if not classes:
            raise ValueError("classes must be a non-empty list")
        self._model = keras_model
        self._classes = list(classes)

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    def predict_labels(self, features: np.ndarray) -> List[str]:
        if features.ndim != 2:
            raise ValueError("features must be shape (num_samples, num_features)")
        probs = self._model.predict(features, verbose=0)
        probs = np.asarray(probs)
        if probs.ndim == 1:
            # Single-output; treat as scores over classes
            probs = probs.reshape(1, -1)
        if probs.shape[1] != len(self._classes):
            raise ValueError("Keras model output dimension does not match classes length")
        indices = np.argmax(probs, axis=1).astype(int).tolist()
        return [self._classes[i] for i in indices]


@dataclass
class LoadedModels:
    gender_bundle: object | None
    age_model: object | None
    emotion_bundle: object | None
    valid_emotions: List[str]


def load_default_models(
    project_root: str | Path,
    valid_emotions: Optional[List[str]] = None,
) -> LoadedModels:
    """Load models from the `models/` directory with sensible fallbacks.

    - Gender: expects `gender_model.pkl` (sklearn bundle)
    - Age: tries `age_model.pkl` (sklearn), else `age_model.h5` (Keras)
    - Emotion: tries `emotion_model.pkl` (sklearn), else `emotion_model.h5` (Keras);
      for Keras, classes must be supplied or defaults are used
    """
    root = Path(project_root).resolve()
    models_dir = root / "models"

    gender_path_pkl = models_dir / "gender_model.pkl"
    age_path_pkl = models_dir / "age_model.pkl"
    age_path_h5 = models_dir / "age_model.h5"
    emotion_path_pkl = models_dir / "emotion_model.pkl"
    emotion_path_h5 = models_dir / "emotion_model.h5"

    # Gender
    gender_bundle = None
    if gender_path_pkl.exists():
        try:
            gender_bundle = load_sklearn_model(str(gender_path_pkl))
        except Exception:
            gender_bundle = None

    # Age
    age_model = None
    if age_path_pkl.exists():
        try:
            age_model = load_sklearn_model(str(age_path_pkl))
        except Exception:
            age_model = None
    elif age_path_h5.exists():
        try:
            age_model = load_keras_model(str(age_path_h5))
        except Exception:
            age_model = None

    # Emotion
    emotion_bundle = None
    emotions = list(valid_emotions) if valid_emotions else list(DEFAULT_EMOTIONS)
    if emotion_path_pkl.exists():
        try:
            emotion_bundle = load_sklearn_model(str(emotion_path_pkl))
        except Exception:
            emotion_bundle = None
    elif emotion_path_h5.exists():
        try:
            keras_model = load_keras_model(str(emotion_path_h5))
            emotion_bundle = KerasEmotionBundle(keras_model, emotions)
        except Exception:
            emotion_bundle = None

    return LoadedModels(
        gender_bundle=gender_bundle,
        age_model=age_model,
        emotion_bundle=emotion_bundle,
        valid_emotions=emotions,
    )

