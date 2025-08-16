"""Lightweight model training and evaluation utilities (Phase 5).

All functions are designed for local execution and small datasets.
They avoid heavyweight architectures and include basic error handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


# -----------------------------
# Data classes for results/bundles
# -----------------------------


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float


@dataclass(frozen=True)
class ClassificationModelBundle:
    """Container bundling a fitted model with label encoding utilities."""

    model: object
    classes: List[str]

    def predict_labels(self, features: np.ndarray) -> List[str]:
        import numpy as _np

        if features.ndim != 2:
            raise ValueError("features must be a 2D array: (num_samples, num_features)")
        raw_pred = self.model.predict(features)
        # raw_pred are integer indices into classes
        return [self.classes[int(i)] for i in _np.asarray(raw_pred).tolist()]


# -----------------------------
# Helper utilities
# -----------------------------


def _validate_2d_features(features: np.ndarray) -> None:
    if features.ndim != 2:
        raise ValueError("features must be 2D (num_samples, num_features)")
    if features.shape[0] == 0:
        raise ValueError("features must contain at least one sample")


def _accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return float(correct) / float(len(y_true))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred))) if y_true.size else 0.0


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if y_true.size else 0.0


# -----------------------------
# Classification: Gender and Emotion
# -----------------------------


def train_gender_model(
    features: np.ndarray,
    labels: Sequence[str],
    model_type: str = "linear_svc",
    random_state: int = 42,
) -> ClassificationModelBundle:
    """Train a lightweight gender classifier.

    - Ensures labels are within {"Male", "Female"}
    - Models: 'linear_svc' (default) or 'logreg'
    """
    _validate_2d_features(features)
    allowed = {"Male", "Female"}
    unique_labels = set(labels)
    if not unique_labels.issubset(allowed):
        raise ValueError(f"Gender labels must be subset of {allowed}, got {unique_labels}")

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if model_type == "linear_svc":
        from sklearn.svm import LinearSVC

        model = make_pipeline(StandardScaler(with_mean=True), LinearSVC(random_state=random_state))
    elif model_type == "logreg":
        from sklearn.linear_model import LogisticRegression

        model = make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(max_iter=200, random_state=random_state),
        )
    else:
        raise ValueError("Unsupported model_type; choose 'linear_svc' or 'logreg'")

    # Map labels to indices based on allowed ordering for predictable outputs
    classes = sorted(list(allowed))  # ['Female', 'Male']
    class_to_index = {c: i for i, c in enumerate(classes)}
    y = np.asarray([class_to_index[c] for c in labels], dtype=int)

    model.fit(features, y)
    return ClassificationModelBundle(model=model, classes=classes)


def evaluate_classification(
    bundle: ClassificationModelBundle,
    features: np.ndarray,
    true_labels: Sequence[str],
) -> ClassificationMetrics:
    _validate_2d_features(features)
    pred_labels = bundle.predict_labels(features)
    acc = _accuracy(true_labels, pred_labels)
    return ClassificationMetrics(accuracy=acc)


def train_emotion_model(
    features: np.ndarray,
    labels: Sequence[str],
    valid_emotions: Optional[Sequence[str]] = None,
    model_type: str = "logreg",
    random_state: int = 42,
) -> ClassificationModelBundle:
    """Train a lightweight multi-class emotion classifier.

    - If `valid_emotions` provided, all labels must be within it
    - Models: 'logreg' (default) or 'linear_svc'
    """
    _validate_2d_features(features)
    unique_labels = sorted(list(set(labels)))
    if valid_emotions is not None:
        valid_set = set(valid_emotions)
        if not set(unique_labels).issubset(valid_set):
            raise ValueError("Emotion labels include values outside the allowed set")

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if model_type == "linear_svc":
        from sklearn.svm import LinearSVC

        model = make_pipeline(StandardScaler(with_mean=True), LinearSVC(random_state=random_state))
    elif model_type == "logreg":
        from sklearn.linear_model import LogisticRegression

        model = make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(max_iter=300, random_state=random_state, multi_class="auto"),
        )
    else:
        raise ValueError("Unsupported model_type; choose 'linear_svc' or 'logreg'")

    classes = sorted(unique_labels)
    class_to_index = {c: i for i, c in enumerate(classes)}
    y = np.asarray([class_to_index[c] for c in labels], dtype=int)

    model.fit(features, y)
    return ClassificationModelBundle(model=model, classes=classes)


# -----------------------------
# Regression: Age prediction
# -----------------------------


def train_age_model(
    features: np.ndarray,
    ages: Sequence[float],
    model_type: str = "rf",
    random_state: int = 42,
):
    """Train a lightweight age regressor.

    Models:
      - 'rf': RandomForestRegressor (default)
      - 'svr': Support Vector Regressor with StandardScaler
    """
    _validate_2d_features(features)
    y = np.asarray(ages, dtype=float)
    if y.ndim != 1 or y.size != features.shape[0]:
        raise ValueError("ages must be a 1D array with same length as features")

    if model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=None
        )
    elif model_type == "svr":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        model = make_pipeline(StandardScaler(with_mean=True), SVR())
    else:
        raise ValueError("Unsupported model_type; choose 'rf' or 'svr'")

    model.fit(features, y)
    return model


def evaluate_regression(model, features: np.ndarray, true_ages: Sequence[float]) -> RegressionMetrics:
    _validate_2d_features(features)
    pred = np.asarray(model.predict(features), dtype=float)
    y_true = np.asarray(true_ages, dtype=float)
    return RegressionMetrics(mae=_mae(y_true, pred), rmse=_rmse(y_true, pred))


__all__ = [
    "ClassificationMetrics",
    "RegressionMetrics",
    "ClassificationModelBundle",
    "train_gender_model",
    "evaluate_classification",
    "train_emotion_model",
    "train_age_model",
    "evaluate_regression",
]


# -----------------------------
# Phase 6: Model Saving/Loading
# -----------------------------


def save_sklearn_model(obj: object, path: str) -> str:
    """Save a scikit-learn model or bundle using joblib.dump().

    This function is format-agnostic for the object; it can be a raw sklearn
    estimator or a small container like `ClassificationModelBundle`.
    """
    from pathlib import Path as _Path
    import joblib as _joblib

    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _joblib.dump(obj, str(p))
    return str(p)


def load_sklearn_model(path: str) -> object:
    """Load a scikit-learn model or bundle using joblib.load()."""
    from pathlib import Path as _Path
    import joblib as _joblib

    p = _Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return _joblib.load(str(p))


def save_keras_model(model: object, path: str) -> str:
    """Save a Keras model using its native `model.save()` API.

    Requires that `model` exposes a `.save(...)` method. No TensorFlow import
    is performed here to avoid unnecessary heavy dependencies if unused.
    """
    from pathlib import Path as _Path

    if not hasattr(model, "save"):
        raise TypeError("Provided model does not have a 'save' method typical of Keras models")

    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Call the model's own save method
    model.save(str(p))
    return str(p)


def load_keras_model(path: str) -> object:
    """Load a Keras model using TensorFlow's `keras.models.load_model`.

    This function attempts a lazy import and raises a clear error if
    TensorFlow/Keras is not available in the environment.
    """
    try:
        from tensorflow import keras  # type: ignore
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            "TensorFlow/Keras is required to load Keras models but is not available."
        ) from error

    from pathlib import Path as _Path
    p = _Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return keras.models.load_model(str(p))


__all__ += [
    "save_sklearn_model",
    "load_sklearn_model",
    "save_keras_model",
    "load_keras_model",
]


# -----------------------------
# Phase 7: Integration Inference Pipeline
# -----------------------------

def _aggregate_feature_matrix_to_vector(feature_matrix: np.ndarray) -> np.ndarray:
    """Aggregate (features, time) matrix to a fixed-length vector.

    Concatenates mean and std over time for each feature, yielding a
    (features*2,) vector. Returns float32.
    """
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be 2D (features, time)")
    if feature_matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    mean_vec = feature_matrix.mean(axis=1)
    std_vec = feature_matrix.std(axis=1)
    return np.concatenate([mean_vec, std_vec]).astype(np.float32)


def make_feature_vector_from_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute a compact feature vector from audio.

    Uses MFCC, Chroma, and Mel-spectrogram, then aggregates to a single vector.
    """
    from utils.feature_extraction import (
        extract_mfcc,
        extract_chroma,
        extract_melspectrogram,
        combine_features,
    )

    mfcc = extract_mfcc(audio, sample_rate)
    chroma = extract_chroma(audio, sample_rate)
    mel = extract_melspectrogram(audio, sample_rate)
    combined = combine_features(mfcc, chroma, mel)
    vector = _aggregate_feature_matrix_to_vector(combined)
    return vector


def run_inference(
    audio_path: str,
    gender_bundle,
    age_model,
    emotion_bundle,
    valid_emotions,
    age_threshold: float = 60.0,
):
    """End-to-end inference routing based on gender and age.

    Workflow:
      - Preprocess audio (denoise + normalize)
      - Extract features and aggregate into a fixed-length vector
      - Predict gender; if 'Female' -> return gender only (reject further preds)
      - If 'Male': predict age; if age <= threshold -> return gender+age
      - If 'Male' and age > threshold: also predict emotion (validated)

    Returns a dict with keys: 'gender', 'age' (float), and optionally 'emotion'.
    """
    from utils.preprocessing import preprocess_audio_file

    audio, sr = preprocess_audio_file(audio_path, save=False)
    feat_vec = make_feature_vector_from_audio(audio, sr)
    if feat_vec.size == 0:
        raise RuntimeError("Failed to compute features from audio")

    X = feat_vec.reshape(1, -1)

    gender_pred = gender_bundle.predict_labels(X)[0]
    if gender_pred not in {"Male", "Female"}:
        raise ValueError("Gender model must output 'Male' or 'Female'")

    result = {"gender": gender_pred}
    if gender_pred == "Female":
        return result

    # Male branch: predict age
    age_value = float(np.asarray(age_model.predict(X), dtype=float).reshape(-1)[0])
    result["age"] = age_value

    if age_value <= float(age_threshold):
        return result

    # Senior branch: also predict emotion
    emotion_pred = emotion_bundle.predict_labels(X)[0]
    if valid_emotions is not None and emotion_pred not in set(valid_emotions):
        raise ValueError("Emotion prediction outside of allowed set")
    result["emotion"] = emotion_pred
    return result


__all__ += [
    "_aggregate_feature_matrix_to_vector",
    "make_feature_vector_from_audio",
    "run_inference",
]

