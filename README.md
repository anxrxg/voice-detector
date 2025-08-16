## Phase 1: Dataset Preparation

Follow the structure described in `filestructure.md`.

Required directories for Phase 1:

- `data/gender/male/`
- `data/gender/female/`
- `data/age/wav/`
- `data/emotion/wav/`

Use the validator to check readiness:

```python
from pathlib import Path
from utils.preprocessing import validate_dataset_structure

status = validate_dataset_structure(Path(__file__).parent)
print(status)
```

This function is read-only and reports any missing directories without modifying your files.

## Phase 2: Audio Preprocessing

Functions are provided in `utils/preprocessing.py`:

- `load_audio(path, target_sample_rate=16000)`
- `reduce_noise(audio, sample_rate, ...)`
- `normalize_audio(audio, target_peak=0.95)`
- `preprocess_audio_file(input_path, output_path=None, save=False)`

Example:

```python
from utils.preprocessing import preprocess_audio_file

audio, sr = preprocess_audio_file("data/age/wav/example.wav")
print(audio.shape, sr)
```

All functions include exception handling and are suitable for local execution.

## Phase 3: Feature Extraction

Utilities in `utils/feature_extraction.py`:

- `extract_mfcc(audio, sample_rate, n_mfcc=13)`
- `extract_chroma(audio, sample_rate, n_chroma=12)`
- `extract_melspectrogram(audio, sample_rate, n_mels=64)`
- `combine_features(mfcc, chroma, melspec)`

Example:

```python
from utils.preprocessing import preprocess_audio_file
from utils.feature_extraction import extract_mfcc, extract_chroma, extract_melspectrogram, combine_features

audio, sr = preprocess_audio_file("data/age/wav/example.wav")
mfcc = extract_mfcc(audio, sr)
chroma = extract_chroma(audio, sr)
mel = extract_melspectrogram(audio, sr)
features = combine_features(mfcc, chroma, mel)
print(features.shape)
```

## Phase 4: Dataset Splitting

Utilities in `utils/preprocessing.py`:

- `split_list(items, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)`
- `prepare_gender_splits(project_root, extensions=("wav", ...))` → `(filepath, label)` with label in {"Male", "Female"}
- `prepare_age_splits(project_root, labels_csv=None, extensions=("wav", ...))` → `(filepath, age)`; age optional if no CSV
- `prepare_emotion_splits(project_root, labels_csv=None, extensions=("wav", ...))` → `(filepath, emotion)`; emotion optional if no CSV

All splits are deterministic with a fixed `seed`.

## Phase 5: Model Training

Utilities in `utils/model_utils.py`:

- `train_gender_model(X, y, model_type='linear_svc'|'logreg')` + `evaluate_classification(...)`
- `train_emotion_model(X, y, valid_emotions=None, model_type='logreg'|'linear_svc')`
- `train_age_model(X, ages, model_type='rf'|'svr')` + `evaluate_regression(...)`

Notes:
- Gender labels must be within {"Male", "Female"}.
- Emotion labels should be within a predefined set if provided.
- Designed for lightweight, local execution.

## Phase 6: Model Saving

Utilities in `utils/model_utils.py`:

- `save_sklearn_model(obj, path)` and `load_sklearn_model(path)` for scikit-learn models/bundles
- `save_keras_model(model, path)` and `load_keras_model(path)` for Keras models

Notes:
- Sklearn models are saved using `joblib.dump()`.
- Keras models are saved via the model's native `.save()`; loading requires TensorFlow/Keras available locally.

## Phase 7: Integration Testing

Use the end-to-end pipeline to run predictions on an audio file:

```python
from utils.model_utils import run_inference

result = run_inference(
    "data/age/wav/example.wav",
    gender_bundle=...,     # trained bundle
    age_model=...,         # trained regressor
    emotion_bundle=...,    # trained bundle
    valid_emotions=["neutral", "happy", "sad"],
    age_threshold=60.0,
)
print(result)
```

Behavior:
- If predicted gender is "Female": returns only `{"gender": "Female"}`.
- If "Male" and age ≤ 60: returns `{"gender": "Male", "age": <value>}`.
- If "Male" and age > 60: returns `{"gender": "Male", "age": <value>, "emotion": <class>}`.

## Minimal GUI

- Entry: `gui/gui_main.py` (Tkinter)
- Helpers: `gui/gui_helpers.py`
- Place trained models in `models/`:
  - `gender_model.pkl`
  - `age_model.pkl` or `age_model.h5`
  - `emotion_model.pkl` or `emotion_model.h5`

Run locally:

```bash
python gui/gui_main.py
```

