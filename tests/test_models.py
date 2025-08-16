import numpy as np

from utils.model_utils import (
    train_gender_model,
    evaluate_classification,
    train_emotion_model,
    train_age_model,
    evaluate_regression,
    save_sklearn_model,
    load_sklearn_model,
    make_feature_vector_from_audio,
    run_inference,
)


def _toy_classification_features(n=30, d=8, classes=("Male", "Female")):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.array([classes[i % len(classes)] for i in range(n)])
    return X, y


def _toy_regression_features(n=40, d=10):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=(d,)).astype(np.float32)
    y = X @ w + 10.0
    return X, y


def test_train_evaluate_gender_logreg():
    X, y = _toy_classification_features()
    bundle = train_gender_model(X, y, model_type="logreg")
    metrics = evaluate_classification(bundle, X, y)
    assert 0.0 <= metrics.accuracy <= 1.0


def test_train_evaluate_emotion_linear_svc():
    X, y = _toy_classification_features(classes=("neutral", "happy", "sad"))
    bundle = train_emotion_model(X, y, model_type="linear_svc")
    metrics = evaluate_classification(bundle, X, y)
    assert 0.0 <= metrics.accuracy <= 1.0


def test_train_evaluate_age_rf_and_svr():
    X, y = _toy_regression_features()
    rf = train_age_model(X, y, model_type="rf")
    rf_metrics = evaluate_regression(rf, X, y)
    assert rf_metrics.mae >= 0.0 and rf_metrics.rmse >= 0.0

    svr = train_age_model(X, y, model_type="svr")
    svr_metrics = evaluate_regression(svr, X, y)
    assert svr_metrics.mae >= 0.0 and svr_metrics.rmse >= 0.0


def test_save_and_load_sklearn_bundle(tmp_path):
    X, y = _toy_classification_features()
    bundle = train_gender_model(X, y, model_type="logreg")
    path = tmp_path / "gender_model.pkl"
    saved_path = save_sklearn_model(bundle, str(path))
    loaded = load_sklearn_model(saved_path)
    # Basic prediction roundtrip
    preds = loaded.predict_labels(X[:5])
    assert len(preds) == 5


def test_integration_routing_with_synthetic_features(tmp_path):
    # Create tiny synthetic audio and fake models by training on toy features
    # Build a synthetic feature vector by using the feature extractor on a sine
    sr = 16000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False, dtype=np.float32)
    audio = np.sin(2 * np.pi * 220 * t).astype(np.float32)

    # Create a minimal pipeline: train models on vectors derived from audio variants
    from utils.model_utils import _aggregate_feature_matrix_to_vector
    from utils.feature_extraction import extract_mfcc, extract_chroma, extract_melspectrogram, combine_features

    # Two samples with trivial differences
    mfcc = extract_mfcc(audio, sr)
    chroma = extract_chroma(audio, sr)
    mel = extract_melspectrogram(audio, sr)
    vec = _aggregate_feature_matrix_to_vector(combine_features(mfcc, chroma, mel))

    X = np.stack([vec, vec * 0.9], axis=0)
    # Gender labels alternate; ensure allowed set
    y_gender = np.array(["Male", "Female"])
    gender_bundle = train_gender_model(X, y_gender, model_type="logreg")

    # Age regression: simple mapping
    y_age = np.array([65.0, 30.0])
    age_model = train_age_model(X, y_age, model_type="rf")

    # Emotion classifier with small set
    y_emotion = np.array(["neutral", "happy"])
    emotion_bundle = train_emotion_model(X, y_emotion, valid_emotions=["neutral", "happy"], model_type="logreg")

    # Save audio to tmp and run the full pipeline
    in_wav = tmp_path / "sample.wav"
    import soundfile as sf
    sf.write(str(in_wav), audio, sr)

    result = run_inference(
        str(in_wav),
        gender_bundle=gender_bundle,
        age_model=age_model,
        emotion_bundle=emotion_bundle,
        valid_emotions=["neutral", "happy"],
        age_threshold=60.0,
    )

    assert result.get("gender") in ("Male", "Female")
    if result.get("gender") == "Male":
        assert isinstance(result.get("age"), float)
        if result["age"] > 60.0:
            assert result.get("emotion") in ("neutral", "happy")

