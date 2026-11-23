import ast
import json

import miditoolkit
import numpy as np
import pytest


def test_imports():
    import baseline
    import pipeline

    assert callable(baseline.write_submission_predictions)
    assert callable(pipeline.run_all_v2)


def test_task1_features_have_a_fixed_size(tmp_path, monkeypatch):
    import baseline

    monkeypatch.setattr(baseline, "DATAROOT1", tmp_path)
    features = []
    for name, pitches in [("single", [60]), ("phrase", [64, 67, 72])]:
        midi = miditoolkit.MidiFile()
        instrument = miditoolkit.Instrument(program=0)
        for index, pitch in enumerate(pitches):
            instrument.notes.append(
                miditoolkit.Note(velocity=80, pitch=pitch, start=index * 240, end=index * 240 + 180)
            )
        midi.instruments.append(instrument)
        path = tmp_path / f"{name}.mid"
        midi.dump(str(path))
        features.append(baseline._midi_features_task1(path.name))
    for vector in features:
        assert vector.shape == (baseline.TASK1_FEAT_DIM,)
        assert np.isfinite(vector).all()


def test_prediction_formats(tmp_path):
    from baseline import write_submission_predictions

    path = tmp_path / "prediction.json"
    predictions = {"./piece.wav": ["piano"], "./other.wav": []}
    assert write_submission_predictions(predictions, path) == predictions
    assert ast.literal_eval(path.read_text(encoding="utf-8")) == predictions

    normalized = {"piece.wav": ["piano"], "other.wav": []}
    assert write_submission_predictions(predictions, path, True, fmt="json") == normalized
    assert json.loads(path.read_text(encoding="utf-8")) == normalized

    with pytest.raises(ValueError, match="fmt"):
        write_submission_predictions(predictions, path, fmt="other")
    assert json.loads(path.read_text(encoding="utf-8")) == normalized
