import math

import pytest

from runtime_utils import PredictionGate, env_bool, env_float, env_int, scores_to_probabilities


def test_scores_preserve_existing_probabilities():
    assert scores_to_probabilities([0.05, 0.9, 0.03, 0.02]) == pytest.approx(
        [0.05, 0.9, 0.03, 0.02]
    )


def test_scores_apply_stable_softmax_to_logits():
    probabilities = scores_to_probabilities([1000.0, 1001.0])
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[1] > probabilities[0]
    assert all(math.isfinite(value) for value in probabilities)


def test_prediction_gate_rejects_noise_and_requires_stability():
    gate = PredictionGate(confidence_threshold=0.6, required_hits=2, repeat_seconds=6)

    assert not gate.observe("home", 0.59, 0.0)
    assert not gate.observe("home", 0.9, 1.0)
    assert gate.observe("home", 0.9, 2.0)
    assert not gate.observe("home", 0.9, 3.0)
    assert gate.observe("home", 0.9, 8.0)


def test_prediction_gate_announces_changed_label_after_required_hits():
    gate = PredictionGate(required_hits=2)

    assert not gate.observe("home", 0.9, 0.0)
    assert gate.observe("home", 0.9, 1.0)
    assert not gate.observe("wechat_chat", 0.9, 2.0)
    assert gate.observe("wechat_chat", 0.9, 3.0)


def test_environment_parsers(monkeypatch):
    monkeypatch.setenv("BOOL_SETTING", "yes")
    monkeypatch.setenv("INT_SETTING", "8")
    monkeypatch.setenv("FLOAT_SETTING", "0.75")

    assert env_bool("BOOL_SETTING", False)
    assert env_int("INT_SETTING", 1, minimum=1) == 8
    assert env_float("FLOAT_SETTING", 0.1, minimum=0, maximum=1) == 0.75


def test_environment_parsers_reject_invalid_values(monkeypatch):
    monkeypatch.setenv("BOOL_SETTING", "sometimes")
    monkeypatch.setenv("INT_SETTING", "0")

    with pytest.raises(ValueError, match="BOOL_SETTING"):
        env_bool("BOOL_SETTING", False)
    with pytest.raises(ValueError, match="INT_SETTING"):
        env_int("INT_SETTING", 1, minimum=1)
