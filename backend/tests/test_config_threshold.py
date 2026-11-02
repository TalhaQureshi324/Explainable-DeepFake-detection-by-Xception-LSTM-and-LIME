"""Threshold configuration: central, env-overridable, model-specific."""

import importlib
import sys

from app.core.config import Settings


def test_default_threshold_is_development_setting():
    s = Settings()
    assert s.default_threshold == 0.60
    assert s.threshold_version == "dev-v1"


def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv("DEEPFAKE_THRESHOLD", "0.55")
    s = Settings()
    assert s.default_threshold == 0.55


def test_model_specific_override_wins(monkeypatch):
    monkeypatch.setenv("DEEPFAKE_THRESHOLD", "0.60")
    monkeypatch.setenv("DEEPFAKE_THRESHOLD_V2", "0.5353")
    s = Settings()
    assert s.threshold_for("v2") == 0.5353   # model-specific wins
    assert s.threshold_for("v1") == 0.60     # default applies elsewhere
    assert s.threshold_for("v3") == 0.60     # unknown versions get the default


def test_invalid_override_falls_back(monkeypatch):
    monkeypatch.setenv("DEEPFAKE_THRESHOLD_V1", "not-a-number")
    s = Settings()
    assert s.threshold_for("v1") == s.default_threshold


def test_historical_threshold_is_preserved_in_docs():
    repo_root = Settings().models_dir.parents[1]
    text = (repo_root / "evaluation" / "EVALUATION.md").read_text(encoding="utf-8")
    assert "0.75" in text  # 0.75 remains recorded as the historical baseline
