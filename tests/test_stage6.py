"""Проверки этапа 6: одна база и K, модель без обучения, повтор даёт те же списки."""
import json

import pytest

from src import stage6

ready = pytest.mark.skipif(not (stage6.REPORTS / "final_metrics.csv").exists(), reason="сначала python -m src.stage6")


@ready
def test_same_base_and_k_and_no_fit():
    s = json.loads((stage6.REPORTS / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert s["same_n_and_k_for_both_approaches"] and s["bundle_unchanged_after_scoring"]
    assert s["fit_called"] is False and s["test_used_for_selection"] is False


@ready
def test_repeat_run_gives_same_contacts_and_metrics():
    assert stage6.verify()["ok"]


def test_test_outcomes_are_fully_observed():
    import joblib
    bundle = joblib.load(stage6.BUNDLE)
    d = stage6.load_holdout("test", bundle)
    assert len(d["X"]) == 1916 and set(d["y"]) <= {0, 1}
