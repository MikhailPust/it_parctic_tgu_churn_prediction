"""Проверки этапа 5: метрики кампании, правило baseline, защита от утечек и сохранённый кандидат."""
import json

import numpy as np
import pandas as pd
import pytest

from src import stage5
from src.metrics import baseline_score, campaign_metrics, rank_scores


def keys(n, T="2026-08-01"):
    return pd.DataFrame({"customer_id": [f"C{i:02d}" for i in range(n)], "snapshot_at": T})


def test_top_k_floor_ties_by_customer_id():
    k = keys(25)  # K = floor(0,1 × 25) = 2
    score = np.zeros(25)
    score[[7, 3, 12]] = 1.0  # три равных лучших: выбираются C03 и C07 (по возрастанию ID)
    r = rank_scores(k, score, 0.10)
    assert (r.k == 2).all()
    assert r[r.selected].customer_id.tolist() == ["C03", "C07"]


def test_recall_unknown_without_positives_not_zero():
    k = pd.concat([keys(20, "2026-07-01"), keys(20, "2026-08-01")], ignore_index=True)
    y = np.zeros(40, dtype=int)
    y[25] = 1  # уход только на второй дате
    s, t, _ = campaign_metrics(k, y, np.linspace(1, 0, 40))
    assert np.isnan(t.recall_at_k.iloc[0]) and np.isnan(t.average_precision.iloc[0])
    assert s["n_churn"] == 1 and s["n_dates"] == 2


def test_baseline_unknown_debt_ranked_last():
    k = keys(4)
    debt = pd.Series([0.0, np.nan, 300.0, 300.0])
    r = rank_scores(k, baseline_score(debt), 0.5)
    assert r.customer_id.tolist() == ["C02", "C03", "C00", "C01"]


def test_random_internal_early_stopping_is_rejected():
    with pytest.raises(ValueError):
        stage5.build_pipeline({"family": "hist_gradient_boosting", "params": {"early_stopping": True}}, 42)


@pytest.mark.skipif(not (stage5.config.FEATURES_DATA / "features.csv").exists(), reason="нужен этап 4")
def test_loader_excludes_test_and_joins_by_key():
    data, checks = stage5.load_data()
    assert (checks.result == "OK").all()
    assert set(data) == {"train", "validation"}
    assert set(data["validation"]["keys"].snapshot_at) == {"2026-08-01"}
    assert len(data["train"]["X"]) == 15696 and len(data["validation"]["X"]) == 1929
    assert not {"customer_id", "snapshot_at", "churn_target"} & set(data["train"]["X"].columns)


@pytest.mark.skipif(not stage5.BUNDLE.exists(), reason="сначала python -m src.stage5")
def test_saved_candidate_reproduces_validation_without_fit():
    from src.check_model import check
    res = check()
    assert res["ok"], res
    selection = json.loads((stage5.REPORTS / "selection.json").read_text(encoding="utf-8"))
    assert selection["test_used"] is False
