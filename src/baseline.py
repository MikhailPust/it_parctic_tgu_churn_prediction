"""Базовое правило: сортировка по долгу предыдущего месяца, лимит K = floor(0,10 × N_active)."""
import math

import numpy as np
import pandas as pd

from . import config


def rank_by_debt(df: pd.DataFrame) -> pd.DataFrame:
    """Долг по убыванию; известный долг раньше неизвестного; равенства — customer_id по возрастанию.

    Неизвестный долг не считается нулём: такие абоненты идут после всех известных, включая долг = 0.
    """
    d = df.assign(_unknown=df.debt_prev_rub.isna(), _debt=df.debt_prev_rub.fillna(0))
    d = d.sort_values(["_unknown", "_debt", "customer_id"], ascending=[True, False, True], kind="mergesort")
    d["rank"] = np.arange(1, len(d) + 1)
    return d.drop(columns=["_unknown", "_debt"])


def evaluate(features: pd.DataFrame, labels: pd.DataFrame, dates) -> pd.DataFrame:
    """Recall@K и Precision@K по каждой дате с числителями и знаменателями."""
    df = features[["customer_id", "snapshot_at", "debt_prev_rub"]].merge(
        labels[["customer_id", "snapshot_at", "churn_target", "label_observed"]], on=["customer_id", "snapshot_at"],
        how="left", validate="one_to_one")
    rows = []
    for T in dates:
        part = rank_by_debt(df[df.snapshot_at == T])
        n_active = len(part)  # K считается от всей активной базы, а не от строк с полными признаками
        k = math.floor(config.CAMPAIGN_SHARE * n_active)
        top = part[part["rank"] <= k]
        n_pos = int(part.churn_target.sum())
        tp = int(top.churn_target.sum())
        rows.append({
            "snapshot_at": T, "n_active": n_active, "k": k, "n_churn": n_pos, "churn_in_top_k": tp,
            "recall_at_k": tp / n_pos if n_pos > 0 else np.nan,
            "precision_at_k": tp / k if k > 0 else np.nan,
            "random_expected_recall": k / n_active if n_active > 0 else np.nan,
            "unknown_debt_rows": int(part.debt_prev_rub.isna().sum()),
            "unknown_debt_in_top_k": int(top.debt_prev_rub.isna().sum()),
            "labels_observed": bool(part.label_observed.all()),
        })
    t = pd.DataFrame(rows)
    total = {
        "snapshot_at": "ИТОГО (micro: Σ числителей / Σ знаменателей)",
        "n_active": t.n_active.sum(), "k": t.k.sum(), "n_churn": t.n_churn.sum(),
        "churn_in_top_k": t.churn_in_top_k.sum(),
        "recall_at_k": t.churn_in_top_k.sum() / t.n_churn.sum(),
        "precision_at_k": t.churn_in_top_k.sum() / t.k.sum(),
        "random_expected_recall": t.k.sum() / t.n_active.sum(),
        "unknown_debt_rows": t.unknown_debt_rows.sum(), "unknown_debt_in_top_k": t.unknown_debt_in_top_k.sum(),
        "labels_observed": bool(t.labels_observed.all()),
    }
    macro = dict(total, snapshot_at="СРЕДНЕЕ по датам (macro)", recall_at_k=t.recall_at_k.mean(),
                 precision_at_k=t.precision_at_k.mean(), random_expected_recall=t.random_expected_recall.mean())
    t["snapshot_at"] = t.snapshot_at.dt.strftime("%Y-%m-%d")
    return pd.concat([t, pd.DataFrame([total, macro])], ignore_index=True)
