"""Метрики кампании по датам T: top-K от всей активной базы, равенства — по customer_id."""
import math
from decimal import Decimal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

KEYS = ["customer_id", "snapshot_at"]


def rank_scores(keys: pd.DataFrame, score, fraction: float) -> pd.DataFrame:
    """Ранжирование по убыванию score внутри каждой T; равенства — customer_id по возрастанию."""
    score = np.asarray(score, dtype=float)
    if len(keys) != len(score) or not np.isfinite(score).all():
        raise ValueError("score должен быть конечным и совпадать по длине с ключами")
    f = keys.reset_index(drop=True)[KEYS].copy()
    f["score"] = score
    f = f.sort_values(["snapshot_at", "score", "customer_id"], ascending=[True, False, True], kind="mergesort")
    f["rank"] = f.groupby("snapshot_at").cumcount() + 1
    n = f.groupby("snapshot_at").customer_id.transform("size")
    f["k"] = n.map(lambda x: math.floor(Decimal(str(fraction)) * x))  # Decimal: без ошибок округления float
    f["selected"] = f["rank"] <= f["k"]
    return f.reset_index(drop=True)


def campaign_metrics(keys, y, score, fraction=0.10):
    """Таблица по T (N, K, ушли, найдено, Recall@K, Precision@K, AP) и сводка macro/pooled."""
    ranked = rank_scores(keys, score, fraction)
    target = pd.Series(np.asarray(y, dtype=int), index=pd.MultiIndex.from_frame(keys.reset_index(drop=True)[KEYS]))
    ranked["target"] = target.reindex(pd.MultiIndex.from_frame(ranked[KEYS])).to_numpy()
    rows = []
    for T, part in ranked.groupby("snapshot_at", sort=True):
        n, k = len(part), int(part.k.iloc[0])
        pos, hits = int(part.target.sum()), int(part.target[part.selected].sum())
        rows.append({"snapshot_at": T, "n_active": n, "k": k, "n_churn": pos, "hits": hits,
                     "recall_at_k": hits / pos if pos else np.nan,
                     "precision_at_k": hits / k if k else np.nan,
                     "average_precision": float(average_precision_score(part.target, part.score)) if pos else np.nan})
    table = pd.DataFrame(rows)
    summary = {
        "macro_recall_at_k": float(table.recall_at_k.mean()),
        "macro_precision_at_k": float(table.precision_at_k.mean()),
        "macro_average_precision": float(table.average_precision.mean()),
        "pooled_recall_at_k": int(table.hits.sum()) / int(table.n_churn.sum()) if table.n_churn.sum() else np.nan,
        "hits": int(table.hits.sum()), "n_churn": int(table.n_churn.sum()), "k_total": int(table.k.sum()),
        "n_dates": len(table),
    }
    return summary, table, ranked


def baseline_score(debt: pd.Series) -> np.ndarray:
    """Score правила по долгу: неизвестный долг ставится после всех известных (это ранжирование, не импутация)."""
    d = pd.to_numeric(debt).astype(float)
    return np.where(d.isna(), -1.0, d.fillna(0).to_numpy())  # долг ≥ 0, поэтому −1 ниже любого известного
