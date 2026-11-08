"""Повторная загрузка сохранённого кандидата в новом процессе и проверка validation без fit.

    python -m src.check_model

Сравниваются ключи, score (допуск 1e-12), rank, top-K и метрики с файлами, записанными при обучении.
"""
import json

import joblib
import numpy as np
import pandas as pd

from .load import read_raw_csv
from .metrics import KEYS, campaign_metrics
from .stage5 import BUNDLE, REPORTS, features_sha256, load_data


def check() -> dict:
    bundle = joblib.load(BUNDLE)
    pipe = bundle["pipeline"]
    if bundle["features_sha256"] != features_sha256():
        raise ValueError("features.csv изменился после обучения кандидата")
    data, _ = load_data(bundle["numeric_columns"], bundle["categorical_columns"])
    v = data["validation"]
    pos = int(np.flatnonzero(pipe.classes_ == 1)[0])
    score = pipe.predict_proba(v["X"])[:, pos]  # только predict, без fit
    summary, _, ranked = campaign_metrics(v["keys"], v["y"], score, bundle["campaign_fraction"])

    old = read_raw_csv(REPORTS / "validation_predictions.csv")
    old["score"] = old.score.astype(float)
    paired = ranked.merge(old, on=KEYS, how="outer", validate="one_to_one", suffixes=("_new", "_old"), indicator=True)
    selection = json.loads((REPORTS / "selection.json").read_text(encoding="utf-8"))
    res = {
        "keys_match": bool((paired["_merge"] == "both").all()),
        "max_abs_score_diff": float(np.abs(paired.score_new - paired.score_old).max()),
        "scores_match": bool(np.allclose(paired.score_new, paired.score_old, rtol=1e-12, atol=1e-12)),
        "ranks_match": bool((paired.rank_new == paired.rank_old.astype(int)).all()),
        "top_k_match": bool((paired.selected_new.map({True: "1", False: "0"}) == paired.selected_old).all()),
        "metrics_match": bool(summary["hits"] == selection["validation"]["hits"] and
                              np.isclose(summary["macro_recall_at_k"], selection["validation"]["recall_at_k"])),
        "fitted_without_refit": True,
        "experiment": bundle["experiment"]["id"],
        "validation_hits": summary["hits"], "validation_n_churn": summary["n_churn"],
        "validation_recall_at_k": summary["macro_recall_at_k"],
    }
    res["ok"] = all(res[k] for k in ["keys_match", "scores_match", "ranks_match", "top_k_match", "metrics_match"])
    (REPORTS / "reload_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def main():
    res = check()
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if not res["ok"]:
        raise SystemExit("загруженный кандидат не воспроизводит validation-результаты")


if __name__ == "__main__":
    main()
