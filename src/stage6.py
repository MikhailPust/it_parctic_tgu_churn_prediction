"""Этап 6: итоговая проверка сохранённого кандидата на test без обучения.

    python -m src.stage6            # расчёт validation + test, списки контактов, разбор, коэффициенты
    python -m src.stage6 --verify   # повторная загрузка той же модели и сверка с сохранёнными результатами

Вход: data/features (этап 4) и models/stage5_candidate.joblib (этап 5). Ничего не обучается и не подбирается.
"""
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .load import read_raw_csv, write_csv
from .metrics import KEYS, baseline_score, campaign_metrics
from .stage5 import BUNDLE, features_sha256

REPORTS = config.ROOT / "reports" / "stage6"
OBSERVED_TO = "2026-10-01"


def bundle_sha256() -> str:
    return hashlib.sha256(BUNDLE.read_bytes()).hexdigest()


def load_holdout(part: str, bundle: dict) -> dict:
    """Validation или test: соединение по ключу, исходы должны быть известны полностью."""
    if part not in ("validation", "test"):
        raise ValueError(part)
    D = config.FEATURES_DATA
    feats, labels, split = (read_raw_csv(D / f"{n}.csv") for n in ["features", "labels", "split"])
    kset = lambda t: set(map(tuple, t[KEYS].to_numpy()))  # noqa: E731
    if kset(feats) != kset(labels) or kset(feats) != kset(split) or feats.duplicated(KEYS).any():
        raise ValueError("ключи features/labels/split не совпадают или повторяются")
    num, cat = bundle["numeric_columns"], bundle["categorical_columns"]
    wanted = split.loc[split.split == part, KEYS]
    df = (feats[KEYS + num + cat].merge(wanted, on=KEYS, validate="one_to_one")
          .merge(labels, on=KEYS, validate="one_to_one").sort_values(KEYS).reset_index(drop=True))
    if not df.churn_target.isin(["0", "1"]).all() or not (df.target_end_exclusive <= OBSERVED_TO).all():
        raise ValueError("исход не наблюдается полностью — неизвестную метку нельзя заменять нулём")
    X = df[num + cat].copy()
    for c in num:
        X[c] = pd.to_numeric(X[c].replace("", np.nan), errors="raise")
    for c in cat:
        X[c] = X[c].replace("", np.nan).astype(object)
    return {"X": X, "y": df.churn_target.astype(int).to_numpy(), "keys": df[KEYS].copy(),
            "debt": pd.to_numeric(df.debt_prev_rub.replace("", np.nan)), "raw": df}


def predict_risk(pipe, X) -> np.ndarray:
    """Только predict_proba: подготовка и модель не обучаются."""
    pos = int(np.flatnonzero(pipe.classes_ == 1)[0])
    s = pipe.predict_proba(X)[:, pos]
    if not np.isfinite(s).all():
        raise ValueError("некорректный score")
    return s


def contacts(ranked: pd.DataFrame) -> pd.DataFrame:
    """Список контактов: ключи, score, rank, selected (метка — отдельно в labels.csv)."""
    out = ranked[KEYS + ["score", "rank", "selected"]].copy()
    out["score"] = out.score.map(lambda x: repr(float(x)))
    out["selected"] = out.selected.map({True: "1", False: "0"})
    return out


def coefficients(pipe) -> pd.DataFrame:
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_[0]
    t = pd.DataFrame({"x_column": names, "coefficient": coef})
    t["abs"] = t.coefficient.abs()
    t["direction"] = np.where(t.coefficient > 0, "выше риск", "ниже риск")
    return t.sort_values("abs", ascending=False).drop(columns="abs").round(6)


def segment_debt(ranked: pd.DataFrame, raw: pd.DataFrame, approach: str) -> pd.DataFrame:
    """Сегменты по долгу на конец прошлого месяца; попадание — в общий top-K, а не новые 10% внутри группы."""
    debt = pd.to_numeric(raw.debt_prev_rub.replace("", np.nan))
    g = raw[KEYS].assign(group=np.select([debt > 0, debt == 0], ["долг > 0", "долг = 0"], "долг неизвестен"))
    j = ranked.merge(g, on=KEYS, validate="one_to_one")
    rows = []
    for grp, p in j.groupby("group"):
        pos, hits, sel = int(p.target.sum()), int((p.target.eq(1) & p.selected).sum()), int(p.selected.sum())
        rows.append({"approach": approach, "group": grp, "n": len(p), "n_selected_in_global_top_k": sel,
                     "n_churn": pos, "hits": hits, "found": f"{hits} из {pos}",
                     "recall_at_global_k": hits / pos if pos else np.nan,
                     "precision_in_selected": hits / sel if sel else np.nan, "small_group": len(p) < config.SMALL_GROUP,
                     "few_positive_outcomes": pos < 5})
    return pd.DataFrame(rows)


def evaluate():
    bundle = joblib.load(BUNDLE)
    if bundle["features_sha256"] != features_sha256():
        raise ValueError("признаки изменились после обучения кандидата")
    pipe, frac = bundle["pipeline"], bundle["campaign_fraction"]
    rows, ranked, data = [], {}, {}
    for part in ["validation", "test"]:
        d = data[part] = load_holdout(part, bundle)
        for approach, score in [("Baseline (долг)", baseline_score(d["debt"])),
                                ("Модель exp03", predict_risk(pipe, d["X"]))]:
            s, t, r = campaign_metrics(d["keys"], d["y"], score, frac)
            ranked[(part, approach)] = r
            row = t.iloc[0]
            rows.append({"period": part, "approach": approach, "snapshot_at": row.snapshot_at,
                         "n_active": int(row.n_active), "k": int(row.k), "hits": int(row.hits),
                         "n_churn": int(row.n_churn), "recall_at_k": row.recall_at_k,
                         "precision_at_k": row.precision_at_k, "average_precision": row.average_precision})
    return bundle, pd.DataFrame(rows), ranked, data


def run():
    sha_before = bundle_sha256()
    bundle, table, ranked, data = evaluate()
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_csv(table.round(6), REPORTS / "final_metrics.csv")
    for part in ["validation", "test"]:
        for approach, name in [("Baseline (долг)", "baseline"), ("Модель exp03", "model")]:
            write_csv(contacts(ranked[(part, approach)]), REPORTS / f"{part}_contacts_{name}.csv")

    tm, tb = ranked[("test", "Модель exp03")], ranked[("test", "Baseline (долг)")]
    both = set(map(tuple, tm[tm.selected][KEYS].to_numpy())) & set(map(tuple, tb[tb.selected][KEYS].to_numpy()))
    seg = pd.concat([segment_debt(tm, data["test"]["raw"], "Модель exp03"),
                     segment_debt(tb, data["test"]["raw"], "Baseline (долг)")], ignore_index=True)
    write_csv(seg.round(5), REPORTS / "test_segment_debt.csv")
    write_csv(coefficients(bundle["pipeline"]), REPORTS / "model_coefficients.csv")

    # Пересечение списков и найденные только одним из подходов
    hit_m = set(map(tuple, tm[tm.selected & tm.target.eq(1)][KEYS].to_numpy()))
    hit_b = set(map(tuple, tb[tb.selected & tb.target.eq(1)][KEYS].to_numpy()))
    cmp = table.groupby("period").agg(n=("n_active", "nunique"), k=("k", "nunique"))
    summary = {
        "bundle": BUNDLE.relative_to(config.ROOT).as_posix(), "bundle_sha256": sha_before,
        "experiment": bundle["experiment"]["id"], "fit_called": False, "test_used_for_selection": False,
        "same_n_and_k_for_both_approaches": bool((cmp.n == 1).all() and (cmp.k == 1).all()),
        "bundle_unchanged_after_scoring": sha_before == bundle_sha256(),
        "test_overlap_selected": len(both), "test_hits_both": len(hit_m & hit_b),
        "test_hits_only_model": len(hit_m - hit_b), "test_hits_only_baseline": len(hit_b - hit_m),
    }
    (REPORTS / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    return table, summary


def verify():
    """Повторная загрузка той же модели на тех же входах: списки контактов и метрики должны совпасть."""
    _, table, ranked, _ = evaluate()
    old = read_raw_csv(REPORTS / "final_metrics.csv")
    same_metrics = bool((old.hits.astype(int).to_numpy() == table.hits.to_numpy()).all() and
                        (old.k.astype(int).to_numpy() == table.k.to_numpy()).all())
    same_lists = True
    for part in ["validation", "test"]:
        for approach, name in [("Baseline (долг)", "baseline"), ("Модель exp03", "model")]:
            new = contacts(ranked[(part, approach)]).reset_index(drop=True)
            prev = read_raw_csv(REPORTS / f"{part}_contacts_{name}.csv")
            same_lists &= bool(new[KEYS + ["rank", "selected"]].astype(str).equals(prev[KEYS + ["rank", "selected"]]) and
                               np.allclose(new.score.astype(float), prev.score.astype(float), rtol=0, atol=1e-12))
    res = {"same_contact_lists": same_lists, "same_metrics": same_metrics, "bundle_sha256": bundle_sha256(),
           "fit_called": False}
    res["ok"] = same_lists and same_metrics
    (REPORTS / "repeat_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    if p.parse_args().verify:
        res = verify()
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if not res["ok"]:
            raise SystemExit("повторный запуск не совпал")
        return
    table, summary = run()
    print(table.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
