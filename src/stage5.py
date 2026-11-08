"""Этап 5: обучение двух методов, сравнение с baseline и сохранение кандидата.

    python -m src.stage5          # обучение, журнал, выбор, сохранение Pipeline
    python -m src.check_model     # повторная загрузка кандидата и проверка без fit

Вход: data/features (этап 4, не изменяется). Test не загружается в модели и не оценивается.
"""
import hashlib
import json
import platform
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config, features as F
from .load import read_raw_csv, write_csv
from .metrics import KEYS, baseline_score, campaign_metrics
from .preprocess import FORBIDDEN, make_preprocessor

CONFIG_PATH = config.ROOT / "config" / "stage5.json"
REPORTS = config.ROOT / "reports" / "stage5"
MODELS = config.ROOT / "models"
BUNDLE = MODELS / "stage5_candidate.joblib"
FORBIDDEN_X = FORBIDDEN | {"termination", "event_date"}


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def features_sha256() -> str:
    return hashlib.sha256((config.FEATURES_DATA / "features.csv").read_bytes()).hexdigest()


# ---------- вход ----------

def load_data(numeric=F.NUMERIC, categorical=F.CATEGORICAL):
    """Соединение по ключу (не по позиции), проверки и разбор типов. Возвращает train и validation."""
    D = config.FEATURES_DATA
    feats, labels, split = (read_raw_csv(D / f"{n}.csv") for n in ["features", "labels", "split"])
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "result": "OK" if ok else "FAIL", "detail": detail})
        if not ok:
            raise ValueError(f"{name}: {detail}")

    for name, t in [("features", feats), ("labels", labels), ("split", split)]:
        add(f"{name}: ключи непустые и уникальны", not t[KEYS].eq("").any().any() and not t.duplicated(KEYS).any(),
            f"{len(t)} строк")
    kset = lambda t: set(map(tuple, t[KEYS].to_numpy()))  # noqa: E731
    add("Множества ключей совпадают", kset(feats) == kset(labels) == kset(split), f"{len(feats)} ключей")
    add("Имена частей разбиения", set(split.split) <= {"train", "validation", "test"}, ", ".join(sorted(set(split.split))))
    selected = list(numeric) + list(categorical)
    add("Явный список X без ключей, метки и служебных полей",
        len(set(selected)) == len(selected) and not set(selected) & FORBIDDEN_X and set(selected) <= set(feats),
        f"{len(numeric)} числовых + {len(categorical)} категориальных")

    wanted = split[split.split.isin(["train", "validation"])]  # test не загружается
    df = (feats[KEYS + selected].merge(wanted, on=KEYS, how="inner", validate="one_to_one")
          .merge(labels, on=KEYS, how="left", validate="one_to_one"))
    add("Строки не потеряны при соединении", len(df) == len(wanted), f"{len(df)} строк train+validation")
    add("Метки train/validation известны и бинарны", df.churn_target.isin(["0", "1"]).all() and
        df.label_observed.eq("1").all(), "")
    df["churn_target"] = df.churn_target.astype(int)
    for c in numeric:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="raise")
        if np.isinf(df[c]).any():
            raise ValueError(f"бесконечное значение в {c}")
    for c in categorical:
        df[c] = df[c].replace("", np.nan).astype(object)
    df["snapshot_at"] = df.snapshot_at.astype(str)
    df = df.sort_values(KEYS, kind="mergesort").reset_index(drop=True)

    tr, va = df[df.split == "train"], df[df.split == "validation"]
    add("Горизонты train заканчиваются не позже начала validation",
        tr.target_end_exclusive.max() <= va.snapshot_at.min() and tr.snapshot_at.max() < va.snapshot_at.min(),
        f"max конец горизонта train {tr.target_end_exclusive.max()}, начало validation {va.snapshot_at.min()}")
    add("В train есть оба класса", set(tr.churn_target) == {0, 1}, f"уходов в train: {int(tr.churn_target.sum())}")
    counts = df.groupby("snapshot_at").size()
    base = feats.groupby("snapshot_at").size()
    add("Активная база по каждой T сохранена", bool((counts == base.reindex(counts.index)).all()),
        "; ".join(f"{t}: {n}" for t, n in counts.items()))
    parts = {p: {"X": g[selected].reset_index(drop=True), "y": g.churn_target.to_numpy(),
                 "keys": g[KEYS].reset_index(drop=True), "debt": g.debt_prev_rub.reset_index(drop=True)}
             for p, g in [("train", tr), ("validation", va)]}
    return parts, pd.DataFrame(checks)


# ---------- модели ----------

def build_pipeline(exp: dict, seed: int) -> Pipeline:
    params = dict(exp.get("params", {}))
    if exp["family"] == "logistic_regression":
        est = LogisticRegression(**{"max_iter": 2000, "solver": "lbfgs", "random_state": seed, **params})
    elif exp["family"] == "hist_gradient_boosting":
        if params.get("early_stopping", False) is not False:
            raise ValueError("случайная внутренняя ранняя остановка не используется во временной задаче")
        est = HistGradientBoostingClassifier(**{"early_stopping": False, "random_state": seed, **params})
    else:
        raise ValueError(exp["family"])
    # Общая подготовка этапа 4: для деревьев масштабирование не нужно, но безопасно.
    return Pipeline([("preprocess", make_preprocessor(F.NUMERIC, F.CATEGORICAL)), ("model", est)])


def positive_score(pipe, X) -> np.ndarray:
    pos = int(np.flatnonzero(pipe.classes_ == 1)[0])
    return pipe.predict_proba(X)[:, pos]


def fit(exp, seed, X, y):
    pipe = build_pipeline(exp, seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        start = time.perf_counter()
        pipe.fit(X, y)
        seconds = time.perf_counter() - start
    msgs = sorted({f"{w.category.__name__}: {str(w.message).splitlines()[0]}" for w in caught
                   if issubclass(w.category, (ConvergenceWarning, UserWarning, RuntimeWarning))})
    return pipe, seconds, msgs


def temporal_folds(exp, seed, data, folds):
    """Диагностика устойчивости внутри train: подготовка и модель обучаются заново в каждом фолде."""
    keys, X, y = data["keys"], data["X"], data["y"]
    rows = []
    for f in folds:
        lo, hi = f["train"]
        tr = (keys.snapshot_at >= lo) & (keys.snapshot_at <= hi)
        ev = keys.snapshot_at == f["evaluate"]
        pipe, _, _ = fit(exp, seed, X[tr], y[tr])
        s, _, _ = campaign_metrics(keys[ev], y[ev], positive_score(pipe, X[ev]))
        rows.append({"experiment": exp["id"], "fold_train": f"{lo}…{hi}", "fold_evaluate": f["evaluate"],
                     "recall_at_k": s["macro_recall_at_k"], "hits": s["hits"], "n_churn": s["n_churn"],
                     "average_precision": s["macro_average_precision"]})
    return rows


def run():
    cfg = load_config()
    seed, frac = cfg["seed"], cfg["campaign_fraction"]
    data, input_checks = load_data()
    REPORTS.mkdir(parents=True, exist_ok=True)

    log, by_t, fold_rows, models = [], [], [], {}

    def record(exp_id, family, params, part, summary, table, seconds=np.nan, status="ok", msgs=()):
        log.append({"id": exp_id, "family": family, "params": json.dumps(params, ensure_ascii=False), "seed": seed,
                    "feature_version": cfg["feature_version"], "part": part,
                    "recall_at_k": summary["macro_recall_at_k"], "precision_at_k": summary["macro_precision_at_k"],
                    "average_precision": summary["macro_average_precision"],
                    "pooled_recall_at_k": summary["pooled_recall_at_k"], "hits": summary["hits"],
                    "n_churn": summary["n_churn"], "k_total": summary["k_total"], "fit_seconds": seconds,
                    "status": status, "warnings": " | ".join(msgs)})
        by_t.append(table.assign(id=exp_id, part=part))

    # Baseline: та же база, тот же K, те же метрики
    for part in ["train", "validation"]:
        d = data[part]
        s, t, _ = campaign_metrics(d["keys"], d["y"], baseline_score(d["debt"]), frac)
        record("baseline_debt", "rule", {"sort": "debt_prev_rub desc, unknown last, customer_id asc"}, part, s, t)

    for exp in cfg["experiments"]:
        try:
            pipe, seconds, msgs = fit(exp, seed, data["train"]["X"], data["train"]["y"])
        except Exception as e:  # ошибка не должна исчезать из журнала
            record(exp["id"], exp["family"], exp["params"], "train",
                   {k: np.nan for k in ["macro_recall_at_k", "macro_precision_at_k", "macro_average_precision",
                                        "pooled_recall_at_k", "hits", "n_churn", "k_total"]},
                   pd.DataFrame(), status=f"error: {e}")
            continue
        models[exp["id"]] = pipe
        status = "ok" if not msgs else "ok с предупреждениями"
        for part in ["train", "validation"]:
            d = data[part]
            s, t, _ = campaign_metrics(d["keys"], d["y"], positive_score(pipe, d["X"]), frac)
            record(exp["id"], exp["family"], exp["params"], part, s, t, seconds, status, msgs)
        fold_rows += temporal_folds(exp, seed, data["train"], cfg["temporal_folds"])

    tk = data["train"]["keys"]
    for f in cfg["temporal_folds"]:  # baseline на тех же оценочных месяцах фолдов
        ev = tk.snapshot_at == f["evaluate"]
        s, _, _ = campaign_metrics(tk[ev], data["train"]["y"][ev], baseline_score(data["train"]["debt"][ev]), frac)
        fold_rows.append({"experiment": "baseline_debt", "fold_train": "—", "fold_evaluate": f["evaluate"],
                          "recall_at_k": s["macro_recall_at_k"], "hits": s["hits"], "n_churn": s["n_churn"],
                          "average_precision": s["macro_average_precision"]})

    log = pd.DataFrame(log)
    by_t = pd.concat(by_t, ignore_index=True)[["id", "part", "snapshot_at", "n_active", "k", "n_churn", "hits",
                                                "recall_at_k", "precision_at_k", "average_precision"]]
    folds = pd.DataFrame(fold_rows)

    # Выбор по заранее записанному правилу: Recall@K -> AP -> порядок в конфигурации
    order = {e["id"]: i for i, e in enumerate(cfg["experiments"])}
    val = log[(log.part == "validation") & log.id.isin(models)].copy()
    val["order"] = val.id.map(order)
    val = val.sort_values(["recall_at_k", "average_precision", "order"], ascending=[False, False, True])
    chosen_id = val.id.iloc[0]
    chosen = next(e for e in cfg["experiments"] if e["id"] == chosen_id)
    pipe = models[chosen_id]

    vd = data["validation"]
    s_val, _, ranked = campaign_metrics(vd["keys"], vd["y"], positive_score(pipe, vd["X"]), frac)
    base_val = log[(log.id == "baseline_debt") & (log.part == "validation")].iloc[0]
    preds = ranked[KEYS + ["score", "rank", "selected"]]
    write_csv(preds.assign(score=preds.score.map(lambda x: repr(float(x))),
                           selected=preds.selected.map({True: "1", False: "0"})),
              REPORTS / "validation_predictions.csv")

    versions = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                "scikit-learn": sklearn.__version__, "joblib": joblib.__version__, "platform": platform.platform()}
    MODELS.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": pipe, "numeric_columns": list(F.NUMERIC), "categorical_columns": list(F.CATEGORICAL),
              "experiment": chosen, "seed": seed, "campaign_fraction": frac, "feature_version": cfg["feature_version"],
              "features_sha256": features_sha256(), "train_dates": sorted(set(data["train"]["keys"].snapshot_at)),
              "versions": versions}
    joblib.dump(bundle, BUNDLE)

    selection = {
        "rule": cfg["selection_rule"], "selected": chosen_id, "params": chosen["params"],
        "validation": {"recall_at_k": s_val["macro_recall_at_k"], "hits": s_val["hits"], "n_churn": s_val["n_churn"],
                       "precision_at_k": s_val["macro_precision_at_k"], "average_precision": s_val["macro_average_precision"]},
        "baseline_validation": {"recall_at_k": float(base_val.recall_at_k), "hits": int(base_val.hits),
                                "precision_at_k": float(base_val.precision_at_k),
                                "average_precision": float(base_val.average_precision)},
        "extra_hits_vs_baseline": int(s_val["hits"] - base_val.hits),
        "bundle": BUNDLE.relative_to(config.ROOT).as_posix(), "features_sha256": bundle["features_sha256"],
        "test_used": False, "versions": versions,
    }
    write_csv(input_checks, REPORTS / "input_checks.csv")
    write_csv(log.round(6), REPORTS / "experiment_log.csv")
    write_csv(by_t.round(6), REPORTS / "metrics_by_T.csv")
    write_csv(folds.round(6), REPORTS / "temporal_folds.csv")
    (REPORTS / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    return log, selection, folds


def main():
    log, selection, folds = run()
    cols = ["id", "part", "recall_at_k", "hits", "n_churn", "precision_at_k", "average_precision", "fit_seconds", "status"]
    print(log[cols].to_string(index=False))
    print(folds.groupby("experiment")[["recall_at_k", "average_precision"]].mean().round(4).to_string())
    print(json.dumps({k: selection[k] for k in ["selected", "validation", "baseline_validation",
                                                 "extra_hits_vs_baseline"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
