"""Этап 4: единый запуск формирования признаков, baseline и преобразования.

    python -m src.stage4

Вход: data/prepared (этап 2) и data/raw/lesson4. Выход: data/features, reports/stage4.
Метрики считаются только на train; исходы validation/test не используются.
"""
import json

import numpy as np
import pandas as pd
import sklearn

from . import baseline, config, features as F, snapshots
from .clean import fmt_bool, fmt_date, fmt_ts
from .eda import independent_base_counts
from .load import write_csv
from .preprocess import make_preprocessor

KEYS = ["customer_id", "snapshot_at"]


# ---------- сериализация ----------

def _fmt_num(s: pd.Series) -> pd.Series:
    return s.map(lambda x: "" if pd.isna(x) else (str(int(x)) if float(x).is_integer() else f"{x:.6f}".rstrip("0")))


def ser_features(df):
    out = df.copy()
    out["snapshot_at"] = fmt_date(out.snapshot_at)
    for c in F.NUMERIC:
        out[c] = _fmt_num(out[c])
    for c in F.CATEGORICAL:
        out[c] = out[c].fillna("")
    return out


def ser_labels(df):
    out = df.copy()
    out["snapshot_at"] = fmt_date(out.snapshot_at)
    out["target_end_exclusive"] = fmt_date(out.target_end_exclusive)
    out["label_observed"] = fmt_bool(out.label_observed.astype(bool))
    out["churn_target"] = out.churn_target.map(lambda x: "" if pd.isna(x) else str(int(x)))
    return out


def ser_audit(df):
    out = df.copy()
    for c in ["snapshot_at", "window_start", "window_end_exclusive", "prev_month"]:
        out[c] = fmt_date(out[c])
    for c in ["max_billing_available_at", "max_support_available_at", "max_digital_available_at"]:
        out[c] = fmt_ts(out[c])
    for c in ["prev_billing_known", "prev_digital_known"]:
        out[c] = fmt_bool(out[c].astype(bool))
    return out


# ---------- проверки нового источника ----------

def digital_checks(digital, subs, billing):
    d = digital
    month_end = (d.usage_month + pd.offsets.MonthBegin(1)).dt.tz_localize(config.TZ)
    obs = d.observation_status.eq("observed")
    metrics = ["observed_days", "app_sessions", "payment_attempts", "payment_failures", "selfservice_actions"]
    lag_h = (d.available_at - month_end).dt.total_seconds() / 3600
    pairs_equal = set(zip(d.customer_id, d.usage_month)) == set(zip(billing.customer_id, billing.billing_month))
    rows = [
        ("Строк", len(d), ""),
        ("Повтор ключа (customer_id, usage_month)", int(d.duplicated(["customer_id", "usage_month"]).sum()), ""),
        ("Ссылка на неизвестного абонента", int((~d.customer_id.isin(subs.customer_id)).sum()), ""),
        ("Пары совпадают с billing_clean", int(pairs_equal), "1 = да"),
        ("Статус вне observed/not_enabled/export_gap",
         int((~d.observation_status.isin(["observed", "not_enabled", "export_gap"])).sum()), ""),
        ("observed: пустые показатели", int(d.loc[obs, metrics].isna().any(axis=1).sum()), ""),
        ("not_enabled/export_gap: заполненные показатели", int(d.loc[~obs, metrics].notna().any(axis=1).sum()), ""),
        ("Отрицательные показатели", int((d[metrics] < 0).any(axis=1).sum()), ""),
        ("payment_failures > payment_attempts", int((d.payment_failures > d.payment_attempts).sum()), ""),
        ("available_at раньше конца месяца", int((lag_h < 0).sum()), ""),
        ("Отчёт опубликован позже 00:00 1-го числа", int((lag_h > 0).sum()),
         "задержка, ч: " + ", ".join(f"{int(k)}: {v}" for k, v in lag_h.value_counts().sort_index().items())),
    ] + [(f"Статус {k}", int(v), "") for k, v in d.observation_status.value_counts().items()]
    return pd.DataFrame(rows, columns=["check", "value", "detail"])


# ---------- проверки признаков ----------

def run_checks(features, labels, split, inputs, prep_info):
    events, subs, billing, support, digital = inputs
    res = []

    def add(name, ok, detail):
        res.append({"check": name, "result": "OK" if ok else "FAIL", "detail": detail})

    add("Ключ (customer_id, snapshot_at) уникален во всех слоях",
        all(not t.duplicated(KEYS).any() for t in (features, labels, split)) and
        features[KEYS].equals(labels[KEYS]) and features[KEYS].equals(split[KEYS]),
        f"{len(features)} строк; порядок ключей одинаков в features/labels/split")
    base = independent_base_counts(config.ALL_DATES)
    m = features.groupby("snapshot_at").size().rename("n").reset_index().merge(base, on="snapshot_at")
    add("Число строк по T = независимо рассчитанная активная база", bool((m.n == m.independent_base).all()),
        "; ".join(f"{d:%d.%m.%Y}: {a}" for d, a in zip(m.snapshot_at, m.n)))

    # Будущие записи не меняют прошлый снимок
    T0 = pd.Timestamp("2026-03-01")
    T0m = T0.tz_localize(config.TZ)
    ids = snapshots.active_base(snapshots.contract_table(events), T0).sort_values()
    ref, _ = F.features_at(T0, ids, subs, billing, support, digital)
    b2 = billing.copy()
    fut = b2.available_at > T0m
    b2.loc[fut, ["traffic_gb", "closing_debt_rub"]] *= 10
    s2 = pd.concat([support, support[support.available_at > T0m].assign(
        ticket_id=lambda x: x.ticket_id + "_future", available_at=lambda x: x.available_at + pd.Timedelta(days=1))])
    d2 = digital.copy()
    d2.loc[d2.available_at > T0m, "app_sessions"] = 999
    alt, _ = F.features_at(T0, ids, subs, b2, s2, d2)
    trunc, _ = F.features_at(T0, ids, subs, billing[billing.available_at <= T0m], support[support.available_at <= T0m],
                             digital[digital.available_at <= T0m])
    add("Будущие записи не меняют признаки (T = 01.03.2026)", ref.equals(alt) and ref.equals(trunc),
        "изменены/добавлены строки с available_at > T и отдельно удалены все будущие строки")

    # Изменение будущей метки не меняет признаки
    e2 = events.copy()
    ter = e2.event_type.eq("termination") & (pd.to_datetime(e2.event_date) >= T0)
    e2.loc[ter, "event_date"] = (pd.to_datetime(e2.loc[ter, "event_date"]) + pd.Timedelta(days=40)).dt.strftime(
        "%Y-%m-%d")
    f_alt, lab_alt, _, _ = F.build(e2, subs, billing, support, digital, dates=[T0])
    lab_ref = labels[labels.snapshot_at == T0].reset_index(drop=True)
    add("Изменение будущей метки не меняет признаки", f_alt.equals(ref) and not lab_alt.churn_target.equals(
        lab_ref.churn_target), "расторжения ≥ T сдвинуты на 40 дней: метки изменились, признаки — нет")

    # Короткая история и пропущенные месяцы
    one = features.billing_months_known_3m == 1
    ok_short = ((features.traffic_mean_3m_gb - features.traffic_prev_gb).abs()[one & features.traffic_prev_gb.notna()]
                < 1e-9).all()
    add("История из одного месяца не превращается в три месяца с нулями", bool(ok_short),
        f"строк с одним месяцем истории: {int(one.sum())}; среднее = значению единственного месяца")
    leaked = (features.traffic_prev_gb.isna() & features.traffic_change_1m_gb.notna()).sum() + \
        (features.debt_prev_rub.isna() & features.debt_change_1m_rub.notna()).sum()
    add("Пропущенный месяц не подменяется при расчёте разности", leaked == 0,
        f"разность при неизвестном месяце: {int(leaked)}; известных разностей трафика: "
        f"{int(features.traffic_change_1m_gb.notna().sum())}")

    # Цифровой источник: ноль и неизвестное различаются
    zero_att = features.payment_attempts_prev == 0
    non_obs = features.digital_status_prev != "observed"
    bad = int(features.payment_failure_rate_prev[zero_att].notna().sum() + features.app_sessions_prev[non_obs].notna().sum())
    add("Ноль попыток -> доля ошибок неизвестна; not_enabled/export_gap/поздний отчёт -> сессии неизвестны", bad == 0,
        f"попыток = 0: {int(zero_att.sum())}; статусы: " +
        ", ".join(f"{k} {v}" for k, v in features.digital_status_prev.value_counts().items()))

    add("Преобразователь обучен только на train", prep_info["fit_rows"] == prep_info["shapes"]["train"][0],
        f"n_samples_seen = {prep_info['fit_rows']}")
    add("Transform validation/test без fit, строки и ключи сохранены", prep_info["rows_preserved"],
        ", ".join(f"{k}: {v[0]}×{v[1]}" for k, v in prep_info["shapes"].items()))
    add("Новая категория обрабатывается без ошибки", prep_info["unknown_category_ok"],
        "tariff_prev = 'Ultra' в копии validation -> нулевой one-hot, размер не изменился")
    add("Метки train наблюдаемы полностью",
        bool(labels.merge(split, on=KEYS).query("split == 'train'").label_observed.all()), "конец горизонта ≤ 01.10.2026")
    return pd.DataFrame(res)


# ---------- преобразование ----------

def fit_preprocessor(features, split):
    df = features.merge(split, on=KEYS, how="left", validate="one_to_one")
    parts = {s: df[df.split == s].reset_index(drop=True) for s in ["train", "validation", "test"]}
    prep = make_preprocessor(F.NUMERIC, F.CATEGORICAL)
    X = {"train": prep.fit_transform(parts["train"][F.FEATURES])}
    for s in ["validation", "test"]:
        X[s] = prep.transform(parts[s][F.FEATURES])
    names = prep.get_feature_names_out()
    unknown = parts["validation"][F.FEATURES].copy()
    unknown["tariff_prev"] = "Ultra"
    Xu = prep.transform(unknown)
    rows_preserved = all(X[s].shape[0] == len(parts[s]) for s in X)
    order = pd.concat([parts[s][KEYS].assign(split=s, matrix_row=np.arange(len(parts[s]))) for s in parts],
                      ignore_index=True)
    info = {
        "fit_rows": int(prep.named_transformers_["numeric"].named_steps["scale"].n_samples_seen_),
        "shapes": {s: list(X[s].shape) for s in X},
        "rows_preserved": bool(rows_preserved),
        "unknown_category_ok": bool(Xu.shape == X["validation"].shape and np.isfinite(Xu).all()),
        "all_finite": bool(all(np.isfinite(X[s]).all() for s in X)),
        "n_input_numeric": len(F.NUMERIC), "n_input_categorical": len(F.CATEGORICAL),
    }
    return prep, names, info, order


def summary_train(features, split):
    tr = features.merge(split, on=KEYS).query("split == 'train'")
    rows = []
    for c in F.FEATURES:
        v = tr[c]
        row = {"feature": c, "n_rows": len(v), "n_known": int(v.notna().sum()), "missing_share": round(v.isna().mean(), 5)}
        if c in F.NUMERIC:
            q = v.quantile([0.1, 0.5, 0.9])
            row.update(p10=q[0.1], median=q[0.5], p90=q[0.9], share_zero=round(float((v == 0).mean()), 5))
        else:
            row.update(categories="; ".join(f"{k}: {n}" for k, n in v.fillna("(пусто)").value_counts().items()))
        rows.append(row)
    return pd.DataFrame(rows)


def train_signal(features, labels, split):
    """Доли оттока по группам новых признаков — только train (обоснование, не отбор по validation/test)."""
    tr = features.merge(labels, on=KEYS).merge(split, on=KEYS).query("split == 'train'")
    groups = {
        "digital_status_prev": tr.digital_status_prev,
        "payment_failure_rate_3m": pd.cut(tr.payment_failure_rate_3m, [-0.01, 0, 0.34, 1.0],
                                          labels=["0", "(0; 0,34]", "> 0,34"]).astype(object).fillna("неизвестно"),
        "app_sessions_mean_3m": pd.cut(tr.app_sessions_mean_3m, [-0.01, 3, 8, 100],
                                       labels=["0–3", "3–8", "> 8"]).astype(object).fillna("неизвестно"),
        "debt_months_3m": tr.debt_months_3m.astype("Int64").astype(str),
        "unpaid_months_3m": tr.unpaid_months_3m.astype("Int64").astype(str),
        "traffic_change_1m_gb": pd.cut(tr.traffic_change_1m_gb, [-1e9, -10, 10, 1e9],
                                       labels=["снижение > 10 ГБ", "±10 ГБ", "рост > 10 ГБ"]).astype(object)
        .fillna("неизвестно"),
    }
    rows = []
    for name, g in groups.items():
        t = tr.groupby(g.values).agg(n_observations=("customer_id", "size"), n_customers=("customer_id", "nunique"),
                                     n_churn=("churn_target", "sum"))
        for value, r in t.iterrows():
            rows.append({"feature": name, "group": value, "n_observations": int(r.n_observations),
                         "n_customers": int(r.n_customers), "n_churn": int(r.n_churn),
                         "churn_rate": round(r.n_churn / r.n_observations, 5),
                         "small_group": r.n_observations < config.SMALL_GROUP})
    return pd.DataFrame(rows)


def run():
    inputs = F.load_inputs()
    events, subs, billing, support, digital = inputs
    features, labels, split, audit = F.build(*inputs)

    out = config.FEATURES_DATA
    write_csv(ser_features(features), out / "features.csv")
    write_csv(ser_labels(labels), out / "labels.csv")
    write_csv(split.assign(snapshot_at=fmt_date(split.snapshot_at)), out / "split.csv")
    write_csv(ser_audit(audit), out / "feature_audit.csv")

    rep = config.REPORTS_STAGE4
    write_csv(digital_checks(digital, subs, billing), rep / "digital_source_checks.csv")
    train_dates = list(config.TRAIN_DATES)
    write_csv(baseline.evaluate(features[features.snapshot_at.isin(train_dates)],
                                labels[labels.snapshot_at.isin(train_dates)], train_dates).round(5),
              rep / "baseline_train_by_date.csv")

    prep, names, info, order = fit_preprocessor(features, split)
    write_csv(pd.DataFrame({"x_column": names}), rep / "x_columns.csv")
    write_csv(order.assign(snapshot_at=fmt_date(order.snapshot_at)), rep / "x_row_order.csv")
    write_csv(summary_train(features, split), rep / "feature_summary_train.csv")
    write_csv(train_signal(features, labels, split), rep / "train_churn_by_feature_group.csv")

    checks = run_checks(features, labels, split, inputs, info)
    write_csv(checks, rep / "checks.csv")
    sizes = features.merge(split, on=KEYS).groupby("split").agg(rows=("customer_id", "size"),
                                                                customers=("customer_id", "nunique"))
    run_config = {
        "horizon": config.HORIZON, "active_mode": "before_events", "active_base_rule": config.ACTIVE_BASE_RULE,
        "window_months": config.WINDOW_MONTHS, "train_dates": [f"{d:%Y-%m-%d}" for d in config.TRAIN_DATES],
        "validation_dates": [f"{d:%Y-%m-%d}" for d in config.VALIDATION_DATES],
        "test_dates": [f"{d:%Y-%m-%d}" for d in config.TEST_DATES],
        "observation_end_exclusive": f"{config.EVENTS_OBSERVED_TO:%Y-%m-%d}", "campaign_fraction": config.CAMPAIGN_SHARE,
        "split_sizes": {s: {"rows": int(r.rows), "customers": int(r.customers)} for s, r in sizes.iterrows()},
        "x_input_columns": {"numeric": F.NUMERIC, "categorical": F.CATEGORICAL},
        "x_output_columns": len(names), "matrix_shapes": info["shapes"], "all_finite": info["all_finite"],
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    (rep / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_config, checks


def main():
    run_config, checks = run()
    print(json.dumps({k: run_config[k] for k in ["split_sizes", "matrix_shapes"]}, ensure_ascii=False))
    print(checks[["result", "check"]].to_string(index=False))
    if (checks.result != "OK").any():
        raise SystemExit("есть непройденные проверки")


if __name__ == "__main__":
    main()
