"""Этап 4: признаки на дату расчёта T.

Каждый источник сначала ограничивается записями с available_at ≤ T, затем
окном из трёх предыдущих календарных месяцев [T − 3 мес., T). Поэтому
добавление или изменение будущих записей не меняет признаки прошлого снимка.
Метка строится отдельно (snapshots.add_target) и в эту таблицу не входит.
"""
import numpy as np
import pandas as pd

from . import config, snapshots
from .load import read_raw_csv

CATEGORICAL = ["region", "customer_segment", "acquisition_channel", "tariff_prev", "digital_status_prev"]
NUMERIC = [
    # договор и сегмент
    "tenure_days",
    # биллинг
    "debt_prev_rub", "debt_change_1m_rub", "debt_months_3m", "payment_delay_prev_days", "unpaid_months_3m",
    "paid_to_charged_3m",
    # использование связи
    "traffic_prev_gb", "traffic_mean_3m_gb", "traffic_change_1m_gb", "traffic_per_active_day_3m_gb",
    "voice_mean_3m_min",
    # поддержка (пересчёт на T)
    "support_count_3m", "support_network_3m", "support_billing_3m",
    # цифровой источник
    "app_sessions_prev", "payment_attempts_prev", "payment_failure_rate_prev", "app_sessions_per_day_prev",
    "app_sessions_mean_3m", "payment_failure_rate_3m", "app_sessions_last_observed", "digital_last_observed_age_m",
    # полнота
    "billing_history_months", "billing_months_known_3m", "traffic_known_months_3m", "voice_known_months_3m",
    "digital_observed_months_3m",
]
FEATURES = CATEGORICAL + NUMERIC
AUDIT = ["window_start", "window_end_exclusive", "prev_month", "billing_rows_used", "max_billing_available_at",
         "support_tickets_used", "max_support_available_at", "digital_rows_used", "max_digital_available_at",
         "prev_billing_known", "prev_digital_known"]


# ---------- загрузка ----------

def _num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c].replace("", np.nan))
    return df


def _ts(s):
    return pd.to_datetime(s.replace("", np.nan), format="ISO8601").dt.tz_convert(config.TZ)


def load_inputs():
    P = config.PREPARED
    events = read_raw_csv(P / "contract_events_clean.csv")
    subs = read_raw_csv(P / "subscribers_clean.csv")
    subs["activation_date"] = pd.to_datetime(subs.activation_date)
    billing = read_raw_csv(P / "billing_clean.csv")
    billing = _num(billing, ["charged_rub", "paid_rub", "closing_debt_rub", "traffic_gb", "voice_minutes",
                             "active_days", "payment_delay_days"])
    billing["billing_month"] = pd.to_datetime(billing.billing_month)
    billing["available_at"] = _ts(billing.available_at)
    support = read_raw_csv(P / "support_clean.csv")
    for c in ["created_at", "available_at"]:
        support[c] = _ts(support[c])
    digital = read_raw_csv(config.SOURCES["digital"])
    digital = _num(digital, ["observed_days", "app_sessions", "payment_attempts", "payment_failures",
                             "selfservice_actions"])
    digital["usage_month"] = pd.to_datetime(digital.usage_month)
    digital["available_at"] = _ts(digital.available_at)
    return events, subs, billing, support, digital


# ---------- признаки на одну дату ----------

def _months_back(month: pd.Series, T: pd.Timestamp) -> pd.Series:
    return (T.year * 12 + T.month) - (month.dt.year * 12 + month.dt.month)


def _ratio(num, den):
    """Отношение; нулевой или неизвестный знаменатель даёт пропуск, а не ноль."""
    den = den.where(den > 0)
    return num / den


def features_at(T, base_ids, subs, billing, support, digital):
    T = pd.Timestamp(T)
    T_msk = T.tz_localize(config.TZ)
    start = T - pd.DateOffset(months=config.WINDOW_MONTHS)
    out = pd.DataFrame(index=pd.Index(base_ids, name="customer_id"))

    # постоянные поля и стаж
    s = subs.set_index("customer_id").reindex(out.index)
    for c in ["region", "customer_segment", "acquisition_channel"]:
        out[c] = s[c]
    out["tenure_days"] = (T - s.activation_date).dt.days

    # биллинг: сначала доступность, потом окно
    b = billing[(billing.available_at <= T_msk) & billing.customer_id.isin(out.index)]
    out["billing_history_months"] = b.groupby("customer_id").size().reindex(out.index).fillna(0).astype(int)
    w = b[(b.billing_month >= start) & (b.billing_month < T)].copy()
    w["k"] = _months_back(w.billing_month, T)
    prev = w[w.k == 1].set_index("customer_id").reindex(out.index)
    prev2 = w[w.k == 2].set_index("customer_id").reindex(out.index)
    out["tariff_prev"] = prev.tariff
    out["debt_prev_rub"] = prev.closing_debt_rub
    out["debt_change_1m_rub"] = prev.closing_debt_rub - prev2.closing_debt_rub  # NaN, если месяц неизвестен
    out["payment_delay_prev_days"] = prev.payment_delay_days
    out["traffic_prev_gb"] = prev.traffic_gb
    out["traffic_change_1m_gb"] = prev.traffic_gb - prev2.traffic_gb
    w["has_debt"] = w.closing_debt_rub > 0
    w["unpaid"] = w.paid_rub == 0
    w["traffic_days"] = w.active_days.where(w.traffic_gb.notna())  # пары с известным трафиком
    g = w.groupby("customer_id")
    agg = pd.DataFrame({
        "billing_months_known_3m": g.size(),
        "debt_months_3m": g.has_debt.sum(),
        "unpaid_months_3m": g.unpaid.sum(),
        "paid_sum": g.paid_rub.sum(),
        "charged_sum": g.charged_rub.sum(),
        "traffic_known_months_3m": g.traffic_gb.count(),
        "traffic_mean_3m_gb": g.traffic_gb.mean(),
        "traffic_sum": g.traffic_gb.sum(min_count=1),
        "traffic_days_sum": g.traffic_days.sum(min_count=1),
        "voice_known_months_3m": g.voice_minutes.count(),
        "voice_mean_3m_min": g.voice_minutes.mean(),
        "max_billing_available_at": g.available_at.max(),
    }).reindex(out.index)
    for c in ["billing_months_known_3m", "debt_months_3m", "unpaid_months_3m", "traffic_known_months_3m",
              "voice_known_months_3m"]:
        agg[c] = agg[c].fillna(0).astype(int)
    agg.loc[agg.billing_months_known_3m == 0, ["debt_months_3m", "unpaid_months_3m"]] = np.nan
    out["paid_to_charged_3m"] = _ratio(agg.paid_sum, agg.charged_sum)
    out["traffic_per_active_day_3m_gb"] = _ratio(agg.traffic_sum, agg.traffic_days_sum)
    for c in ["billing_months_known_3m", "debt_months_3m", "unpaid_months_3m", "traffic_known_months_3m",
              "traffic_mean_3m_gb", "voice_known_months_3m", "voice_mean_3m_min"]:
        out[c] = agg[c]

    # поддержка: пересчёт на T — создано в окне и известно к T
    sp = support[(support.available_at <= T_msk) & (support.created_at >= start.tz_localize(config.TZ))
                 & (support.created_at < T_msk) & support.customer_id.isin(out.index)]
    sp = sp.assign(net=sp.topic.eq("network"), bill=sp.topic.eq("billing"))
    gs = sp.groupby("customer_id")
    out["support_count_3m"] = gs.size().reindex(out.index).fillna(0).astype(int)
    out["support_network_3m"] = gs.net.sum().reindex(out.index).fillna(0).astype(int)
    out["support_billing_3m"] = gs.bill.sum().reindex(out.index).fillna(0).astype(int)

    # цифровой источник: вся строка (и статус) известна только с available_at
    d = digital[(digital.available_at <= T_msk) & digital.customer_id.isin(out.index)
                & (digital.usage_month >= start) & (digital.usage_month < T)].copy()
    d["k"] = _months_back(d.usage_month, T)
    dp = d[d.k == 1].set_index("customer_id").reindex(out.index)
    out["digital_status_prev"] = dp.observation_status.astype(object).fillna("not_published_at_T")
    obs_prev = dp.observation_status.eq("observed")
    out["app_sessions_prev"] = dp.app_sessions.where(obs_prev)
    out["payment_attempts_prev"] = dp.payment_attempts.where(obs_prev)
    out["payment_failure_rate_prev"] = _ratio(dp.payment_failures, dp.payment_attempts).where(obs_prev)
    out["app_sessions_per_day_prev"] = _ratio(dp.app_sessions, dp.observed_days).where(obs_prev)
    o = d[d.observation_status == "observed"]
    go = o.groupby("customer_id")
    out["digital_observed_months_3m"] = go.size().reindex(out.index).fillna(0).astype(int)
    out["app_sessions_mean_3m"] = go.app_sessions.mean().reindex(out.index)
    out["payment_failure_rate_3m"] = _ratio(go.payment_failures.sum().reindex(out.index),
                                            go.payment_attempts.sum().reindex(out.index))
    last = o.sort_values("usage_month").groupby("customer_id").tail(1).set_index("customer_id").reindex(out.index)
    out["app_sessions_last_observed"] = last.app_sessions
    out["digital_last_observed_age_m"] = last.k

    audit = pd.DataFrame(index=out.index)
    audit["window_start"] = start
    audit["window_end_exclusive"] = T
    audit["prev_month"] = T - pd.DateOffset(months=1)
    audit["billing_rows_used"] = out.billing_months_known_3m
    audit["max_billing_available_at"] = agg.max_billing_available_at
    audit["support_tickets_used"] = out.support_count_3m
    audit["max_support_available_at"] = gs.available_at.max().reindex(out.index)
    audit["digital_rows_used"] = d.groupby("customer_id").size().reindex(out.index).fillna(0).astype(int)
    audit["max_digital_available_at"] = d.groupby("customer_id").available_at.max().reindex(out.index)
    audit["prev_billing_known"] = prev.available_at.notna()
    audit["prev_digital_known"] = dp.available_at.notna()

    out = out[FEATURES].reset_index()
    out.insert(1, "snapshot_at", T)
    audit = audit.reset_index()
    audit.insert(1, "snapshot_at", T)
    return out, audit


def build(events, subs, billing, support, digital, dates=config.ALL_DATES):
    contract = snapshots.contract_table(events)
    feats, audits, keys = [], [], []
    for T in dates:
        base = snapshots.active_base(contract, pd.Timestamp(T)).sort_values()
        f, a = features_at(T, base, subs, billing, support, digital)
        feats.append(f)
        audits.append(a)
    features = pd.concat(feats, ignore_index=True)
    audit = pd.concat(audits, ignore_index=True)
    keys = features[["customer_id", "snapshot_at"]]
    labels = snapshots.add_target(keys, contract)[["customer_id", "snapshot_at", "target_end_exclusive",
                                                   "label_observed", "churn_target"]]
    split = keys.assign(split=np.select([keys.snapshot_at.isin(config.TRAIN_DATES),
                                         keys.snapshot_at.isin(config.VALIDATION_DATES),
                                         keys.snapshot_at.isin(config.TEST_DATES)],
                                        ["train", "validation", "test"], "unused"))
    return features, labels, split, audit
