"""Исследовательские снимки «абонент × T»: активная база, признаки на T, метка отдельно.

Правило базы (устав): состояние непосредственно перед событиями дня T —
activation_date < T и (termination_date отсутствует или termination_date >= T).
"""
import numpy as np
import pandas as pd

from . import config


def contract_table(events: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на абонента из событий договора (даты — naive, Москва)."""
    e = events.assign(event_date=pd.to_datetime(events.event_date))
    piv = e.pivot_table(index="customer_id", columns="event_type", values="event_date", aggfunc="min")
    return piv.reindex(columns=["activation", "termination"]).rename(
        columns={"activation": "activation_date", "termination": "termination_date"})


def active_base(contract: pd.DataFrame, T: pd.Timestamp) -> pd.Index:
    act, ter = contract.activation_date, contract.termination_date
    mask = (act < T) & (ter.isna() | (ter >= T))
    return contract.index[mask]


def horizon_end(T: pd.Timestamp, horizon: str = config.HORIZON) -> pd.Timestamp:
    if horizon == "calendar_month":
        return T + pd.offsets.MonthBegin(1)
    if horizon == "30d":
        return T + pd.Timedelta(days=30)
    raise ValueError(horizon)


FEATURES = ["tariff", "monthly_fee_rub", "charged_rub", "paid_rub", "opening_debt_rub", "closing_debt_rub",
            "traffic_gb", "voice_minutes", "active_days", "payment_delay_days", "tariff_changes_count",
            "traffic_gb_missing", "voice_minutes_invalid", "support_count", "support_network_count",
            "support_billing_count", "support_unknown_topic_count"]
STATIC = ["region", "acquisition_channel", "customer_segment", "activation_date"]


def build_features(contract: pd.DataFrame, subscribers: pd.DataFrame, customer_month: pd.DataFrame,
                   dates) -> pd.DataFrame:
    """Признаки на T без метки: только строки customer_month с available_at <= T."""
    cm = customer_month
    parts = []
    for T in dates:
        T = pd.Timestamp(T)
        T_msk = T.tz_localize(config.TZ)
        base = pd.DataFrame({"customer_id": active_base(contract, T)})
        known = cm[cm.available_at <= T_msk]
        hist = known.groupby("customer_id").size().rename("billing_months_available")
        last = known.sort_values("billing_month").groupby("customer_id").tail(1)
        last = last[["customer_id", "billing_month", "available_at"] + FEATURES].rename(
            columns={"billing_month": "source_billing_month", "available_at": "billing_available_at"})
        df = base.merge(last, on="customer_id", how="left").merge(hist, on="customer_id", how="left")
        df.insert(1, "snapshot_at", T)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["billing_months_available"] = out.billing_months_available.fillna(0).astype(int)
    out = out.merge(subscribers[["customer_id"] + STATIC], on="customer_id", how="left", validate="many_to_one")
    out["tenure_days"] = (out.snapshot_at - out.activation_date).dt.days
    out["tenure_group"] = np.where(out.tenure_days < config.TENURE_THRESHOLD_DAYS, "<90 дней", "≥90 дней")
    out["debt_group"] = np.select([out.closing_debt_rub > 0, out.closing_debt_rub == 0], [">0", "=0"], "unknown")
    out["support_group"] = np.select(
        [out.support_count.isna(), out.support_count == 0, out.support_count == 1], ["unknown", "0", "1"], "2+")
    out["split"] = np.select([out.snapshot_at.isin(config.VALIDATION_DATES), out.snapshot_at.isin(config.TEST_DATES)],
                             ["validation", "test"], "eda")
    return out


def add_target(sample: pd.DataFrame, contract: pd.DataFrame, horizon: str = config.HORIZON,
               observed_to: pd.Timestamp = config.EVENTS_OBSERVED_TO) -> pd.DataFrame:
    """Отдельный шаг: метка по расторжению в [T, конец горизонта). Дата расторжения в выход не попадает."""
    out = sample.copy()
    ter = out.customer_id.map(contract.termination_date)
    end = out.snapshot_at.map(lambda T: horizon_end(T, horizon))
    observed = end <= observed_to
    churn = (ter.notna() & (ter >= out.snapshot_at) & (ter < end)).astype("Int64")
    out["target_end_exclusive"] = end
    out["label_observed"] = observed
    out["churn_target"] = churn.where(observed, pd.NA)
    return out
