"""Агрегаты поддержки на ключ (customer_id, billing_month) и сборка customer_month."""
import numpy as np
import pandas as pd

from . import config

STATIC = ["region", "acquisition_channel", "customer_segment", "activation_date", "activation_date_source"]


def created_month(ts: pd.Series) -> pd.Series:
    """Календарный месяц создания по Москве (первый день месяца, без зоны)."""
    return ts.dt.tz_convert(config.TZ).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()


def aggregate_support(support: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Агрегаты для каждой строки keys (customer_id, billing_month, available_at).

    Учитываются обращения, созданные в billing_month по Москве и доступные не позднее
    available_at строки биллинга. Закрытие и оценка — только при resolution_available_at
    не позднее той же даты. Полные копии строк должны быть удалены заранее; на всякий
    случай счёт ведётся по уникальным ticket_id.
    """
    t = support.drop_duplicates("ticket_id").copy()
    t["billing_month"] = created_month(t.created_at)
    t = t.merge(keys[["customer_id", "billing_month", "available_at"]].rename(columns={"available_at": "row_available_at"}),
                on=["customer_id", "billing_month"], how="inner")
    t = t[t.available_at <= t.row_available_at]
    resolved = t.resolution_available_at.notna() & (t.resolution_available_at <= t.row_available_at)
    rated = resolved & t.satisfaction_score.notna()
    t = t.assign(
        is_network=t.topic.eq("network"), is_billing=t.topic.eq("billing"), is_unknown=t.topic.eq("unknown"),
        is_resolved=resolved, is_rated=rated, score_available=t.satisfaction_score.where(rated),
        is_tz_assumed=t.created_tz_assumed, is_resolution_invalid=t.resolution_invalid,
        is_score_invalid=t.satisfaction_score_invalid,
    )
    agg = t.groupby(["customer_id", "billing_month"]).agg(
        support_count=("ticket_id", "size"),
        support_network_count=("is_network", "sum"),
        support_billing_count=("is_billing", "sum"),
        support_unknown_topic_count=("is_unknown", "sum"),
        support_resolved_count=("is_resolved", "sum"),
        support_rating_count=("is_rated", "sum"),
        support_rating_mean=("score_available", "mean"),
        dq_support_tz_assumed_count=("is_tz_assumed", "sum"),
        dq_support_resolution_invalid_count=("is_resolution_invalid", "sum"),
        dq_support_score_invalid_count=("is_score_invalid", "sum"),
    ).reset_index()
    out = keys[["customer_id", "billing_month"]].merge(agg, on=["customer_id", "billing_month"], how="left")
    counts = [c for c in out.columns if c.endswith("_count")]
    # Нет доступных корректно связанных обращений -> 0; средняя оценка без оценок остаётся пропуском.
    out[counts] = out[counts].fillna(0).astype(int)
    return out


def quarantine_counts(quarantine: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Сколько обращений из карантина связано с абонентом и месяцем строки (флаг качества, не признак)."""
    q = quarantine[quarantine.linked_customer].copy()
    q["billing_month"] = created_month(q.created_at)
    agg = q.groupby(["customer_id", "billing_month"]).size().rename("dq_support_quarantined_count").reset_index()
    out = keys[["customer_id", "billing_month"]].merge(agg, on=["customer_id", "billing_month"], how="left")
    out["dq_support_quarantined_count"] = out.dq_support_quarantined_count.fillna(0).astype(int)
    return out


def build_customer_month(billing: pd.DataFrame, subscribers: pd.DataFrame, support: pd.DataFrame,
                         quarantine: pd.DataFrame) -> pd.DataFrame:
    keys = billing[["customer_id", "billing_month", "available_at"]]
    cm = billing.merge(subscribers[["customer_id"] + STATIC], on="customer_id", how="left", validate="many_to_one")
    cm = cm.merge(aggregate_support(support, keys), on=["customer_id", "billing_month"], how="left",
                  validate="one_to_one")
    cm = cm.merge(quarantine_counts(quarantine, keys), on=["customer_id", "billing_month"], how="left",
                  validate="one_to_one")
    if len(cm) != len(billing):
        raise AssertionError("объединение изменило число строк биллинга")
    for c in ["charged_rub", "paid_rub", "closing_debt_rub"]:
        if not np.isclose(cm[c].sum(), billing[c].sum()):
            raise AssertionError(f"объединение изменило сумму {c}")
    return cm
