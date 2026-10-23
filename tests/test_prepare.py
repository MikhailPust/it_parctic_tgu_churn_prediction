"""Проверки этапа 2: правила агрегации и доступности на маленьких примерах и на итоговых таблицах."""
import numpy as np
import pandas as pd
import pytest

from src import clean, config, merge
from src.load import read_raw_csv

MSK = config.TZ


def ts(s):
    return pd.Timestamp(s, tz=MSK)


def billing_rows(*rows):
    df = pd.DataFrame(rows, columns=["customer_id", "billing_month"])
    df["billing_month"] = pd.to_datetime(df.billing_month)
    df["available_at"] = (df.billing_month + pd.offsets.MonthBegin(1)).dt.tz_localize(MSK)
    for c in ["charged_rub", "paid_rub", "closing_debt_rub"]:
        df[c] = 100.0
    return df


def ticket(tid, cid, created, available, topic="network", resolved=None, res_av=None, score=np.nan):
    return {"ticket_id": tid, "customer_id": cid, "created_at": ts(created), "available_at": ts(available),
            "topic": topic, "channel": "chat",
            "resolved_at": ts(resolved) if resolved else pd.NaT, "resolution_available_at": ts(res_av) if res_av else pd.NaT,
            "satisfaction_score": score, "created_tz_assumed": False, "resolution_invalid": False,
            "satisfaction_score_invalid": False}


def frame(*tickets):
    df = pd.DataFrame(list(tickets))
    for c in clean.SUPPORT_TIME:
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert(MSK)
    return df


def test_duplicate_ticket_row_does_not_double_count():
    b = billing_rows(("C1", "2026-03-01"))
    t = ticket("T1", "C1", "2026-03-10 10:00", "2026-03-11 10:00")
    agg = merge.aggregate_support(frame(t, t), b)
    assert agg.support_count.tolist() == [1]
    assert agg.support_network_count.tolist() == [1]


def test_late_ticket_is_not_in_month_snapshot():
    # создано 31 августа 23:00, загружено 1 сентября 09:00: на 01.09 00:00 ещё неизвестно
    b = billing_rows(("C1", "2026-08-01"))
    agg = merge.aggregate_support(frame(ticket("T1", "C1", "2026-08-31 23:00", "2026-09-01 09:00")), b)
    assert agg.support_count.tolist() == [0]


def test_ticket_available_exactly_at_row_date_is_included():
    b = billing_rows(("C1", "2026-08-01"))
    agg = merge.aggregate_support(frame(ticket("T1", "C1", "2026-08-31 23:00", "2026-09-01 00:00")), b)
    assert agg.support_count.tolist() == [1]


def test_resolution_and_score_only_when_available():
    b = billing_rows(("C1", "2026-03-01"))
    early = ticket("T1", "C1", "2026-03-02", "2026-03-02 12:00", resolved="2026-03-05", res_av="2026-03-06", score=4)
    late = ticket("T2", "C1", "2026-03-20", "2026-03-21", resolved="2026-03-30", res_av="2026-04-02", score=1)
    agg = merge.aggregate_support(frame(early, late), b)
    assert agg.support_count.item() == 2
    assert agg.support_resolved_count.item() == 1
    assert agg.support_rating_mean.item() == 4


def test_customer_without_tickets_is_kept_with_zero_counts():
    b = billing_rows(("C1", "2026-03-01"), ("C2", "2026-03-01"))
    agg = merge.aggregate_support(frame(ticket("T1", "C1", "2026-03-10", "2026-03-10 12:00", topic="unknown")), b)
    c2 = agg.set_index("customer_id").loc["C2"]
    assert len(agg) == 2
    assert c2.support_count == 0 and c2.support_unknown_topic_count == 0
    assert np.isnan(c2.support_rating_mean)  # нет оценок — пропуск, а не ноль


def test_current_timestamps_utc_and_missing_zone():
    s = pd.Series(["2026-08-31T20:00:00Z", "2026-08-31T23:00:00", ""])
    out = clean.parse_current_ts(s)
    assert out[0] == ts("2026-08-31 23:00") and out[1] == ts("2026-08-31 23:00") and pd.isna(out[2])


# ---------- проверки на результатах полного запуска ----------

prepared = pytest.mark.skipif(not (config.PREPARED / "customer_month.csv").exists(),
                              reason="сначала запустите python -m src.prepare")


@pytest.fixture(scope="module")
def tables():
    return {n: read_raw_csv(config.PREPARED / f"{n}.csv") for n in
            ["subscribers_clean", "billing_clean", "contract_events_clean", "support_clean", "customer_month",
             "quarantine_support"]}


@prepared
def test_customer_month_preserves_billing(tables):
    cm, b = tables["customer_month"], tables["billing_clean"]
    assert len(cm) == len(b)
    assert not cm.duplicated(["customer_id", "billing_month"]).any()
    for c in ["charged_rub", "paid_rub", "closing_debt_rub"]:
        assert pd.to_numeric(cm[c]).sum() == pytest.approx(pd.to_numeric(b[c]).sum())


@prepared
def test_keys_unique_and_subscribers_kept(tables):
    assert tables["subscribers_clean"].customer_id.is_unique
    assert len(tables["subscribers_clean"]) == len(read_raw_csv(config.SOURCES["subscribers"]))
    assert tables["support_clean"].ticket_id.is_unique
    assert tables["contract_events_clean"].event_id.is_unique


@prepared
def test_support_counts_recomputed_independently(tables):
    s, cm = tables["support_clean"], tables["customer_month"]
    created = pd.to_datetime(s.created_at, format="ISO8601", utc=True).dt.tz_convert(MSK)
    avail = pd.to_datetime(s.available_at, format="ISO8601", utc=True).dt.tz_convert(MSK)
    month = created.dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    row_av = (month + pd.offsets.MonthBegin(1)).dt.tz_localize(MSK)
    ok = s[avail <= row_av].assign(billing_month=month.dt.strftime("%Y-%m-%d"))
    expected = ok.groupby(["customer_id", "billing_month"]).size()
    got = cm.set_index(["customer_id", "billing_month"]).support_count.astype(int)
    assert got[got > 0].sort_index().equals(expected.sort_index().astype(int))


@prepared
def test_quarantine_has_reasons(tables):
    q = tables["quarantine_support"]
    assert q.reason.ne("").all()
    assert not q.ticket_id.isin(tables["support_clean"].ticket_id).any()
