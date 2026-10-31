"""Проверки этапа 4 на маленьких искусственных примерах."""
import numpy as np
import pandas as pd
import pytest

from src import baseline, config, features as F
from src.preprocess import make_preprocessor

MSK = config.TZ
T = pd.Timestamp("2026-07-01")
T_MSK = T.tz_localize(MSK)


def subs(*ids):
    return pd.DataFrame({"customer_id": list(ids), "region": "Москва", "customer_segment": "individual",
                         "acquisition_channel": "online", "activation_date": pd.Timestamp("2025-01-01")})


def billing(rows):
    """rows: (customer_id, 'YYYY-MM', traffic, debt, available_at|None)."""
    out = []
    for cid, month, traffic, debt, av in rows:
        m = pd.Timestamp(month + "-01")
        out.append({"customer_id": cid, "billing_month": m, "tariff": "Basic", "charged_rub": 450.0, "paid_rub": 450.0,
                    "closing_debt_rub": debt, "traffic_gb": traffic, "voice_minutes": 100.0, "active_days": 30.0,
                    "payment_delay_days": 0.0,
                    "available_at": pd.Timestamp(av, tz=MSK) if av else (m + pd.offsets.MonthBegin(1)).tz_localize(MSK)})
    return pd.DataFrame(out)


def no_support():
    return pd.DataFrame({"customer_id": pd.Series(dtype=str), "ticket_id": pd.Series(dtype=str),
                         "topic": pd.Series(dtype=str),
                         "created_at": pd.Series(dtype=f"datetime64[ns, {MSK}]"),
                         "available_at": pd.Series(dtype=f"datetime64[ns, {MSK}]")})


def digital(rows):
    """rows: (customer_id, 'YYYY-MM', status, sessions, attempts, failures, available_at)."""
    out = []
    for cid, month, status, sess, att, fail, av in rows:
        obs = status == "observed"
        out.append({"customer_id": cid, "usage_month": pd.Timestamp(month + "-01"), "observation_status": status,
                    "observed_days": 30.0 if obs else np.nan, "app_sessions": sess if obs else np.nan,
                    "payment_attempts": att if obs else np.nan, "payment_failures": fail if obs else np.nan,
                    "selfservice_actions": 0.0 if obs else np.nan, "available_at": pd.Timestamp(av, tz=MSK)})
    return pd.DataFrame(out)


def feats(b, d=None, s=None, ids=("C1",)):
    d = d if d is not None else digital([])
    if d.empty:
        d = d.reindex(columns=["customer_id", "usage_month", "observation_status", "observed_days", "app_sessions",
                               "payment_attempts", "payment_failures", "selfservice_actions", "available_at"])
        d["usage_month"] = pd.to_datetime(d.usage_month)
        d["available_at"] = pd.to_datetime(d.available_at).dt.tz_localize(MSK)
    f, a = F.features_at(T, list(ids), subs(*ids), b, s if s is not None else no_support(), d)
    return f.set_index("customer_id"), a.set_index("customer_id")


def test_boundary_exact_T_included_one_second_later_excluded():
    b = billing([("C1", "2026-06", 10.0, 0.0, "2026-07-01 00:00:00"), ("C2", "2026-06", 10.0, 0.0, "2026-07-01 00:00:01")])
    f, _ = feats(b, ids=("C1", "C2"))
    assert f.loc["C1", "traffic_prev_gb"] == 10.0
    assert np.isnan(f.loc["C2", "traffic_prev_gb"])


def test_three_month_window_and_change():
    b = billing([("C1", "2026-03", 99.0, 0.0, None), ("C1", "2026-04", 10.0, 100.0, None),
                 ("C1", "2026-05", 20.0, 0.0, None), ("C1", "2026-06", 30.0, 50.0, None)])
    f, _ = feats(b)
    r = f.loc["C1"]
    assert r.traffic_mean_3m_gb == 20.0  # апрель–июнь, март вне окна
    assert r.traffic_change_1m_gb == 10.0  # июнь − май
    assert r.debt_months_3m == 2 and r.billing_months_known_3m == 3
    assert r.traffic_per_active_day_3m_gb == pytest.approx(60 / 90)


def test_short_history_is_not_padded_with_zeros():
    f, _ = feats(billing([("C1", "2026-06", 12.0, 0.0, None)]))
    r = f.loc["C1"]
    assert r.traffic_mean_3m_gb == 12.0 and r.billing_months_known_3m == 1
    assert np.isnan(r.traffic_change_1m_gb) and np.isnan(r.debt_change_1m_rub)


def test_missing_month_is_not_replaced_by_older_one():
    # май отсутствует: разность июнь − май неизвестна, апрель её не подменяет
    f, _ = feats(billing([("C1", "2026-04", 5.0, 0.0, None), ("C1", "2026-06", 30.0, 0.0, None)]))
    assert np.isnan(f.loc["C1", "traffic_change_1m_gb"])
    # неизвестный трафик месяца не входит в знаменатель «трафика на активный день»
    f2, _ = feats(billing([("C1", "2026-05", np.nan, 0.0, None), ("C1", "2026-06", 30.0, 0.0, None)]))
    assert f2.loc["C1", "traffic_per_active_day_3m_gb"] == 1.0


def test_future_rows_do_not_change_features():
    base = billing([("C1", "2026-05", 20.0, 0.0, None), ("C1", "2026-06", 30.0, 0.0, None)])
    future = pd.concat([base, billing([("C1", "2026-07", 999.0, 999.0, None)])], ignore_index=True)
    assert feats(base)[0].equals(feats(future)[0])


def test_digital_states_and_zero_attempts():
    b = billing([(c, "2026-06", 10.0, 0.0, None) for c in ["A", "B", "C", "D", "E"]])
    d = digital([
        ("A", "2026-06", "observed", 0.0, 0.0, 0.0, "2026-07-01 00:00"),     # измеренный ноль, попыток нет
        ("B", "2026-06", "not_enabled", None, None, None, "2026-07-01 00:00"),
        ("C", "2026-06", "export_gap", None, None, None, "2026-07-01 00:00"),
        ("D", "2026-06", "observed", 5.0, 2.0, 1.0, "2026-07-01 06:00"),     # опубликован после T
        ("D", "2026-05", "observed", 4.0, 2.0, 0.0, "2026-06-01 00:00"),
        ("E", "2026-06", "observed", 6.0, 4.0, 1.0, "2026-07-01 00:00"),
    ])
    f, _ = feats(b, d, ids=("A", "B", "C", "D", "E"))
    assert f.loc["A", "app_sessions_prev"] == 0 and np.isnan(f.loc["A", "payment_failure_rate_prev"])
    for cid, status in [("B", "not_enabled"), ("C", "export_gap"), ("D", "not_published_at_T")]:
        assert f.loc[cid, "digital_status_prev"] == status
        assert np.isnan(f.loc[cid, "app_sessions_prev"])
    assert f.loc["D", "app_sessions_last_observed"] == 4 and f.loc["D", "digital_last_observed_age_m"] == 2
    assert f.loc["E", "payment_failure_rate_prev"] == 0.25


def test_support_recomputed_at_T_with_availability():
    s = no_support()
    rows = [("T1", "2026-04-10 10:00", "2026-04-11 10:00", "network"),   # в окне
            ("T2", "2026-06-30 23:00", "2026-07-01 00:00", "billing"),   # ровно на границе — включается
            ("T3", "2026-06-30 23:30", "2026-07-01 09:00", "network"),   # известно после T
            ("T4", "2026-03-31 12:00", "2026-04-01 00:00", "network")]   # создано до окна
    s = pd.DataFrame([{"customer_id": "C1", "ticket_id": t, "topic": topic, "created_at": pd.Timestamp(c, tz=MSK),
                       "available_at": pd.Timestamp(a, tz=MSK)} for t, c, a, topic in rows])
    f, _ = feats(billing([("C1", "2026-06", 1.0, 0.0, None)]), s=s)
    assert f.loc["C1", "support_count_3m"] == 2
    assert f.loc["C1", "support_network_3m"] == 1 and f.loc["C1", "support_billing_3m"] == 1


def test_baseline_order_ties_and_unknown_debt():
    df = pd.DataFrame({"customer_id": ["C3", "C1", "C2", "C4", "C5"], "debt_prev_rub": [100, 100, 0, np.nan, 500]})
    order = baseline.rank_by_debt(df).customer_id.tolist()
    assert order == ["C5", "C1", "C3", "C2", "C4"]


def test_preprocessor_new_category_and_forbidden_columns():
    tr = pd.DataFrame({"x": [1.0, np.nan, 3.0], "cat": ["a", "b", None]})
    va = pd.DataFrame({"x": [np.nan, 10.0], "cat": ["zzz", "a"]})
    prep = make_preprocessor(["x"], ["cat"])
    Xt = prep.fit_transform(tr)
    Xv = prep.transform(va)
    assert Xv.shape == (2, Xt.shape[1]) and np.isfinite(Xv).all()
    assert prep.named_transformers_["numeric"].named_steps["impute"].statistics_[0] == 2.0  # медиана train
    with pytest.raises(ValueError):
        make_preprocessor(["x", "churn_target"], ["cat"])
