"""Проверки этапа 3: граница T, конец горизонта, неизвестные метки и доступность сетевых отчётов."""
import pandas as pd

from src import config, eda, snapshots


def test_boundary_events_follow_charter_rule():
    b = eda.boundary_example()
    assert b.ok.all(), b.to_string()


def test_label_unknown_when_horizon_beyond_observation():
    contract = pd.DataFrame({"activation_date": [pd.Timestamp("2025-01-01")], "termination_date": [pd.NaT]},
                            index=["X"])
    sample = pd.DataFrame({"customer_id": ["X", "X"],
                           "snapshot_at": [pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-01")]})
    out = snapshots.add_target(sample, contract)
    assert out.churn_target.iloc[0] == 0          # [01.09, 01.10) наблюдаем полностью
    assert pd.isna(out.churn_target.iloc[1])      # конец 01.11 > 01.10.2026: метка неизвестна, а не 0
    out30 = snapshots.add_target(sample.iloc[:1], contract, horizon="30d")
    assert out30.target_end_exclusive.iloc[0] == pd.Timestamp("2026-10-01")


def test_late_network_report_unknown_at_snapshot():
    net = pd.DataFrame({
        "region": ["Москва", "Москва"],
        "measurement_month": [pd.Timestamp("2026-06-01")] * 2,
        "report_available_at": pd.to_datetime(["2026-07-01T00:00:00+03:00", "2026-07-02T00:00:00+03:00"],
                                              format="ISO8601").tz_convert(config.TZ),
        "probe_count": [100, 100], "failed_probe_count": [5, 5]})
    av = eda.network_availability(net, [pd.Timestamp("2026-07-01")])
    assert av.report_known_at_snapshot.tolist() == [True, False]
    assert av.failed_probe_rate_known.iloc[0] == 0.05 and pd.isna(av.failed_probe_rate_known.iloc[1])


def test_features_use_only_rows_available_at_T():
    contract = pd.DataFrame({"activation_date": [pd.Timestamp("2025-01-01")], "termination_date": [pd.NaT]},
                            index=["X"])
    subs = pd.DataFrame({"customer_id": ["X"], "region": ["Москва"], "acquisition_channel": ["online"],
                         "customer_segment": ["individual"], "activation_date": [pd.Timestamp("2025-01-01")]})
    months = pd.to_datetime(["2026-05-01", "2026-06-01", "2026-07-01"])
    cm = pd.DataFrame({"customer_id": "X", "billing_month": months,
                       "available_at": (months + pd.offsets.MonthBegin(1)).tz_localize(config.TZ)})
    for c in snapshots.FEATURES:
        cm[c] = 0
    cm["closing_debt_rub"] = [10.0, 20.0, 999.0]
    f = snapshots.build_features(contract, subs, cm, [pd.Timestamp("2026-07-01")])
    assert f.source_billing_month.item() == pd.Timestamp("2026-06-01")
    assert f.closing_debt_rub.item() == 20.0
    assert f.billing_months_available.item() == 2
