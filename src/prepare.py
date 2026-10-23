"""Этап 2: единый запуск подготовки данных.

    python -m src.prepare                  # основной путь: SQL-запрос к SQLite
    python -m src.prepare --support-source csv   # резервное чтение CSV-копии

Вход: data/raw (не изменяется). Выход: data/prepared, data/interim, reports/stage2.
"""
import argparse
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd

from . import audit, clean, config, merge
from .load import load_sources, read_raw_csv, write_csv

CHECKSUMS = config.RAW / "checksums.sha256"


def raw_checksums() -> dict[str, str]:
    files = sorted(p for p in config.RAW.rglob("*") if p.is_file() and p != CHECKSUMS)
    return {p.relative_to(config.RAW).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def verify_raw_unchanged() -> str:
    current = raw_checksums()
    if not CHECKSUMS.exists():
        CHECKSUMS.write_text("".join(f"{h}  {f}\n" for f, h in current.items()), encoding="utf-8")
        return "created"
    saved = dict(reversed(line.split("  ", 1)) for line in CHECKSUMS.read_text(encoding="utf-8").splitlines())
    if saved != current:
        changed = sorted(set(saved.items()) ^ set(current.items()))
        raise RuntimeError(f"исходные файлы изменены: {changed[:5]}")
    return "verified"


# ---------- сериализация ----------

def ser_subscribers(df):
    out = df[["customer_id", "activation_date", "activation_date_source", "region", "acquisition_channel",
              "customer_segment"]].copy()
    out["activation_date"] = clean.fmt_date(out.activation_date)
    return out


def ser_events(df):
    return df.assign(event_date=clean.fmt_date(df.event_date))[["event_id", "customer_id", "event_date", "event_type"]]


BILLING_COLS = ["customer_id", "billing_month", "available_at", "tariff"] + clean.MONEY + ["traffic_gb"] + clean.INTS
BILLING_FLAGS = ["closing_debt_restored", "traffic_gb_missing", "voice_minutes_invalid"]


def ser_billing_like(df):
    out = df.copy()
    out["billing_month"] = clean.fmt_date(out.billing_month)
    out["available_at"] = clean.fmt_ts(out.available_at)
    for c in clean.MONEY + ["traffic_gb"]:
        out[c] = clean.fmt_money(out[c])
    for c in clean.INTS:
        out[c] = clean.fmt_int(out[c])
    for c in BILLING_FLAGS:
        out[c] = clean.fmt_bool(out[c])
    if "activation_date" in out:
        out["activation_date"] = clean.fmt_date(out.activation_date)
    if "support_rating_mean" in out:
        out["support_rating_mean"] = out.support_rating_mean.map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return out


SUPPORT_COLS = ["ticket_id", "customer_id", "source_system", "created_at", "available_at", "topic", "channel",
                "resolved_at", "resolution_available_at", "satisfaction_score", "created_tz_assumed",
                "resolution_invalid", "satisfaction_score_invalid"]


def ser_support(df):
    out = df.copy()
    for c in clean.SUPPORT_TIME:
        out[c] = clean.fmt_ts(out[c])
    out["satisfaction_score"] = clean.fmt_int(out.satisfaction_score)
    for c in ["created_tz_assumed", "resolution_invalid", "satisfaction_score_invalid"]:
        out[c] = clean.fmt_bool(out[c])
    return out[SUPPORT_COLS]


def ser_quarantine(df):
    out = df.copy()
    for c in clean.SUPPORT_TIME:
        out[c] = clean.fmt_ts(out[c])
    out["created_month"] = clean.fmt_date(out.created_month)
    out["linked_customer"] = clean.fmt_bool(out.linked_customer)
    out["satisfaction_score"] = clean.fmt_int(out.satisfaction_score)
    return out[["source_system", "ticket_id", "raw_customer_id", "customer_id", "raw_created_at", "created_at",
                "available_at", "topic", "channel", "resolved_at", "resolution_available_at", "satisfaction_score",
                "reason", "linked_customer", "created_month"]]


# ---------- запуск ----------

def run(support_source: str = "sqlite") -> dict:
    raw_status = verify_raw_unchanged()
    raw, meta = load_sources(support_source)
    schema = read_raw_csv(config.SOURCES["support_schema"])

    before = audit.run_all("before", raw["subscribers"], raw["billing"], raw["events"],
                           audit.support_view_raw(raw["support_legacy"], raw["support_current"], schema), raw_stage=True)

    log = clean.DecisionLog()
    events = clean.clean_events(raw["events"], log)
    subscribers = clean.clean_subscribers(raw["subscribers"], events, log)
    billing = clean.clean_billing(raw["billing"], log)
    support, quarantine = clean.clean_support(raw["support_legacy"], raw["support_current"], schema, subscribers, log)
    cm = merge.build_customer_month(billing, subscribers, support, quarantine)

    out = config.PREPARED
    write_csv(ser_subscribers(subscribers), out / "subscribers_clean.csv")
    write_csv(ser_events(events), out / "contract_events_clean.csv")
    write_csv(ser_billing_like(billing)[BILLING_COLS + BILLING_FLAGS], out / "billing_clean.csv")
    write_csv(ser_support(support), out / "support_clean.csv")
    write_csv(ser_billing_like(cm), out / "customer_month.csv")
    write_csv(ser_quarantine(quarantine), out / "quarantine_support.csv")

    # Повторные проверки на прочитанных обратно очищенных таблицах
    rb = {name: read_raw_csv(out / f"{name}.csv") for name in
          ["subscribers_clean", "billing_clean", "contract_events_clean", "support_clean", "customer_month"]}
    after = audit.run_all("after", rb["subscribers_clean"], rb["billing_clean"][BILLING_COLS],
                          rb["contract_events_clean"], audit.support_view_clean(rb["support_clean"]), raw_stage=False)

    checks = before.merge(after, on=["source", "check", "field"], how="left", suffixes=("_before", "_after"),
                          sort=False).drop(columns=["stage_before", "stage_after"])
    rep = config.REPORTS_STAGE2
    write_csv(checks, rep / "quality_checks.csv")
    write_csv(log.frame(), rep / "decision_log.csv")

    summary = build_summary(raw, meta, subscribers, billing, events, support, quarantine, cm, rb, raw_status)
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_summary(raw, meta, subscribers, billing, events, support, quarantine, cm, rb, raw_status):
    t = support.copy()
    t["billing_month"] = merge.created_month(t.created_at)
    keys = billing[["customer_id", "billing_month", "available_at"]].rename(columns={"available_at": "row_av"})
    t = t.merge(keys, on=["customer_id", "billing_month"], how="left")
    late = int((t.row_av.notna() & (t.available_at > t.row_av)).sum())
    no_billing_row = int(t.row_av.isna().sum())
    return {
        "run": {
            "python": sys.version.split()[0], "platform": platform.platform(), "pandas": pd.__version__,
            "numpy": np.__version__, "support_current_source": meta["support_current_source"],
            "raw_checksums": raw_status,
        },
        "sqlite_diagnostics": meta.get("sqlite_diagnostics"),
        "rows_before": {k: len(v) for k, v in raw.items()},
        "rows_after": {k: len(v) for k, v in rb.items()} | {"quarantine_support": len(quarantine)},
        "unique": {
            "subscribers": int(subscribers.customer_id.nunique()),
            "billing_customers": int(billing.customer_id.nunique()),
            "support_tickets": int(support.ticket_id.nunique()),
        },
        "quarantine_by_reason": quarantine.reason.value_counts().to_dict(),
        "quarantine_linked_to_customer": int(quarantine.linked_customer.sum()),
        "customer_month": {
            "rows": len(cm), "billing_rows": len(billing),
            "key_unique": bool(not cm.duplicated(["customer_id", "billing_month"]).any()),
            "sum_charged_rub": round(float(cm.charged_rub.sum()), 2),
            "sum_paid_rub": round(float(cm.paid_rub.sum()), 2),
            "sum_closing_debt_rub": round(float(cm.closing_debt_rub.sum()), 2),
            "rows_with_zero_support": int((cm.support_count == 0).sum()),
            "share_rows_with_zero_support": round(float((cm.support_count == 0).mean()), 4),
            "customers_without_any_available_support": int(
                (cm.groupby("customer_id").support_count.sum() == 0).sum()),
            "support_tickets_counted": int(cm.support_count.sum()),
            "tickets_excluded_late_availability": late,
            "tickets_without_billing_row": no_billing_row,
            "rows_with_quarantined_linked_ticket": int((cm.dq_support_quarantined_count > 0).sum()),
            "traffic_missing_cells": int(cm.traffic_gb.isna().sum()),
            "voice_missing_cells": int(cm.voice_minutes.isna().sum()),
        },
        "subscribers_activation_source": subscribers.activation_date_source.value_counts().to_dict(),
        "subscribers_without_billing": int((~subscribers.customer_id.isin(billing.customer_id)).sum()),
        "events": events.event_type.value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--support-source", choices=["sqlite", "csv"], default="sqlite")
    args = parser.parse_args()
    summary = run(args.support_source)
    print(json.dumps({k: summary[k] for k in ["rows_before", "rows_after", "quarantine_by_reason", "customer_month"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
