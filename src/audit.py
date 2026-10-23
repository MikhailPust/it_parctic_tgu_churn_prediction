"""Проверки качества. Одни и те же проверки запускаются до и после очистки.

На вход подаются текстовые таблицы: сырые исходники или прочитанные обратно
очищенные CSV. Каждая проверка возвращает число строк с проблемой, число
ошибочных ячеек и число повторных копий (там, где единица применима).
"""
import numpy as np
import pandas as pd

from . import config
from .clean import parse_current_ts, parse_iso_ts, parse_legacy_ts, parse_month, parse_number

MONEY = ["monthly_fee_rub", "charged_rub", "paid_rub", "opening_debt_rub", "closing_debt_rub"]
NUMERIC = MONEY + ["traffic_gb", "voice_minutes", "active_days", "payment_delay_days", "tariff_changes_count"]


class Checks:
    def __init__(self, stage: str):
        self.stage = stage
        self.rows = []

    def add(self, source, check, field, rows=np.nan, cells=np.nan, copies=np.nan, total=np.nan):
        self.rows.append({"stage": self.stage, "source": source, "check": check, "field": field,
                          "rows_with_problem": rows, "problem_cells": cells, "duplicate_copies": copies,
                          "total_rows": total})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _contract(events: pd.DataFrame) -> pd.DataFrame:
    e = events.drop_duplicates()
    d = pd.to_datetime(e.event_date, format="%Y-%m-%d", errors="coerce")
    e = e.assign(d=d)
    act = e[e.event_type == "activation"].groupby("customer_id").d.min()
    ter = e[e.event_type == "termination"].groupby("customer_id").d.min()
    return pd.DataFrame({"activation": act, "termination": ter})


def audit_events(ch: Checks, events: pd.DataFrame, subscribers: pd.DataFrame):
    src, n = "contract_events", len(events)
    ch.add(src, "Всего строк", "*", total=n)
    ch.add(src, "Полные повторные строки", "*", copies=int(events.duplicated().sum()), total=n)
    ch.add(src, "Повтор ключа event_id", "event_id", rows=int(events.event_id.duplicated(keep=False).sum()), total=n)
    bad_date = pd.to_datetime(events.event_date, format="%Y-%m-%d", errors="coerce").isna()
    ch.add(src, "Дата не в формате YYYY-MM-DD", "event_date", cells=int(bad_date.sum()), total=n)
    ch.add(src, "Неизвестный тип события", "event_type",
           cells=int((~events.event_type.isin(["activation", "termination"])).sum()), total=n)
    ch.add(src, "Ссылка на неизвестного абонента", "customer_id",
           rows=int((~events.customer_id.str.strip().isin(subscribers.customer_id.str.strip())).sum()), total=n)
    cnt = events.drop_duplicates().pivot_table(index="customer_id", columns="event_type", values="event_id",
                                               aggfunc="count").reindex(columns=["activation", "termination"])
    ch.add(src, "Абоненты без ровно одного activation", "event_type", rows=int((cnt.activation.fillna(0) != 1).sum()))
    ch.add(src, "Абоненты с >1 termination", "event_type", rows=int((cnt.termination.fillna(0) > 1).sum()))
    c = _contract(events)
    ch.add(src, "termination не позже activation", "event_date",
           rows=int((c.termination.notna() & (c.termination <= c.activation)).sum()))


def audit_subscribers(ch: Checks, subs: pd.DataFrame, events: pd.DataFrame):
    src, n = "subscribers", len(subs)
    ch.add(src, "Всего строк", "*", total=n)
    ch.add(src, "Полные повторные строки", "*", copies=int(subs.duplicated().sum()), total=n)
    ch.add(src, "Повтор ключа customer_id", "customer_id", rows=int(subs.customer_id.duplicated(keep=False).sum()),
           total=n)
    missing = subs.activation_date.str.strip().eq("")
    ch.add(src, "Пропуск", "activation_date", cells=int(missing.sum()), total=n)
    parsed = pd.to_datetime(subs.activation_date, format="%Y-%m-%d", errors="coerce")
    ch.add(src, "Дата не в формате YYYY-MM-DD", "activation_date", cells=int((parsed.isna() & ~missing).sum()), total=n)
    act = subs.customer_id.map(_contract(events).activation)
    ch.add(src, "Расхождение с событием activation", "activation_date",
           cells=int((parsed.notna() & act.notna() & (parsed != act)).sum()), total=n)
    for f, allowed in [("region", config.REGIONS), ("acquisition_channel", ("online", "store", "partner")),
                       ("customer_segment", ("individual", "small_business"))]:
        ch.add(src, "Значение вне справочника", f, cells=int((~subs[f].isin(allowed)).sum()), total=n)


def audit_billing(ch: Checks, bill: pd.DataFrame, subs: pd.DataFrame, events: pd.DataFrame):
    """Проверки ячеек считаются после снятия полных копий, чтобы не учитывать одну ошибку дважды."""
    src, n = "billing_monthly", len(bill)
    ch.add(src, "Всего строк", "*", total=n)
    ch.add(src, "Полные повторные строки", "*", copies=int(bill.duplicated().sum()), total=n)
    b = bill.drop_duplicates().copy()
    b["month"] = parse_month(b.billing_month)
    ch.add(src, "Конфликт значений ключа (customer_id, billing_month)", "customer_id, billing_month",
           rows=int(b.duplicated(["customer_id", "month"], keep=False).sum()), total=n)
    ch.add(src, "Месяц не в формате YYYY-MM-DD", "billing_month",
           cells=int((~b.billing_month.str.match(r"^\d{4}-\d{2}-01$")).sum()), total=n)
    av = pd.to_datetime(b.available_at.str.replace("T", " ").str[:19], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ch.add(src, "available_at ≠ 1-е число следующего месяца", "available_at",
           rows=int((av != b.month + pd.offsets.MonthBegin(1)).sum()), total=n)
    ch.add(src, "Тариф вне Basic/Comfort/Premium", "tariff",
           cells=int((~b.tariff.isin(config.TARIFF_PRICE)).sum()), total=n)
    num = {}
    for c in NUMERIC:
        v = b[c]
        num[c] = parse_number(b[c])
        ch.add(src, "Десятичная запятая", c, cells=int(v.str.contains(",", regex=False).sum()), total=n)
        ch.add(src, "Пропуск", c, cells=int(v.str.strip().eq("").sum()), total=n)
    num = pd.DataFrame(num, index=b.index)
    ch.add(src, "Отрицательное значение", "voice_minutes", cells=int((num.voice_minutes < 0).sum()), total=n)
    neg = (num[MONEY + ["traffic_gb"]] < 0).sum().sum()
    ch.add(src, "Отрицательные суммы или трафик", ", ".join(MONEY + ["traffic_gb"]), cells=int(neg), total=n)
    dim = b.month.dt.days_in_month
    ch.add(src, "active_days вне [1, дней в месяце]", "active_days",
           cells=int(((num.active_days < 1) | (num.active_days > dim)).sum()), total=n)
    ch.add(src, "payment_delay_days вне [0, дней в месяце]", "payment_delay_days",
           cells=int(((num.payment_delay_days < 0) | (num.payment_delay_days > dim)).sum()), total=n)
    tariff = b.tariff.str.strip().str.lower().map(config.TARIFF_ALIASES)
    ch.add(src, "Цена не соответствует тарифу", "monthly_fee_rub",
           rows=int((tariff.map(config.TARIFF_PRICE) != num.monthly_fee_rub).sum()), total=n)
    eq = (num.opening_debt_rub + num.charged_rub - num.paid_rub - num.closing_debt_rub).round(2)
    ch.add(src, "Нарушено opening + charged − paid = closing", "closing_debt_rub",
           rows=int((eq.abs() > 0.01).sum()), total=n)
    order = b.assign(**num).sort_values(["customer_id", "month"])
    prev_close = order.groupby("customer_id").closing_debt_rub.shift()
    first = order.groupby("customer_id").cumcount() == 0
    ch.add(src, "opening_debt ≠ closing_debt предыдущего месяца", "opening_debt_rub",
           rows=int(((order.opening_debt_rub - prev_close).abs() > 0.01).sum()), total=n)
    ch.add(src, "Ненулевой долг в первой строке абонента", "opening_debt_rub",
           rows=int((first & (order.opening_debt_rub > 0)).sum()), total=n)
    ch.add(src, "Ссылка на неизвестного абонента", "customer_id",
           rows=int((~b.customer_id.str.strip().isin(subs.customer_id.str.strip())).sum()), total=n)
    c = _contract(events)
    a = b.customer_id.map(c.activation).dt.to_period("M").dt.to_timestamp()
    t = b.customer_id.map(c.termination)
    outside = (b.month < a) | (t.notna() & (b.month >= t))
    ch.add(src, "Месяц вне периода договора", "billing_month", rows=int(outside.sum()), total=n)


def support_view_raw(legacy: pd.DataFrame, current: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    mapping = dict(zip(schema.legacy_field, schema.canonical_field))
    leg = legacy.rename(columns=mapping)[list(schema.canonical_field)].assign(source_system="legacy")
    cur = current[list(schema.canonical_field)].assign(source_system="current")
    parts = []
    for df, parser in [(leg, parse_legacy_ts), (cur, parse_current_ts)]:
        df = df.copy()
        for c in ["created_at", "available_at", "resolved_at", "resolution_available_at"]:
            df["t_" + c] = parser(df[c])
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def support_view_clean(clean: pd.DataFrame) -> pd.DataFrame:
    df = clean.copy()
    for c in ["created_at", "available_at", "resolved_at", "resolution_available_at"]:
        df["t_" + c] = parse_iso_ts(df[c])
    return df


def audit_support(ch: Checks, view: pd.DataFrame, subs: pd.DataFrame, events: pd.DataFrame, raw_stage: bool):
    src = "support"
    cols = ["source_system", "ticket_id", "customer_id", "created_at", "available_at", "topic", "channel",
            "resolved_at", "resolution_available_at", "satisfaction_score"]
    n = len(view)
    ch.add(src, "Всего строк", "*", total=n)
    for system, part in view.groupby("source_system"):
        ch.add(f"{src}_{system}", "Всего строк", "*", total=len(part))
        ch.add(f"{src}_{system}", "Полные повторные строки", "*", copies=int(part[cols].duplicated().sum()),
               total=len(part))
    v = view.drop_duplicates(cols)
    ch.add(src, "Конфликт значений ticket_id", "ticket_id", rows=int(v.ticket_id.duplicated(keep=False).sum()), total=n)
    cid = v.customer_id
    ch.add(src, "Пустой customer_id", "customer_id", cells=int(cid.str.strip().eq("").sum()), total=n)
    ch.add(src, "Пробелы вокруг customer_id", "customer_id", cells=int((cid != cid.str.strip()).sum()), total=n)
    ch.add(src, "Ссылка на неизвестного абонента", "customer_id",
           rows=int((cid.str.strip().ne("") & ~cid.str.strip().isin(subs.customer_id.str.strip())).sum()), total=n)
    if raw_stage:
        no_zone = (v.source_system == "current") & ~v.created_at.str.strip().str.endswith("Z")
        ch.add(src, "Время без часового пояса", "created_at", cells=int(no_zone.sum()), total=n)
        not_msk = v.source_system == "current"
    else:
        not_msk = pd.Series(False, index=v.index)
        for c in ["created_at", "available_at", "resolved_at", "resolution_available_at"]:
            not_msk |= v[c].ne("") & ~v[c].str.endswith("+03:00")
    ch.add(src, "Время не в формате Москвы ISO 8601", "created_at, available_at, ...", rows=int(not_msk.sum()), total=n)
    ch.add(src, "Тема вне кодов network/billing/tariff/services/unknown", "topic",
           cells=int((~v.topic.isin(config.TOPICS)).sum()), total=n)
    ch.add(src, "Пустая тема", "topic", cells=int(v.topic.str.strip().eq("").sum()), total=n)
    ch.add(src, "Канал вне кодов phone/chat/office", "channel", cells=int((~v.channel.isin(config.CHANNELS)).sum()),
           total=n)
    created = v.t_created_at
    ch.add(src, "Создание вне [01.09.2025, 01.09.2026)", "created_at",
           rows=int(((created < config.SUPPORT_CREATED_FROM) | (created >= config.SUPPORT_CREATED_TO)).sum()), total=n)
    ch.add(src, "available_at раньше created_at", "available_at", rows=int((v.t_available_at < created).sum()), total=n)
    ch.add(src, "resolved_at раньше created_at", "resolved_at", rows=int((v.t_resolved_at < created).sum()), total=n)
    ch.add(src, "resolution_available_at раньше resolved_at или available_at", "resolution_available_at",
           rows=int(((v.t_resolution_available_at < v.t_resolved_at) |
                     (v.t_resolution_available_at < v.t_available_at)).sum()), total=n)
    ch.add(src, "Закрытие без момента доступности (или наоборот)", "resolved_at",
           rows=int((v.t_resolved_at.isna() != v.t_resolution_available_at.isna()).sum()), total=n)
    score = parse_number(v.satisfaction_score)
    ch.add(src, "Оценка вне 1–5", "satisfaction_score", cells=int((score.notna() & ~score.isin([1, 2, 3, 4, 5])).sum()),
           total=n)
    ch.add(src, "Оценка без подтверждённого закрытия", "satisfaction_score",
           cells=int((score.notna() & v.t_resolved_at.isna()).sum()), total=n)
    c = _contract(events)
    a = v.customer_id.str.strip().map(c.activation).dt.tz_localize(config.TZ)
    t = v.customer_id.str.strip().map(c.termination).dt.tz_localize(config.TZ)
    ch.add(src, "Создание вне периода договора", "created_at",
           rows=int(((created < a) | (t.notna() & (created >= t))).sum()), total=n)


def run_all(stage: str, subs, bill, events, support_view, raw_stage: bool) -> pd.DataFrame:
    ch = Checks(stage)
    audit_subscribers(ch, subs, events)
    audit_billing(ch, bill, subs, events)
    audit_events(ch, events, subs)
    audit_support(ch, support_view, subs, events, raw_stage)
    return ch.frame()
