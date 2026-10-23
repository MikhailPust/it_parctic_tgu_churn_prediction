"""Приведение форматов и очистка. Исходные таблицы не изменяются: функции возвращают копии."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

MONEY = ["monthly_fee_rub", "charged_rub", "paid_rub", "opening_debt_rub", "closing_debt_rub"]
INTS = ["voice_minutes", "active_days", "payment_delay_days", "tariff_changes_count"]
SUPPORT_TIME = ["created_at", "available_at", "resolved_at", "resolution_available_at"]


@dataclass
class DecisionLog:
    rows: list = field(default_factory=list)

    def add(self, source, field_, problem, action, justification, count, unit):
        self.rows.append({
            "source": source, "field": field_, "problem": problem, "action": action,
            "justification": justification, "affected": int(count), "unit": unit,
        })

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ---------- разбор значений ----------

def parse_number(s: pd.Series) -> pd.Series:
    """Десятичная точка или запятая; пустая строка — неизвестное значение (NaN), не ноль."""
    return pd.to_numeric(s.str.strip().str.replace(",", ".", regex=False).replace("", np.nan), errors="coerce")


def parse_month(s: pd.Series) -> pd.Series:
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    ru = pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")
    return iso.fillna(ru)


def parse_current_ts(s: pd.Series) -> pd.Series:
    """Новая система: суффикс Z — UTC; без зоны — Москва (подтверждено заказчиком)."""
    s = s.str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype=f"datetime64[ns, {config.TZ}]")
    utc = s.str.endswith("Z")
    local = ~utc & s.ne("")
    out[utc] = pd.to_datetime(s[utc], format="%Y-%m-%dT%H:%M:%SZ", utc=True).dt.tz_convert(config.TZ)
    out[local] = pd.to_datetime(s[local], format="%Y-%m-%dT%H:%M:%S").dt.tz_localize(config.TZ)
    return out


def parse_legacy_ts(s: pd.Series) -> pd.Series:
    """Старая система: DD.MM.YYYY HH:MM:SS по Москве."""
    return pd.to_datetime(s.str.strip().replace("", np.nan), format="%d.%m.%Y %H:%M:%S").dt.tz_localize(config.TZ)


def parse_iso_ts(s: pd.Series) -> pd.Series:
    """Очищенный формат: ISO 8601 со смещением +03:00."""
    return pd.to_datetime(s.replace("", np.nan), format="ISO8601").dt.tz_convert(config.TZ)


def fmt_ts(s: pd.Series) -> pd.Series:
    return s.map(lambda x: x.isoformat() if pd.notna(x) else "")


def fmt_date(s: pd.Series) -> pd.Series:
    return s.dt.strftime("%Y-%m-%d").fillna("")


def fmt_money(s: pd.Series) -> pd.Series:
    return s.map(lambda x: f"{x:.2f}" if pd.notna(x) else "")


def fmt_int(s: pd.Series) -> pd.Series:
    return s.map(lambda x: str(int(x)) if pd.notna(x) else "")


def fmt_bool(s: pd.Series) -> pd.Series:
    return s.map({True: "1", False: "0"})


# ---------- события договора ----------

def clean_events(raw: pd.DataFrame, log: DecisionLog) -> pd.DataFrame:
    df = raw.copy()
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log.add("contract_events", "*", "Полные повторные строки", "Проверено, удалять нечего",
            "Журнал полон и уникален по event_id", n_dup, "повторные копии")
    df["event_date"] = pd.to_datetime(df.event_date, format="%Y-%m-%d")
    return df.sort_values(["customer_id", "event_date", "event_type"]).reset_index(drop=True)


def contract_dates(events: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на абонента: дата подключения и (если есть) расторжения."""
    piv = events.pivot_table(index="customer_id", columns="event_type", values="event_date", aggfunc="min")
    return piv.reindex(columns=["activation", "termination"]).rename(
        columns={"activation": "activation_date", "termination": "termination_date"})


# ---------- абоненты ----------

def clean_subscribers(raw: pd.DataFrame, events: pd.DataFrame, log: DecisionLog) -> pd.DataFrame:
    df = raw.copy()
    df["customer_id"] = df.customer_id.str.strip()
    given = pd.to_datetime(df.activation_date.replace("", np.nan), format="%Y-%m-%d")
    dates = contract_dates(events)
    act_events = events[events.event_type == "activation"].groupby("customer_id").event_date
    ev_date = df.customer_id.map(act_events.min())
    ev_unique = df.customer_id.map(act_events.nunique()).eq(1)

    missing = given.isna()
    restorable = missing & ev_date.notna() & ev_unique
    df["activation_date"] = given.where(~restorable, ev_date)
    df["activation_date_source"] = np.select(
        [~missing, restorable], ["subscribers", "contract_events"], default="missing")
    mismatch = ~missing & ev_date.notna() & (given != ev_date)

    log.add("subscribers", "activation_date", "Пропуск даты подключения",
            "Восстановлено по единственному событию activation", "Событие activation однозначно подтверждает дату; "
            "абонентов удалять нельзя", restorable.sum(), "ячейки")
    log.add("subscribers", "activation_date", "Пропуск без возможности восстановления", "Оставлен пропуск",
            "Нет однозначного события activation", (missing & ~restorable).sum(), "ячейки")
    log.add("subscribers", "activation_date", "Расхождение с событием activation", "Проверено",
            "Заполненные даты совпадают с журналом", mismatch.sum(), "ячейки")
    assert dates.index.is_unique
    return df


# ---------- биллинг ----------

def clean_billing(raw: pd.DataFrame, log: DecisionLog) -> pd.DataFrame:
    df = raw.copy()
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.add("billing_monthly", "*", "Полные повторные строки", "Удалены повторные копии, оставлена одна",
            "Копии полностью совпадают и удваивали бы суммы", n_before - len(df), "повторные копии")

    df["customer_id"] = df.customer_id.str.strip()
    month = parse_month(df.billing_month)
    log.add("billing_monthly", "billing_month", "Формат DD.MM.YYYY вместо YYYY-MM-DD",
            "Приведено к YYYY-MM-DD", "Оба формата однозначно задают первый день месяца",
            (month.notna() & ~df.billing_month.str.match(r"^\d{4}-\d{2}-\d{2}$")).sum(), "ячейки")
    df["billing_month"] = month

    conflicts = df.duplicated(["customer_id", "billing_month"], keep=False)
    if conflicts.any():
        raise ValueError(f"конфликт значений ключа биллинга: {int(conflicts.sum())} строк, нужна ручная проверка")

    available = pd.to_datetime(df.available_at, format="%Y-%m-%d %H:%M:%S").dt.tz_localize(config.TZ)
    expected = (df.billing_month + pd.offsets.MonthBegin(1)).dt.tz_localize(config.TZ)
    if (available != expected).any():
        raise ValueError("available_at не совпадает с первым днём следующего месяца")
    df["available_at"] = available

    raw_tariff = df.tariff
    df["tariff"] = raw_tariff.str.strip().str.lower().map(config.TARIFF_ALIASES)
    log.add("billing_monthly", "tariff", "Варианты написания (COMFORT, ' basic ', Премиум)",
            "Приведено к Basic / Comfort / Premium", "Цена monthly_fee_rub подтверждает тариф в каждой строке",
            (raw_tariff != df.tariff).sum(), "ячейки")

    for c in MONEY + ["traffic_gb"] + INTS:
        commas = df[c].str.contains(",", regex=False)
        if commas.any():
            log.add("billing_monthly", c, "Десятичная запятая", "Приведено к точке",
                    "Формат числа, значение не меняется", commas.sum(), "ячейки")
        df[c] = parse_number(df[c])

    restore = df.closing_debt_rub.isna()
    df["closing_debt_restored"] = restore
    df.loc[restore, "closing_debt_rub"] = (df.opening_debt_rub + df.charged_rub - df.paid_rub)[restore].round(2)
    log.add("billing_monthly", "closing_debt_rub", "Пропуск конечного долга",
            "Восстановлено: opening + charged − paid", "Равенство выполняется во всех строках с известным долгом "
            "и согласовано с opening_debt следующего месяца", restore.sum(), "ячейки")

    df["traffic_gb_missing"] = df.traffic_gb.isna()
    log.add("billing_monthly", "traffic_gb", "Пропуск трафика", "Оставлен пропуск + флаг traffic_gb_missing",
            "Пропуск не равен нулю; импутация только позже на обучающих данных", df.traffic_gb_missing.sum(), "ячейки")

    bad_voice = df.voice_minutes < 0
    df["voice_minutes_invalid"] = bad_voice
    df.loc[bad_voice, "voice_minutes"] = np.nan
    log.add("billing_monthly", "voice_minutes", "Отрицательное число минут",
            "Значение заменено пропуском + флаг voice_minutes_invalid",
            "Минуты неотрицательны; знак и величину восстановить нельзя; абонент сохраняется", bad_voice.sum(), "ячейки")

    return df.sort_values(["customer_id", "billing_month"]).reset_index(drop=True)


# ---------- поддержка ----------

def _rename_legacy(raw: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    mapping = dict(zip(schema.legacy_field, schema.canonical_field))
    return raw.rename(columns=mapping)[list(schema.canonical_field)]


def clean_support(legacy_raw: pd.DataFrame, current_raw: pd.DataFrame, schema: pd.DataFrame,
                  subscribers: pd.DataFrame, log: DecisionLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (support_clean, quarantine)."""
    parts = []
    for system, raw, parser in [("legacy", _rename_legacy(legacy_raw, schema), parse_legacy_ts),
                                ("current", current_raw[list(schema.canonical_field)], parse_current_ts)]:
        n = len(raw)
        df = raw.drop_duplicates().copy()
        log.add(f"support_{system}", "*", "Полные повторные строки", "Удалены повторные копии",
                "Копия одного обращения удваивала бы счётчики; разные обращения одного абонента сохраняются",
                n - len(df), "повторные копии")
        df.insert(0, "source_system", system)
        df["raw_created_at"] = df.created_at
        df["raw_customer_id"] = df.customer_id
        df["created_tz_assumed"] = (system == "current") & ~df.created_at.str.strip().str.endswith("Z")
        for c in SUPPORT_TIME:
            df[c] = parser(df[c])
        if system == "legacy":
            df["topic"] = df.topic.map(config.TOPIC_LEGACY).fillna(df.topic)
            df["channel"] = df.channel.map(config.CHANNEL_LEGACY).fillna(df.channel)
        parts.append(df)

    df = pd.concat(parts, ignore_index=True)
    log.add("support_legacy", "все поля", "Русские названия полей и категорий, местное время",
            "Переименовано по schema_support.csv, темы и каналы приведены к кодам, время — Москва ISO 8601",
            "Единая схема обеих систем", (df.source_system == "legacy").sum(), "строки")
    log.add("support_current", "created_at, available_at, ...", "Время в UTC (суффикс Z)",
            "Переведено в Москву +03:00 с сохранением момента", "Единый часовой пояс", (df.source_system == "current").sum(),
            "строки")
    log.add("support_current", "created_at", "Время без часового пояса", "Интерпретировано как Москва, флаг created_tz_assumed",
            "Подтверждено заказчиком; при UTC-интерпретации часть записей нарушала бы порядок created ≤ available",
            df.created_tz_assumed.sum(), "ячейки")

    if df.ticket_id.duplicated().any():
        raise ValueError("конфликт значений ticket_id после удаления полных копий — нужна ручная проверка")

    stripped = df.customer_id.str.strip()
    log.add("support", "customer_id", "Пробелы вокруг идентификатора", "Удалены пробелы",
            "Пробелы незначимы по schema_support.csv", (stripped != df.customer_id).sum(), "ячейки")
    df["customer_id"] = stripped

    empty_topic = df.topic.str.strip().eq("")
    df.loc[empty_topic, "topic"] = "unknown"
    log.add("support", "topic", "Пустая тема", "Отдельная категория unknown",
            "Разрешено схемой; обращение учитывается в support_count и support_unknown_topic_count",
            empty_topic.sum(), "ячейки")

    score = parse_number(df.satisfaction_score)
    bad_score = score.notna() & ~score.isin([1, 2, 3, 4, 5])
    df["satisfaction_score_invalid"] = bad_score
    df["satisfaction_score"] = score.where(~bad_score)
    log.add("support", "satisfaction_score", "Оценка вне шкалы 1–5 (значение 9)",
            "Заменено пропуском + флаг", "Восстановить истинную оценку нельзя; обращение остаётся",
            bad_score.sum(), "ячейки")

    bad_resolution = df.resolved_at.notna() & (df.resolved_at < df.created_at)
    df["resolution_invalid"] = bad_resolution
    for c in ["resolved_at", "resolution_available_at"]:
        df.loc[bad_resolution, c] = pd.NaT
    df.loc[bad_resolution, "satisfaction_score"] = np.nan
    log.add("support", "resolved_at", "Закрытие раньше создания",
            "Закрытие и оценка не используются (пропуск + флаг resolution_invalid); создание обращения сохранено",
            "Факт закрытия недостоверен, но время создания и доступности корректно", bad_resolution.sum(), "строки")

    # Карантин: нельзя установить абонента или безопасную дату доступности
    known = set(subscribers.customer_id)
    reason = pd.Series("", index=df.index)
    reason[df.customer_id.eq("")] = "missing_customer_id"
    reason[reason.eq("") & ~df.customer_id.isin(known)] = "unknown_customer_id"
    reason[reason.eq("") & (df.available_at < df.created_at)] = "available_before_created"
    in_q = reason.ne("")
    for r, text in [("missing_customer_id", "Пустой customer_id"),
                    ("unknown_customer_id", "customer_id отсутствует в справочнике"),
                    ("available_before_created", "available_at раньше created_at")]:
        log.add("support", "customer_id" if "customer" in r else "available_at", text,
                "Перенесено в карантин с причиной", "Нельзя связать с абонентом или установить безопасную дату "
                "доступности для снимка", (reason == r).sum(), "строки")

    quarantine = df[in_q].copy()
    quarantine["reason"] = reason[in_q]
    quarantine["linked_customer"] = quarantine.customer_id.isin(known)
    quarantine["created_month"] = quarantine.created_at.dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()

    clean = df[~in_q].drop(columns=["raw_created_at", "raw_customer_id"])
    clean = clean.sort_values(["created_at", "ticket_id"]).reset_index(drop=True)
    return clean, quarantine.reset_index(drop=True)
