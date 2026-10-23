"""Пути и согласованные параметры проекта (устав, этапы 2–3)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
LESSON1 = RAW / "lesson1"
LESSON2 = RAW / "lesson2"
LESSON3 = RAW / "lesson3"

INTERIM = ROOT / "data" / "interim"
PREPARED = ROOT / "data" / "prepared"
EDA_DATA = ROOT / "data" / "eda"

REPORTS_STAGE2 = ROOT / "reports" / "stage2"
REPORTS_EDA = ROOT / "reports" / "eda"
FIGURES = REPORTS_EDA / "figures"
TABLES = REPORTS_EDA / "tables"

SQL_DIR = ROOT / "sql"

SOURCES = {
    "subscribers": LESSON1 / "subscribers.csv",
    "billing": LESSON1 / "billing_monthly.csv",
    "events": LESSON1 / "contract_events.csv",
    "dictionary": LESSON1 / "data_dictionary.csv",
    "support_legacy": LESSON2 / "data" / "support_legacy_sep_oct_2025.csv",
    "support_sqlite": LESSON2 / "data" / "support.sqlite",
    "support_current_csv": LESSON2 / "data" / "support_current_export.csv",
    "support_schema": LESSON2 / "schema_support.csv",
    "network": LESSON3 / "data" / "network_quality_monthly.csv",
    "business_context": LESSON3 / "data" / "business_context.csv",
}

# Все моменты времени приводятся к Москве (UTC+3, без перехода на летнее время).
TZ = "Europe/Moscow"

# Охват выгрузок
BILLING_FIRST_MONTH = pd.Timestamp("2025-09-01")
BILLING_LAST_MONTH = pd.Timestamp("2026-08-01")
SUPPORT_CREATED_FROM = pd.Timestamp("2025-09-01", tz=TZ)
SUPPORT_CREATED_TO = pd.Timestamp("2026-09-01", tz=TZ)  # конец исключён
EVENTS_OBSERVED_TO = pd.Timestamp("2026-10-01")  # журнал полон до этой даты, не включая её

# Справочники
TARIFF_PRICE = {"Basic": 450.0, "Comfort": 750.0, "Premium": 1200.0}
TARIFF_ALIASES = {"basic": "Basic", "comfort": "Comfort", "premium": "Premium", "премиум": "Premium"}
TOPIC_LEGACY = {
    "Качество связи": "network",
    "Оплата и начисления": "billing",
    "Тариф": "tariff",
    "Дополнительные услуги": "services",
}
CHANNEL_LEGACY = {"Телефон": "phone", "Чат": "chat", "Офис": "office"}
TOPICS = ("network", "billing", "tariff", "services", "unknown")
CHANNELS = ("phone", "chat", "office")
REGIONS = ("Москва", "Московская область", "Новосибирская область", "Томская область", "Красноярский край")

# Устав: дата расчёта, горизонт и разбиение по времени
HORIZON = "calendar_month"  # [T, T + 1 месяц)
EDA_DATES = pd.date_range("2025-12-01", "2026-07-01", freq="MS")
VALIDATION_DATES = pd.DatetimeIndex(["2026-08-01"])
TEST_DATES = pd.DatetimeIndex(["2026-09-01"])
ACTIVE_BASE_RULE = "activation_date < T and (termination_date is null or termination_date >= T)"
CAMPAIGN_SHARE = 0.10

TENURE_THRESHOLD_DAYS = 90
SMALL_GROUP = 30
