"""Загрузка исходников без изменений: всё читается как текст."""
import re
import sqlite3
from pathlib import Path

import pandas as pd

from . import config


def read_raw_csv(path: Path) -> pd.DataFrame:
    """CSV курса: UTF-8 с BOM, «;». Пустая ячейка остаётся пустой строкой, а не NaN."""
    return pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=";", encoding="utf-8-sig", index=False, lineterminator="\n")


def _sql_statements(path: Path) -> list[tuple[str, str]]:
    """Разбивает SQL-файл на (имя, запрос): имя — первый комментарий перед запросом."""
    text = path.read_text(encoding="utf-8")
    result = []
    for chunk in text.split(";"):
        names = re.findall(r"^--\s*(\w+)\s*$", chunk, flags=re.M)
        body = "\n".join(line for line in chunk.splitlines() if not line.strip().startswith("--")).strip()
        if body:
            result.append((names[-1] if names else "query", body))
    return result


def extract_support_sqlite(db: Path) -> tuple[pd.DataFrame, dict]:
    """Читает support_current_raw запросом из sql/ в режиме только чтения."""
    query = (config.SQL_DIR / "extract_support_current.sql").read_text(encoding="utf-8")
    with sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True) as conn:
        cursor = conn.execute(query)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
        diagnostics = {}
        for name, stmt in _sql_statements(config.SQL_DIR / "diagnostics_support_current.sql"):
            diagnostics[name] = conn.execute(stmt).fetchone()[0]
    df = pd.DataFrame(rows, columns=columns, dtype=object).fillna("").astype(str)
    return df, diagnostics


def load_sources(support_source: str = "sqlite") -> tuple[dict[str, pd.DataFrame], dict]:
    """Возвращает сырые таблицы и метаданные загрузки.

    support_source="sqlite" — основной путь; "csv" — резервное чтение точной копии.
    Две копии текущей системы никогда не объединяются.
    """
    raw = {
        "subscribers": read_raw_csv(config.SOURCES["subscribers"]),
        "billing": read_raw_csv(config.SOURCES["billing"]),
        "events": read_raw_csv(config.SOURCES["events"]),
        "support_legacy": read_raw_csv(config.SOURCES["support_legacy"]),
    }
    meta = {"support_current_source": support_source}
    if support_source == "sqlite":
        raw["support_current"], meta["sqlite_diagnostics"] = extract_support_sqlite(config.SOURCES["support_sqlite"])
        write_csv(raw["support_current"], config.INTERIM / "support_current_extracted.csv")
    elif support_source == "csv":
        raw["support_current"] = read_raw_csv(config.SOURCES["support_current_csv"])
    else:
        raise ValueError(f"unknown support source: {support_source}")
    return raw, meta
