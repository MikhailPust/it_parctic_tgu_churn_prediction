# Прогнозирование оттока абонентов «СвязьРегион»

IT-Практикум «Искусственный интеллект и большие данные», осень 2026.
Выполнил: **Пустовой Михаил Евгеньевич** (один участник, все роли).
Задания курса: [Batmad01/tgu_workshop](https://github.com/Batmad01/tgu_workshop).

## Этапы и материалы для сдачи

| Этап | Тег | Документы | Результаты |
|---|---|---|---|
| 1. Инициация | `stage-1` | [Устав проекта](docs/01_project_charter.md) — постановка, отток, T, горизонт, база, вопросы, роли, план | — |
| 2. Подготовка данных | `stage-2` | [Отчёт о качестве](docs/02_quality_report.md), [словарь данных](docs/02_data_dictionary.csv) | `data/prepared/*.csv`, карантин, [журнал решений](reports/stage2/decision_log.csv), [проверки до/после](reports/stage2/quality_checks.csv), [сводка](reports/stage2/quality_summary.json) |
| 3. Разведочный анализ | `stage-3` | [Отчёт EDA](docs/03_eda_report.md), [журнал гипотез](reports/eda/analysis_plan.csv) | `data/eda/eda_sample.csv`, 6 рисунков `reports/eda/figures`, расчётные таблицы `reports/eda/tables`, [проверки](reports/eda/checks.csv) |

## Запуск

Требуется Python 3.10+ (проверено на 3.13.15).

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m src.prepare      # этап 2: data/raw -> data/prepared, reports/stage2
.venv\Scripts\python -m src.eda          # этап 3: data/prepared -> data/eda, reports/eda
.venv\Scripts\python -m pytest           # 14 тестов
```

В Linux/macOS вместо `.venv\Scripts\` используйте `.venv/bin/`. Резервный путь чтения текущей поддержки из CSV-копии вместо SQLite: `python -m src.prepare --support-source csv`. Он даёт те же результаты.

Повторный запуск даёт побайтно одинаковые таблицы. Исходники в `data/raw` не изменяются: пайплайн сверяет их с `data/raw/checksums.sha256`.

## Структура

```
data/raw/lesson1..3     исходники курса без изменений (+ checksums.sha256)
data/prepared/          очищенные таблицы этапа 2 и карантин поддержки
data/eda/               eda_sample.csv, словарь, таблица доступности сетевых отчётов
sql/                    SQL-извлечение и диагностика SQLite
src/config.py           пути, часовой пояс, даты T, горизонт, правило активной базы
src/load.py             чтение CSV и SQLite (только чтение)
src/audit.py            проверки качества (одни и те же до и после очистки)
src/clean.py            очистка и журнал решений
src/merge.py            агрегаты поддержки и customer_month
src/prepare.py          единый запуск этапа 2
src/snapshots.py        активная база на T, признаки на T, метка отдельно
src/eda.py              единый запуск этапа 3: таблицы, рисунки, проверки
tests/                  pytest-проверки этапов 2 и 3
docs/                   устав и отчёты
reports/                машинно-читаемые результаты, рисунки и таблицы
```

## Ключевые договорённости

- **Отток:** событие `termination`.
- **T:** 00:00 по Москве 1-го числа месяца.
- **Горизонт:** календарный месяц `[T, T+1 мес.)`, конец исключён.
- **Активная база:** `activation < T` и (нет расторжения или `termination ≥ T`).
- **Признаки:** только сведения с `available_at ≤ T`.
- **Разбиение:** EDA и обучение на T = 01.12.2025–01.07.2026, валидация — 01.08.2026, тест — 01.09.2026.
- **Кампания:** не более 10% активной базы. Основная метрика — Recall@10%.
