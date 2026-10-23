# Прогнозирование оттока абонентов «СвязьРегион»

IT-Практикум «Искусственный интеллект и большие данные», осень 2026.
Выполнил: Пустовой Михаил Евгеньевич.

- Этап 1: [устав проекта](docs/01_project_charter.md)
- Исходные данные курса (без изменений): `data/raw/lesson1..3`
- Этап 2: [отчёт о качестве](docs/02_quality_report.md), [словарь данных](docs/02_data_dictionary.csv),
  очищенные таблицы в `data/prepared`, журнал решений и проверки в `reports/stage2`

Запуск подготовки данных (Python 3.10+):

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m src.prepare
.venv\Scripts\python -m pytest
```
