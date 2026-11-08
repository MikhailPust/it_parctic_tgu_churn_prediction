# Журнал экспериментов (этап 5)

**Команда:** Пустовой Михаил, Вушанло Фарух, Пилявец Екатерина, Дедков Вячеслав
**Версия кода:** тег `stage-5`
**Версия признаков:** `stage-4` (SHA-256 `features.csv` записан в [`reports/stage5/selection.json`](../reports/stage5/selection.json))

**Периоды:**

- train — T = 01.12.2025–01.07.2026 (15 696 строк, 440 уходов);
- validation — T = 01.08.2026 (1 929 строк, 63 ухода);
- test — 01.09.2026, **не загружался и не оценивался**.

**Основная метрика:** Recall@K при `K = floor(0,10 × N_active)`. На validation одна дата, поэтому macro и pooled совпадают. На train приводится macro — среднее Recall@K по 8 датам; pooled — Σ найденных / Σ ушедших.

**Правило равенства:** при равном score — `customer_id` по возрастанию.

Все параметры, правило выбора и фолды записаны **до запуска** в [`config/stage5.json`](../config/stage5.json). Машинно-читаемый журнал — [`reports/stage5/experiment_log.csv`](../reports/stage5/experiment_log.csv), таблица по каждой T — [`metrics_by_T.csv`](../reports/stage5/metrics_by_T.csv).

## Конфигурации

Fit — время обучения на train на нашей машине (Windows 11, Python 3.13.15). Оно немного меняется от запуска к запуску.

| ID | Метод | Параметры | Seed | Версия X | Fit, с | Статус |
|---|---|---|---:|---|---:|---|
| baseline_debt | Правило | сортировка по `debt_prev_rub` ↓, неизвестный долг — после известных, равенства — `customer_id` ↑ | — | stage-4 | — | ok |
| exp01_lr_c0.1 | LogisticRegression (lbfgs, max_iter 2000) | C = 0,1 | 42 | stage-4, 59 столбцов | 0,09 | ok |
| exp02_lr_c1 | LogisticRegression | C = 1,0 | 42 | stage-4 | 0,07 | ok |
| exp03_lr_c0.1_balanced | LogisticRegression | C = 0,1, class_weight = balanced | 42 | stage-4 | 0,13 | ok |
| exp04_hgb_leaf7 | HistGradientBoostingClassifier | max_leaf_nodes 7, max_iter 100, learning_rate 0,07, min_samples_leaf 50, l2 1,0, early_stopping off | 42 | stage-4 | 0,27 | ok |
| exp05_hgb_leaf15 | HistGradientBoostingClassifier | то же, max_leaf_nodes 15 | 42 | stage-4 | 0,36 | ok |

## Результаты

AP — `sklearn.metrics.average_precision_score` по каждой дате, затем среднее по датам. Это не трапецеидальная PR-AUC.

| ID | Часть | Recall@K | Precision@K | AP | Найдено / ушли | Комментарий |
|---|---|---:|---:|---:|---|---|
| baseline_debt | train | 39,6% | 11,2% | 0,088 | 175 / 440 | Не обучается |
| baseline_debt | validation | 22,2% | 7,3% | 0,058 | 14 / 63 | В августе правило работает заметно хуже, чем на train |
| exp01_lr_c0.1 | train | 40,6% | 11,4% | 0,121 | 179 / 440 | |
| exp01_lr_c0.1 | validation | 28,6% | 9,4% | 0,064 | 18 / 63 | |
| exp02_lr_c1 | train | 42,4% | 11,9% | 0,122 | 187 / 440 | |
| exp02_lr_c1 | validation | 28,6% | 9,4% | 0,063 | 18 / 63 | Ослабление регуляризации прироста не дало |
| exp03_lr_c0.1_balanced | train | 40,5% | 11,4% | 0,108 | 178 / 440 | |
| exp03_lr_c0.1_balanced | validation | **30,2%** | **9,9%** | 0,063 | **19 / 63** | **Выбран** |
| exp04_hgb_leaf7 | train | 55,0% | 15,5% | 0,300 | 243 / 440 | Признаки переобучения |
| exp04_hgb_leaf7 | validation | 27,0% | 8,9% | **0,078** | 17 / 63 | Лучший AP, но ниже Recall@K |
| exp05_hgb_leaf15 | train | 70,4% | 19,9% | 0,546 | 311 / 440 | Сильное переобучение |
| exp05_hgb_leaf15 | validation | 28,6% | 9,4% | 0,062 | 18 / 63 | |

**Временные фолды внутри train** ([`temporal_folds.csv`](../reports/stage5/temporal_folds.csv)). Это дополнительная проверка, в выборе не участвует. Подготовка и модель обучаются заново в каждом фолде.

| Подход | май (обучение дек–апр) | июнь (дек–май) | июль (дек–июнь) | Среднее Recall@K |
|---|---:|---:|---:|---:|
| baseline_debt | 26 / 58 | 23 / 60 | 25 / 62 | **41,2%** |
| exp01_lr_c0.1 | 21 / 58 | 24 / 60 | 23 / 62 | 37,8% |
| exp02_lr_c1 | 23 / 58 | 23 / 60 | 23 / 62 | 38,4% |
| exp03_lr_c0.1_balanced | 24 / 58 | 23 / 60 | 23 / 62 | 38,9% |
| exp04_hgb_leaf7 | 23 / 58 | 20 / 60 | 25 / 62 | 37,8% |
| exp05_hgb_leaf15 | 19 / 58 | 24 / 60 | 21 / 62 | 35,5% |

**Ошибки и предупреждения.** Ошибок нет. Предупреждений о несходимости (`ConvergenceWarning`) нет: lbfgs сошёлся за 2 000 итераций. Предупреждения перехватываются и записываются в столбец `warnings` журнала.

## Выбор

**Правило, записанное до сравнения:** максимальный validation Recall@K; при равенстве — максимальный validation AP; при полном равенстве — более ранний опыт в порядке конфигурации.

**Выбранный кандидат:** `exp03_lr_c0.1_balanced` — единственный опыт с 19 найденными уходами на validation. У остальных 17–18.

**Против baseline на validation:** +5 найденных уходов (19 против 14 из 63), Recall@K 30,2% против 22,2%, Precision@K 9,9% против 7,3%.

**Оговорка.** На трёх фолдах внутри train baseline находит в среднем больше: 74 ухода из 180 против 70 у кандидата. Преимущество модели на одном месяце validation неустойчиво. Модель не объявляется готовой к внедрению, правило по долгу остаётся рабочей альтернативой (подробнее — [отчёт](05_training_report.md)).

**Где сохранён Pipeline и как проверена загрузка:**

- кандидат — `models/stage5_candidate.joblib`: подготовка вместе с классификатором, списки и порядок входных столбцов, параметры, seed, версии библиотек, SHA-256 признаков;
- проверка — `python -m src.check_model`: в новом процессе выполняется только `predict_proba`, без fit;
- результат в [`reports/stage5/reload_check.json`](../reports/stage5/reload_check.json): ключи, score (разница 0), ранги, top-K и метрики совпадают.
