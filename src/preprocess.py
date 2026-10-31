"""Преобразование для первого кандидата (логистическая регрессия): fit только на train.

Числа: медиана train + индикаторы пропуска + стандартизация.
Категории: отдельное значение __MISSING__ + One-Hot, новая категория -> нулевой вектор (handle_unknown='ignore').
Список столбцов задаётся явно (features.NUMERIC / features.CATEGORICAL), а не «всё, кроме target».
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

FORBIDDEN = {"customer_id", "snapshot_at", "churn_target", "target", "target_end_exclusive", "label_observed",
             "split"}


def _as_object(values):
    # Полностью пустой категориальный столбец иначе может стать float.
    return values.astype(object)


def make_preprocessor(numeric_columns, categorical_columns) -> ColumnTransformer:
    numeric_columns, categorical_columns = list(numeric_columns), list(categorical_columns)
    all_columns = numeric_columns + categorical_columns
    if not all_columns or len(set(all_columns)) != len(all_columns):
        raise ValueError("списки признаков должны быть непустыми, уникальными и не пересекаться")
    if set(all_columns) & FORBIDDEN:
        raise ValueError("ключи, метка, разбиение и служебные поля не являются признаками")
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("as_object", FunctionTransformer(_as_object, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("numeric", numeric, numeric_columns), ("categorical", categorical, categorical_columns)],
                             remainder="drop", verbose_feature_names_out=True)
