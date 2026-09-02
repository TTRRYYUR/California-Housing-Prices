import joblib
import pandas as pd
from functools import lru_cache

from features import get_raw_features_columns


@lru_cache(maxsize=1)
def load_model(model_path = "best_model.pkl"):
    """Загружает модель в память"""
    return joblib.load(model_path)


def predict_price(features, model_path = "best_model.pkl"):
    """Предсказание цены для новых данных"""
    pipeline = load_model(model_path)

    num_cols, cat_cols = get_raw_features_columns()
    orger_cols = num_cols + cat_cols
    df = pd.DataFrame([features])[orger_cols]
    return float(pipeline.predict(df)[0])
