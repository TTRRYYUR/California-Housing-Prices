import numpy as np
from sklearn.preprocessing import FunctionTransformer


RAW_NUM_COLS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CAT_COLS = ["ocean_proximity"]

ENGINEERED_NUM_COLS = [
    "rooms_per_household",
    "population_per_household",
    "bedrooms_per_room",
    "population_density",
    "lat_lon_interaction",
]


def get_raw_features_columns():
    """Возвращает списки исходных числовых и категориальных колонок до генерации признаков"""
    return RAW_NUM_COLS, CAT_COLS


def get_all_feature_names():
    """Возвращает списки всех числовых и категориальных колонок после генерации признаков"""
    return RAW_NUM_COLS + ENGINEERED_NUM_COLS, CAT_COLS


def add_features(X):
    """Генерирует новые признаковые колонки на основе исходных данных"""
    X = X.copy()
    X["rooms_per_household"] = X["total_rooms"] / X["households"].replace(0, np.nan)
    X["population_per_household"] = X["population"] / X["households"].replace(0, np.nan)
    X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"].replace(0, np.nan)
    X["population_density"] = X["population"] / X["total_rooms"].replace(0, np.nan)
    X["lat_lon_interaction"] = X["latitude"] * X["longitude"]
    return X


def get_feature_engineering_transformer():
    """Возвращает трансформер для применения генерации признаков в пайплайне"""
    return FunctionTransformer(add_features)
