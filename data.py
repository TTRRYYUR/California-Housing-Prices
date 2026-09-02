import pandas as pd


def load_data(filepath):
    """Загружает датасет из CSV-файла"""
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    """Очистка данных и приведение к стандартному виду(duration удаляем потому что происходит data leaking)"""
    df = df.copy()
    valid_mask = (
        (df["total_bedrooms"] <= df["total_rooms"]) &
        ~((df["households"] == 0) & (df["population"] > 0)))
    return df[valid_mask].reset_index(drop=True)
