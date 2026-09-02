import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline


def get_model_configs():
    """Модели и их гиперпараметры"""
    return {
        "Linear": {
            "model": LinearRegression(),
            "params": {}
        },
        "Ridge": {
            "model": Ridge(random_state=42),
            "params": {"model__alpha": loguniform(1e-2, 1e3)}
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                "model__n_estimators": randint(50, 301),
                "model__max_depth": randint(5, 30),
                "model__min_samples_leaf": randint(1, 6)
            }
        },
        "HistGB": {
            "model": HistGradientBoostingRegressor(random_state=42),
            "params": {
                "model__learning_rate": uniform(0.01, 0.19),
                "model__max_iter": randint(100, 401),
                "model__max_depth": randint(3, 11),
                "model__l2_regularization": uniform(0, 10)
            }
        }
    }


def train_model(Pipeline, X_train, y_train, param_grid):
    """Обучение модели с подбором гиперпараметров"""
    if param_grid:
        search = RandomizedSearchCV(
            Pipeline,
            param_distributions=param_grid,
            n_iter=30,
            cv=3,
            scoring="r2",
            n_jobs=-1,
            random_state=42
        )
        search.fit(X_train, y_train)

        return {
            "pipeline": search.best_estimator_,
            "params": search.best_params_,
            "cv_score": search.best_score_
        }
    else:
        Pipeline.fit(X_train, y_train)
        cv_scores = cross_val_score(Pipeline, X_train, y_train, cv=3, scoring="r2")

        return {
            "pipeline": Pipeline,
            "params": "default",
            "cv_score": cv_scores.mean()
        }


def evaluate_model(pipeline: Pipeline, X_test, y_test):
    """Предсказание и оценка модели"""
    y_pred = pipeline.predict(X_test)

    return {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mape": mean_absolute_percentage_error(y_test, y_pred),
        "predictions": y_pred
    }
