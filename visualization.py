import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance


def plot_predictions(y_test, y_pred, r2, mae, model_name):
    """График реальных vs предсказанных значений"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', label="Предсказания")

    plt.xlabel("Реальные значения")
    plt.ylabel("Предсказанные значения")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], c='r', linestyle='--', label="Идеальное совпадение")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"{model_name} | R² = {r2:.3f} | MAE = {mae:.2f}$")
    plt.tight_layout()
    plt.show()


def plot_residuals(y_test, y_pred, model_name: str):
    """График остатков"""
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', color='green')
    plt.axhline(y=0, color='r', linestyle='--')

    plt.xlabel("Предсказанные значения")
    plt.ylabel("Ошибка (Реальное - Предсказанное)")
    plt.title(f"{model_name} | График остатков")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(pipeline, X_test, y_test):
    """Важность признаков с реальными названиями"""
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring="r2"
    )

    importances = np.abs(result.importances_mean)
    importances_pct = (importances / importances.sum()) * 100

    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except (AttributeError, IndexError):
        feature_names = [f"f{i}" for i in range(len(importances_pct))]

    min_len = min(len(feature_names), len(importances_pct))
    feature_names = feature_names[:min_len]
    importances_pct = importances_pct[:min_len]

    importance_df = pd.DataFrame({
        "Признак": feature_names,
        "Важность_%": importances_pct
    }).sort_values("Важность_%", ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=importance_df.head(20),
        x="Важность_%",
        y="Признак",
        palette="viridis"
    )
    plt.title("Важность признаков (Top-20)", fontsize=14, fontweight='bold')
    plt.xlabel("Важность (%)", fontsize=12)
    plt.ylabel("Признак", fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return importance_df
