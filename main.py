import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from data import load_data, clean_data
from features import get_raw_features_columns
from pipelines import create_model_pipeline
from training import train_model, evaluate_model, get_model_configs
from visualization import plot_predictions, plot_residuals, plot_feature_importance


def main():
    raw_df = load_data("housing.csv")
    df = clean_data(raw_df)

    raw_num_cols, raw_cat_cols = get_raw_features_columns()
    X = df[raw_num_cols + raw_cat_cols]
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model_configs = get_model_configs()
    results = []

    for name, config in model_configs.items():
        pipe = create_model_pipeline(config["model"])
        train_result = train_model(pipe, X_train, y_train, config["params"])
        eval_result = evaluate_model(train_result["pipeline"], X_test, y_test)

        print(f"name: {name},"
              f"CV R²: {train_result['cv_score']:.3f},"
              f"Test R²: {eval_result['r2']:.3f},"
              f"Test MAE: ${eval_result['mae']:,.0f},"
              f"Test RMSE: ${eval_result['rmse']:,.0f},"
              f"Test mape: {eval_result['mape']:.2%}")


        results.append({
            "name": name,
            "params": train_result["params"],
            "cv_r2": train_result["cv_score"],
            "test_r2": eval_result["r2"],
            "test_mae": eval_result["mae"],
            "test_rmse": eval_result["rmse"],
            "test_mape": eval_result["mape"],
            "pipeline": train_result["pipeline"],
            "predictions": eval_result["predictions"],
        })

    result_df = pd.DataFrame(results).sort_values("test_r2", ascending=False)
    print("\nИтоговые результаты:")
    print(result_df[["name", "cv_r2", "test_r2", "test_mae"]].to_string(index=False))

    best_result = result_df.iloc[0]
    print(f"\nЛучшая модель: {best_result["name"]}")
    print(f"Test R²: {best_result["test_r2"]:.3f}")
    print(f"MAE: ${best_result["test_mae"]:,.0f}")
    print(f"RMSE: ${best_result["test_rmse"]:,.0f}")
    print(f"MAPE: {best_result["test_mape"]:.2%}")

    plot_predictions(
        y_test,
        best_result["predictions"],
        best_result["test_r2"],
        best_result["test_mae"],
        best_result["name"]
    )
    plot_residuals(y_test, best_result["predictions"], best_result["name"])

    plot_feature_importance(best_result["pipeline"], X_test, y_test)

    joblib.dump(best_result["pipeline"], "best_model.pkl")

if __name__ == "__main__":
    main()