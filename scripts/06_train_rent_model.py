from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

project_dir = Path(__file__).resolve().parents[1]

processed_data_dir = project_dir / "data" / "processed"
models_dir = project_dir / "models"
reports_tables_dir = project_dir / "reports" / "tables"
reports_figures_dir = project_dir / "reports" / "figures"

models_dir.mkdir(parents=True, exist_ok=True)
reports_tables_dir.mkdir(parents=True, exist_ok=True)
reports_figures_dir.mkdir(parents=True, exist_ok=True)

rentals = pd.read_csv(processed_data_dir / "rentals_clean.csv")

print("=" * 80)
print("Training rent prediction model")
print("Dataset shape:", rentals.shape)
print()

rentals = rentals.dropna(subset=["monthly_rent_rub"])

rentals["is_first_floor"] = rentals["is_first_floor"].astype(int)
rentals["is_last_floor"] = rentals["is_last_floor"].astype(int)
rentals["furnished"] = rentals["furnished"].astype(int)
rentals["pets_allowed"] = rentals["pets_allowed"].astype(int)

target_column = "monthly_rent_rub"

numeric_features = [
    "total_area",
    "rooms",
    "floor",
    "total_floors",
    "is_first_floor",
    "is_last_floor",
    "floor_ratio",
    "building_year",
    "building_age",
    "furnished",
    "furnished_int",
    "pets_allowed",
    "pets_allowed_int",
    "deposit_months",
    "metro_distance_min",
    "to_center_km",
    "listing_year",
    "listing_month",
    "listing_quarter",
    "lat",
    "lon",
]

categorical_features = [
    "okrug",
    "district",
    "building_type",
    "renovation",
    "metro_station",
    "metro_line",
    "distance_to_center_group",
]

numeric_features = [column for column in numeric_features if column in rentals.columns]
categorical_features = [column for column in categorical_features if column in rentals.columns]

features = numeric_features + categorical_features

model_data = rentals[features + [target_column]].copy()

print("Target:", target_column)
print("Numeric features:")
print(numeric_features)
print()
print("Categorical features:")
print(categorical_features)
print()

X = model_data[features]
y = model_data[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


def create_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


models = {
    "baseline_median": DummyRegressor(strategy="median"),
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    ),
}

results = []
trained_pipelines = {}

for model_name, model in models.items():
    print("=" * 80)
    print(f"Training model: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results.append(
        {
            "model": model_name,
            "mae_rub": round(mae, 2),
            "mae_thousand_rub": round(mae / 1_000, 2),
            "rmse_rub": round(rmse, 2),
            "rmse_thousand_rub": round(rmse / 1_000, 2),
            "r2": round(r2, 4),
            "mape_pct": round(mape, 2),
        }
    )

    trained_pipelines[model_name] = pipeline

    print(f"MAE: {mae:,.0f} RUB")
    print(f"RMSE: {rmse:,.0f} RUB")
    print(f"R2: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

metrics = pd.DataFrame(results)
metrics = metrics.sort_values("mae_rub")

metrics.to_csv(
    reports_tables_dir / "rent_model_metrics.csv",
    index=False,
)

print("=" * 80)
print("Model comparison")
print(metrics)
print()

best_model_name = metrics.iloc[0]["model"]
best_pipeline = trained_pipelines[best_model_name]

joblib.dump(
    best_pipeline,
    models_dir / "rent_model.pkl",
)

print("=" * 80)
print("Best model:", best_model_name)
print(f"Model saved to: {models_dir / 'rent_model.pkl'}")
print()

best_predictions = best_pipeline.predict(X_test)

predictions_report = X_test.copy()
predictions_report["actual_rent_rub"] = y_test.values
predictions_report["predicted_rent_rub"] = best_predictions
predictions_report["error_rub"] = predictions_report["actual_rent_rub"] - predictions_report["predicted_rent_rub"]
predictions_report["abs_error_rub"] = predictions_report["error_rub"].abs()
predictions_report["actual_rent_thousand"] = predictions_report["actual_rent_rub"] / 1_000
predictions_report["predicted_rent_thousand"] = predictions_report["predicted_rent_rub"] / 1_000
predictions_report["abs_error_thousand"] = predictions_report["abs_error_rub"] / 1_000

predictions_report = predictions_report.round(2)

predictions_report.to_csv(
    reports_tables_dir / "rent_model_predictions.csv",
    index=False,
)

plot_data = predictions_report.sample(
    n=min(3000, len(predictions_report)),
    random_state=42,
)

plt.figure(figsize=(8, 8))
plt.scatter(
    plot_data["actual_rent_thousand"],
    plot_data["predicted_rent_thousand"],
    alpha=0.4,
)

min_rent = min(
    plot_data["actual_rent_thousand"].min(),
    plot_data["predicted_rent_thousand"].min(),
)

max_rent = max(
    plot_data["actual_rent_thousand"].max(),
    plot_data["predicted_rent_thousand"].max(),
)

plt.plot(
    [min_rent, max_rent],
    [min_rent, max_rent],
)

plt.title("Actual vs Predicted Monthly Rent")
plt.xlabel("Actual monthly rent, k RUB")
plt.ylabel("Predicted monthly rent, k RUB")
plt.tight_layout()
plt.savefig(
    reports_figures_dir / "rent_model_actual_vs_predicted.png",
    dpi=300,
)
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(
    predictions_report["abs_error_thousand"],
    bins=50,
)
plt.title("Rent Model Absolute Error Distribution")
plt.xlabel("Absolute error, k RUB")
plt.ylabel("Listings")
plt.tight_layout()
plt.savefig(
    reports_figures_dir / "rent_model_error_distribution.png",
    dpi=300,
)
plt.close()

preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
model_fitted = best_pipeline.named_steps["model"]

feature_names = preprocessor_fitted.get_feature_names_out()

feature_importance = None

if hasattr(model_fitted, "feature_importances_"):
    feature_importance = model_fitted.feature_importances_

if hasattr(model_fitted, "coef_"):
    feature_importance = np.abs(model_fitted.coef_)

if feature_importance is not None:
    feature_importance_report = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": feature_importance,
        }
    )

    feature_importance_report = feature_importance_report.sort_values(
        "importance",
        ascending=False,
    )

    feature_importance_report.to_csv(
        reports_tables_dir / "rent_model_feature_importance.csv",
        index=False,
    )

    top_features = feature_importance_report.head(25).sort_values(
        "importance",
        ascending=True,
    )

    plt.figure(figsize=(10, 8))
    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )
    plt.title("Top 25 Features for Monthly Rent Prediction")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        reports_figures_dir / "rent_model_feature_importance.png",
        dpi=300,
    )
    plt.close()

    print("Feature importance saved to:")
    print(reports_tables_dir / "rent_model_feature_importance.csv")
    print(reports_figures_dir / "rent_model_feature_importance.png")
    print()

print("=" * 80)
print("Rent model training completed")
print()
print(f"Metrics saved to: {reports_tables_dir / 'rent_model_metrics.csv'}")
print(f"Predictions saved to: {reports_tables_dir / 'rent_model_predictions.csv'}")
print(f"Figures saved to: {reports_figures_dir}")
