from pathlib import Path

import joblib
import numpy as np
import pandas as pd

project_dir = Path(__file__).resolve().parents[1]

processed_data_dir = project_dir / "data" / "processed"
models_dir = project_dir / "models"
powerbi_data_dir = project_dir / "data" / "powerbi"
reports_tables_dir = project_dir / "reports" / "tables"

powerbi_data_dir.mkdir(parents=True, exist_ok=True)
reports_tables_dir.mkdir(parents=True, exist_ok=True)

rentals = pd.read_csv(processed_data_dir / "rentals_clean.csv")
sales = pd.read_csv(processed_data_dir / "sales_clean.csv")

price_model = joblib.load(models_dir / "price_model.pkl")
rent_model = joblib.load(models_dir / "rent_model.pkl")

print("=" * 80)
print("Preparing ML-based investment predictions")
print("Rentals:", rentals.shape)
print("Sales:", sales.shape)
print()

rentals["date_posted"] = pd.to_datetime(rentals["date_posted"], errors="coerce")
sales["date_posted"] = pd.to_datetime(sales["date_posted"], errors="coerce")

rentals["is_first_floor"] = rentals["is_first_floor"].astype(int)
rentals["is_last_floor"] = rentals["is_last_floor"].astype(int)

rentals["furnished"] = rentals["furnished"].astype(int)
rentals["furnished_int"] = rentals["furnished_int"].astype(int)

rentals["pets_allowed"] = rentals["pets_allowed"].astype(int)
rentals["pets_allowed_int"] = rentals["pets_allowed_int"].astype(int)

mortgage_by_month = (
    sales
    .groupby(["listing_year", "listing_month"], as_index=False)
    .agg(
        mortgage_rate_at_listing=("mortgage_rate_at_listing", "median")
    )
)

overall_mortgage_rate = sales["mortgage_rate_at_listing"].median()

investment_data = rentals.copy()

investment_data = investment_data.merge(
    mortgage_by_month,
    on=["listing_year", "listing_month"],
    how="left",
)

investment_data["mortgage_rate_at_listing"] = investment_data["mortgage_rate_at_listing"].fillna(
    overall_mortgage_rate
)

investment_data["market_type"] = "Secondary Market"

predicted_sale_price = price_model.predict(investment_data)
predicted_monthly_rent = rent_model.predict(investment_data)

investment_data["predicted_sale_price_rub"] = predicted_sale_price
investment_data["predicted_monthly_rent_rub"] = predicted_monthly_rent

investment_data["predicted_sale_price_rub"] = investment_data["predicted_sale_price_rub"].clip(lower=1)
investment_data["predicted_monthly_rent_rub"] = investment_data["predicted_monthly_rent_rub"].clip(lower=1)

investment_data["predicted_sale_price_mln"] = (
        investment_data["predicted_sale_price_rub"] / 1_000_000
)

investment_data["actual_monthly_rent_thousand"] = (
        investment_data["monthly_rent_rub"] / 1_000
)

investment_data["predicted_monthly_rent_thousand"] = (
        investment_data["predicted_monthly_rent_rub"] / 1_000
)

investment_data["predicted_annual_rent_rub"] = (
        investment_data["predicted_monthly_rent_rub"] * 12
)

investment_data["actual_annual_rent_rub"] = (
        investment_data["monthly_rent_rub"] * 12
)

investment_data["predicted_rental_yield_pct"] = (
        investment_data["predicted_annual_rent_rub"]
        / investment_data["predicted_sale_price_rub"]
        * 100
)

investment_data["actual_rental_yield_pct"] = (
        investment_data["actual_annual_rent_rub"]
        / investment_data["predicted_sale_price_rub"]
        * 100
)

investment_data["predicted_rental_yield_pct"] = investment_data["predicted_rental_yield_pct"].replace(
    [np.inf, -np.inf],
    np.nan,
)

investment_data["actual_rental_yield_pct"] = investment_data["actual_rental_yield_pct"].replace(
    [np.inf, -np.inf],
    np.nan,
)

investment_data["investment_score"] = pd.cut(
    investment_data["predicted_rental_yield_pct"],
    bins=[0, 5, 6, 100],
    labels=["Low Yield", "Medium Yield", "High Yield"],
    include_lowest=True,
)

investment_data["predicted_sale_price_group"] = pd.cut(
    investment_data["predicted_sale_price_mln"],
    bins=[0, 7, 10, 15, 25, 50, 10_000],
    labels=[
        "<7 mln RUB",
        "7-10 mln RUB",
        "10-15 mln RUB",
        "15-25 mln RUB",
        "25-50 mln RUB",
        "50+ mln RUB",
    ],
    include_lowest=True,
)

investment_data["predicted_rent_price_group"] = pd.cut(
    investment_data["predicted_monthly_rent_thousand"],
    bins=[0, 30, 50, 70, 100, 150, 10_000],
    labels=[
        "<30k RUB",
        "30-50k RUB",
        "50-70k RUB",
        "70-100k RUB",
        "100-150k RUB",
        "150k+ RUB",
    ],
    include_lowest=True,
)

investment_data["listing_date"] = investment_data["date_posted"].dt.date
investment_data["listing_year_month"] = investment_data["date_posted"].dt.to_period("M").astype(str)

investment_predictions = investment_data[
    [
        "id",
        "listing_date",
        "listing_year_month",
        "listing_year",
        "listing_month",
        "listing_quarter",
        "market_type",
        "okrug",
        "district",
        "lat",
        "lon",
        "total_area",
        "rooms",
        "rooms_group",
        "floor",
        "total_floors",
        "is_first_floor",
        "is_last_floor",
        "floor_ratio",
        "building_year",
        "building_age",
        "building_type",
        "renovation",
        "furnished",
        "furnished_status",
        "pets_allowed",
        "pets_allowed_status",
        "deposit_months",
        "metro_station",
        "metro_line",
        "metro_distance_min",
        "to_center_km",
        "distance_to_center_group",
        "monthly_rent_rub",
        "actual_monthly_rent_thousand",
        "predicted_monthly_rent_rub",
        "predicted_monthly_rent_thousand",
        "predicted_rent_price_group",
        "predicted_sale_price_rub",
        "predicted_sale_price_mln",
        "predicted_sale_price_group",
        "predicted_annual_rent_rub",
        "actual_annual_rent_rub",
        "predicted_rental_yield_pct",
        "actual_rental_yield_pct",
        "investment_score",
    ]
].copy()

district_investment_summary = (
    investment_predictions
    .groupby(["okrug", "district"], as_index=False)
    .agg(
        listings_count=("id", "count"),
        avg_predicted_sale_price_rub=("predicted_sale_price_rub", "mean"),
        median_predicted_sale_price_rub=("predicted_sale_price_rub", "median"),
        avg_predicted_sale_price_mln=("predicted_sale_price_mln", "mean"),
        median_predicted_sale_price_mln=("predicted_sale_price_mln", "median"),
        avg_actual_monthly_rent_rub=("monthly_rent_rub", "mean"),
        median_actual_monthly_rent_rub=("monthly_rent_rub", "median"),
        avg_predicted_monthly_rent_rub=("predicted_monthly_rent_rub", "mean"),
        median_predicted_monthly_rent_rub=("predicted_monthly_rent_rub", "median"),
        avg_predicted_monthly_rent_thousand=("predicted_monthly_rent_thousand", "mean"),
        median_predicted_monthly_rent_thousand=("predicted_monthly_rent_thousand", "median"),
        avg_predicted_rental_yield_pct=("predicted_rental_yield_pct", "mean"),
        median_predicted_rental_yield_pct=("predicted_rental_yield_pct", "median"),
        avg_actual_rental_yield_pct=("actual_rental_yield_pct", "mean"),
        median_actual_rental_yield_pct=("actual_rental_yield_pct", "median"),
        avg_area=("total_area", "mean"),
        avg_to_center_km=("to_center_km", "mean"),
    )
)

district_investment_summary["investment_score"] = pd.cut(
    district_investment_summary["median_predicted_rental_yield_pct"],
    bins=[0, 5, 6, 100],
    labels=["Low Yield", "Medium Yield", "High Yield"],
    include_lowest=True,
)

district_investment_summary = district_investment_summary.sort_values(
    "median_predicted_rental_yield_pct",
    ascending=False,
)

okrug_investment_summary = (
    investment_predictions
    .groupby("okrug", as_index=False)
    .agg(
        listings_count=("id", "count"),
        avg_predicted_sale_price_rub=("predicted_sale_price_rub", "mean"),
        median_predicted_sale_price_rub=("predicted_sale_price_rub", "median"),
        avg_predicted_sale_price_mln=("predicted_sale_price_mln", "mean"),
        median_predicted_sale_price_mln=("predicted_sale_price_mln", "median"),
        avg_predicted_monthly_rent_rub=("predicted_monthly_rent_rub", "mean"),
        median_predicted_monthly_rent_rub=("predicted_monthly_rent_rub", "median"),
        avg_predicted_monthly_rent_thousand=("predicted_monthly_rent_thousand", "mean"),
        median_predicted_monthly_rent_thousand=("predicted_monthly_rent_thousand", "median"),
        avg_predicted_rental_yield_pct=("predicted_rental_yield_pct", "mean"),
        median_predicted_rental_yield_pct=("predicted_rental_yield_pct", "median"),
        avg_area=("total_area", "mean"),
        avg_to_center_km=("to_center_km", "mean"),
    )
)

okrug_investment_summary["investment_score"] = pd.cut(
    okrug_investment_summary["median_predicted_rental_yield_pct"],
    bins=[0, 5, 6, 100],
    labels=["Low Yield", "Medium Yield", "High Yield"],
    include_lowest=True,
)

okrug_investment_summary = okrug_investment_summary.sort_values(
    "median_predicted_rental_yield_pct",
    ascending=False,
)

for table in [investment_predictions, district_investment_summary, okrug_investment_summary]:
    numeric_columns = table.select_dtypes(include=["float"]).columns
    table[numeric_columns] = table[numeric_columns].round(2)

investment_predictions.to_csv(
    powerbi_data_dir / "investment_predictions.csv",
    index=False,
)

district_investment_summary.to_csv(
    powerbi_data_dir / "district_investment_summary.csv",
    index=False,
)

okrug_investment_summary.to_csv(
    powerbi_data_dir / "okrug_investment_summary.csv",
    index=False,
)

investment_report = pd.DataFrame(
    [
        {
            "table_name": "investment_predictions.csv",
            "rows": len(investment_predictions),
            "columns": investment_predictions.shape[1],
        },
        {
            "table_name": "district_investment_summary.csv",
            "rows": len(district_investment_summary),
            "columns": district_investment_summary.shape[1],
        },
        {
            "table_name": "okrug_investment_summary.csv",
            "rows": len(okrug_investment_summary),
            "columns": okrug_investment_summary.shape[1],
        },
    ]
)

investment_report.to_csv(
    reports_tables_dir / "investment_predictions_report.csv",
    index=False,
)

print("=" * 80)
print("ML-based investment analysis prepared")
print()
print(investment_report)
print()

print("Average predicted gross rental yield:")
print(round(investment_predictions["predicted_rental_yield_pct"].mean(), 2), "%")
print()

print("Top 10 districts by median predicted gross rental yield:")
print(
    district_investment_summary[
        [
            "okrug",
            "district",
            "listings_count",
            "median_predicted_sale_price_mln",
            "median_predicted_monthly_rent_thousand",
            "median_predicted_rental_yield_pct",
            "investment_score",
        ]
    ].head(10)
)
print()

print("Files saved to:")
print(powerbi_data_dir / "investment_predictions.csv")
print(powerbi_data_dir / "district_investment_summary.csv")
print(powerbi_data_dir / "okrug_investment_summary.csv")
