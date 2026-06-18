from pathlib import Path

import numpy as np
import pandas as pd

project_dir = Path(__file__).resolve().parents[1]

processed_data_dir = project_dir / "data" / "processed"
powerbi_data_dir = project_dir / "data" / "powerbi"
reports_tables_dir = project_dir / "reports" / "tables"

powerbi_data_dir.mkdir(parents=True, exist_ok=True)
reports_tables_dir.mkdir(parents=True, exist_ok=True)

sales = pd.read_csv(processed_data_dir / "sales_clean.csv")
secondary = pd.read_csv(processed_data_dir / "secondary_clean.csv")
rentals = pd.read_csv(processed_data_dir / "rentals_clean.csv")
new_builds = pd.read_csv(processed_data_dir / "new_builds_clean.csv")
metro = pd.read_csv(processed_data_dir / "metro_stations_clean.csv")
district_monthly = pd.read_csv(processed_data_dir / "district_monthly_clean.csv")

sales["date_posted"] = pd.to_datetime(sales["date_posted"], errors="coerce")
secondary["date_posted"] = pd.to_datetime(secondary["date_posted"], errors="coerce")
rentals["date_posted"] = pd.to_datetime(rentals["date_posted"], errors="coerce")
new_builds["date_posted"] = pd.to_datetime(new_builds["date_posted"], errors="coerce")
district_monthly["year_month"] = pd.to_datetime(district_monthly["year_month"], errors="coerce")

print("=" * 80)
print("Data for Power BI loaded")
print("Sales:", sales.shape)
print("Secondary market:", secondary.shape)
print("Rentals:", rentals.shape)
print("New builds:", new_builds.shape)
print("Metro stations:", metro.shape)
print("District monthly prices:", district_monthly.shape)

sales_listings = sales.copy()

sales_listings["listing_date"] = sales_listings["date_posted"].dt.date
sales_listings["listing_year_month"] = sales_listings["date_posted"].dt.to_period("M").astype(str)

sales_listings = sales_listings[
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
        "metro_station",
        "metro_line",
        "metro_distance_min",
        "to_center_km",
        "distance_to_center_group",
        "sale_price_group",
        "price_rub",
        "price_mln",
        "price_per_sqm",
        "mortgage_rate_at_listing",
    ]
]

rentals_listings = rentals.copy()

rentals_listings["listing_date"] = rentals_listings["date_posted"].dt.date
rentals_listings["listing_year_month"] = rentals_listings["date_posted"].dt.to_period("M").astype(str)

rentals_listings = rentals_listings[
    [
        "id",
        "listing_date",
        "listing_year_month",
        "listing_year",
        "listing_month",
        "listing_quarter",
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
        "furnished_int",
        "furnished_status",
        "pets_allowed",
        "pets_allowed_int",
        "pets_allowed_status",
        "deposit_months",
        "metro_station",
        "metro_line",
        "metro_distance_min",
        "to_center_km",
        "distance_to_center_group",
        "rent_price_group",
        "monthly_rent_rub",
        "monthly_rent_thousand",
        "rent_per_sqm",
    ]
]

district_monthly_powerbi = district_monthly.copy()
district_monthly_powerbi["year_month_text"] = district_monthly_powerbi["year_month"].dt.to_period("M").astype(str)
district_monthly_powerbi["year"] = district_monthly_powerbi["year_month"].dt.year
district_monthly_powerbi["month"] = district_monthly_powerbi["year_month"].dt.month
district_monthly_powerbi["quarter"] = district_monthly_powerbi["year_month"].dt.quarter

sales_by_district = (
    sales
    .groupby(["okrug", "district"], as_index=False)
    .agg(
        sale_listings=("id", "count"),
        avg_sale_price_rub=("price_rub", "mean"),
        median_sale_price_rub=("price_rub", "median"),
        avg_sale_price_mln=("price_mln", "mean"),
        median_sale_price_mln=("price_mln", "median"),
        avg_sale_price_per_sqm=("price_per_sqm", "mean"),
        median_sale_price_per_sqm=("price_per_sqm", "median"),
        avg_sale_area=("total_area", "mean"),
        avg_sale_to_center_km=("to_center_km", "mean"),
        avg_mortgage_rate=("mortgage_rate_at_listing", "mean"),
    )
)

rentals_by_district = (
    rentals
    .groupby(["okrug", "district"], as_index=False)
    .agg(
        rental_listings=("id", "count"),
        avg_monthly_rent_rub=("monthly_rent_rub", "mean"),
        median_monthly_rent_rub=("monthly_rent_rub", "median"),
        avg_monthly_rent_thousand=("monthly_rent_thousand", "mean"),
        median_monthly_rent_thousand=("monthly_rent_thousand", "median"),
        avg_rent_per_sqm=("rent_per_sqm", "mean"),
        median_rent_per_sqm=("rent_per_sqm", "median"),
        avg_rent_area=("total_area", "mean"),
        avg_rent_to_center_km=("to_center_km", "mean"),
    )
)

district_summary = sales_by_district.merge(
    rentals_by_district,
    on=["okrug", "district"],
    how="outer",
)

district_summary["total_listings"] = (
        district_summary["sale_listings"].fillna(0)
        + district_summary["rental_listings"].fillna(0)
)

district_summary["rental_yield_pct"] = (
        district_summary["median_monthly_rent_rub"] * 12
        / district_summary["median_sale_price_rub"]
        * 100
)

district_summary["rental_yield_pct"] = district_summary["rental_yield_pct"].replace(
    [np.inf, -np.inf],
    np.nan,
)

district_summary["investment_score"] = pd.cut(
    district_summary["rental_yield_pct"],
    bins=[0, 5, 6, 100],
    labels=["Low Yield", "Medium Yield", "High Yield"],
    include_lowest=True,
)

district_summary = district_summary.sort_values(
    "rental_yield_pct",
    ascending=False,
)

okrug_sales = (
    sales
    .groupby("okrug", as_index=False)
    .agg(
        sale_listings=("id", "count"),
        avg_sale_price_rub=("price_rub", "mean"),
        median_sale_price_rub=("price_rub", "median"),
        avg_sale_price_per_sqm=("price_per_sqm", "mean"),
        median_sale_price_per_sqm=("price_per_sqm", "median"),
        avg_sale_area=("total_area", "mean"),
    )
)

okrug_rentals = (
    rentals
    .groupby("okrug", as_index=False)
    .agg(
        rental_listings=("id", "count"),
        avg_monthly_rent_rub=("monthly_rent_rub", "mean"),
        median_monthly_rent_rub=("monthly_rent_rub", "median"),
        avg_rent_per_sqm=("rent_per_sqm", "mean"),
        median_rent_per_sqm=("rent_per_sqm", "median"),
        avg_rent_area=("total_area", "mean"),
    )
)

okrug_summary = okrug_sales.merge(
    okrug_rentals,
    on="okrug",
    how="outer",
)

okrug_summary["total_listings"] = (
        okrug_summary["sale_listings"].fillna(0)
        + okrug_summary["rental_listings"].fillna(0)
)

okrug_summary["rental_yield_pct"] = (
        okrug_summary["median_monthly_rent_rub"] * 12
        / okrug_summary["median_sale_price_rub"]
        * 100
)

okrug_summary["investment_score"] = pd.cut(
    okrug_summary["rental_yield_pct"],
    bins=[0, 5, 6, 100],
    labels=["Low Yield", "Medium Yield", "High Yield"],
    include_lowest=True,
)

okrug_summary = okrug_summary.sort_values(
    "rental_yield_pct",
    ascending=False,
)

sales_by_month = (
    sales
    .groupby(
        ["listing_year", "listing_month", "market_type"],
        as_index=False,
    )
    .agg(
        listings_count=("id", "count"),
        avg_price_rub=("price_rub", "mean"),
        median_price_rub=("price_rub", "median"),
        avg_price_mln=("price_mln", "mean"),
        median_price_mln=("price_mln", "median"),
        avg_price_per_sqm=("price_per_sqm", "mean"),
        median_price_per_sqm=("price_per_sqm", "median"),
        avg_area=("total_area", "mean"),
        avg_mortgage_rate=("mortgage_rate_at_listing", "mean"),
    )
)

sales_by_month["year_month"] = (
        sales_by_month["listing_year"].astype(str)
        + "-"
        + sales_by_month["listing_month"].astype(str).str.zfill(2)
)

rentals_by_month = (
    rentals
    .groupby(["listing_year", "listing_month"], as_index=False)
    .agg(
        listings_count=("id", "count"),
        avg_monthly_rent_rub=("monthly_rent_rub", "mean"),
        median_monthly_rent_rub=("monthly_rent_rub", "median"),
        avg_monthly_rent_thousand=("monthly_rent_thousand", "mean"),
        median_monthly_rent_thousand=("monthly_rent_thousand", "median"),
        avg_rent_per_sqm=("rent_per_sqm", "mean"),
        median_rent_per_sqm=("rent_per_sqm", "median"),
        avg_area=("total_area", "mean"),
    )
)

rentals_by_month["year_month"] = (
        rentals_by_month["listing_year"].astype(str)
        + "-"
        + rentals_by_month["listing_month"].astype(str).str.zfill(2)
)

sales_room_summary = (
    sales
    .groupby(["market_type", "rooms", "rooms_group"], as_index=False)
    .agg(
        sale_listings=("id", "count"),
        avg_sale_price_rub=("price_rub", "mean"),
        median_sale_price_rub=("price_rub", "median"),
        avg_sale_price_per_sqm=("price_per_sqm", "mean"),
        median_sale_price_per_sqm=("price_per_sqm", "median"),
        avg_sale_area=("total_area", "mean"),
    )
)

rentals_room_summary = (
    rentals
    .groupby(["rooms", "rooms_group"], as_index=False)
    .agg(
        rental_listings=("id", "count"),
        avg_monthly_rent_rub=("monthly_rent_rub", "mean"),
        median_monthly_rent_rub=("monthly_rent_rub", "median"),
        avg_rent_per_sqm=("rent_per_sqm", "mean"),
        median_rent_per_sqm=("rent_per_sqm", "median"),
        avg_rent_area=("total_area", "mean"),
    )
)

room_summary = sales_room_summary.merge(
    rentals_room_summary,
    on=["rooms", "rooms_group"],
    how="outer",
)

sales_distance_summary = (
    sales
    .groupby(
        ["market_type", "distance_to_center_group"],
        as_index=False,
    )
    .agg(
        sale_listings=("id", "count"),
        avg_sale_price_rub=("price_rub", "mean"),
        median_sale_price_rub=("price_rub", "median"),
        avg_sale_price_per_sqm=("price_per_sqm", "mean"),
        median_sale_price_per_sqm=("price_per_sqm", "median"),
    )
)

rentals_distance_summary = (
    rentals
    .groupby(["distance_to_center_group"], as_index=False)
    .agg(
        rental_listings=("id", "count"),
        avg_monthly_rent_rub=("monthly_rent_rub", "mean"),
        median_monthly_rent_rub=("monthly_rent_rub", "median"),
        avg_rent_per_sqm=("rent_per_sqm", "mean"),
        median_rent_per_sqm=("rent_per_sqm", "median"),
    )
)

distance_summary = sales_distance_summary.merge(
    rentals_distance_summary,
    on=["distance_to_center_group"],
    how="outer",
)

market_kpi = pd.DataFrame(
    [
        {"metric": "sales_listings", "value": len(sales)},
        {"metric": "secondary_listings", "value": len(secondary)},
        {"metric": "new_builds_listings", "value": len(new_builds)},
        {"metric": "rental_listings", "value": len(rentals)},
        {"metric": "avg_sale_price_rub", "value": sales["price_rub"].mean()},
        {"metric": "median_sale_price_rub", "value": sales["price_rub"].median()},
        {"metric": "avg_sale_price_per_sqm", "value": sales["price_per_sqm"].mean()},
        {"metric": "median_sale_price_per_sqm", "value": sales["price_per_sqm"].median()},
        {"metric": "avg_monthly_rent_rub", "value": rentals["monthly_rent_rub"].mean()},
        {"metric": "median_monthly_rent_rub", "value": rentals["monthly_rent_rub"].median()},
        {"metric": "avg_rent_per_sqm", "value": rentals["rent_per_sqm"].mean()},
        {"metric": "median_rent_per_sqm", "value": rentals["rent_per_sqm"].median()},
        {"metric": "avg_rental_yield_pct", "value": district_summary["rental_yield_pct"].mean()},
    ]
)

tables_for_rounding = [
    sales_listings,
    rentals_listings,
    district_monthly_powerbi,
    district_summary,
    okrug_summary,
    sales_by_month,
    rentals_by_month,
    room_summary,
    distance_summary,
    market_kpi,
]

for table in tables_for_rounding:
    numeric_columns = table.select_dtypes(include=["float"]).columns
    table[numeric_columns] = table[numeric_columns].round(2)

sales_listings.to_csv(powerbi_data_dir / "sales_listings.csv", index=False)
rentals_listings.to_csv(powerbi_data_dir / "rentals_listings.csv", index=False)
district_monthly_powerbi.to_csv(powerbi_data_dir / "district_monthly.csv", index=False)
metro.to_csv(powerbi_data_dir / "metro_stations.csv", index=False)

district_summary.to_csv(powerbi_data_dir / "district_summary.csv", index=False)
okrug_summary.to_csv(powerbi_data_dir / "okrug_summary.csv", index=False)
sales_by_month.to_csv(powerbi_data_dir / "sales_by_month.csv", index=False)
rentals_by_month.to_csv(powerbi_data_dir / "rentals_by_month.csv", index=False)
room_summary.to_csv(powerbi_data_dir / "room_summary.csv", index=False)
distance_summary.to_csv(powerbi_data_dir / "distance_summary.csv", index=False)
market_kpi.to_csv(powerbi_data_dir / "market_kpi.csv", index=False)

powerbi_tables_report = pd.DataFrame(
    [
        {"table_name": "sales_listings.csv", "rows": len(sales_listings), "columns": sales_listings.shape[1]},
        {"table_name": "rentals_listings.csv", "rows": len(rentals_listings), "columns": rentals_listings.shape[1]},
        {"table_name": "district_monthly.csv", "rows": len(district_monthly_powerbi),
         "columns": district_monthly_powerbi.shape[1]},
        {"table_name": "metro_stations.csv", "rows": len(metro), "columns": metro.shape[1]},
        {"table_name": "district_summary.csv", "rows": len(district_summary), "columns": district_summary.shape[1]},
        {"table_name": "okrug_summary.csv", "rows": len(okrug_summary), "columns": okrug_summary.shape[1]},
        {"table_name": "sales_by_month.csv", "rows": len(sales_by_month), "columns": sales_by_month.shape[1]},
        {"table_name": "rentals_by_month.csv", "rows": len(rentals_by_month), "columns": rentals_by_month.shape[1]},
        {"table_name": "room_summary.csv", "rows": len(room_summary), "columns": room_summary.shape[1]},
        {"table_name": "distance_summary.csv", "rows": len(distance_summary), "columns": distance_summary.shape[1]},
        {"table_name": "market_kpi.csv", "rows": len(market_kpi), "columns": market_kpi.shape[1]},
    ]
)

powerbi_tables_report.to_csv(
    reports_tables_dir / "powerbi_tables_report.csv",
    index=False,
)

print("=" * 80)
print("Power BI tables prepared")
print()
print(powerbi_tables_report)
print()
print(f"Power BI data saved to: {powerbi_data_dir}")
print(f"Report saved to: {reports_tables_dir / 'powerbi_tables_report.csv'}")
