from pathlib import Path

import pandas as pd

project_dir = Path(__file__).resolve().parents[1]

raw_data_dir = project_dir / "data" / "raw"
processed_data_dir = project_dir / "data" / "processed"
reports_tables_dir = project_dir / "reports" / "tables"

processed_data_dir.mkdir(parents=True, exist_ok=True)
reports_tables_dir.mkdir(parents=True, exist_ok=True)

secondary = pd.read_csv(raw_data_dir / "secondary_market.csv")
rentals = pd.read_csv(raw_data_dir / "rentals.csv")
new_builds = pd.read_csv(raw_data_dir / "new_builds.csv")
metro = pd.read_csv(raw_data_dir / "metro_stations.csv")
district_monthly = pd.read_csv(raw_data_dir / "district_prices_monthly.csv")

print("=" * 80)
print("Raw data loaded")
print("Secondary market:", secondary.shape)
print("Rentals:", rentals.shape)
print("New builds:", new_builds.shape)
print("Metro stations:", metro.shape)
print("District monthly prices:", district_monthly.shape)

secondary = secondary.drop_duplicates()
rentals = rentals.drop_duplicates()
new_builds = new_builds.drop_duplicates()
metro = metro.drop_duplicates()
district_monthly = district_monthly.drop_duplicates()

secondary["date_posted"] = pd.to_datetime(secondary["date_posted"], errors="coerce")
rentals["date_posted"] = pd.to_datetime(rentals["date_posted"], errors="coerce")
new_builds["date_posted"] = pd.to_datetime(new_builds["date_posted"], errors="coerce")
district_monthly["year_month"] = pd.to_datetime(district_monthly["year_month"], errors="coerce")

secondary = secondary.dropna(subset=["date_posted"])
rentals = rentals.dropna(subset=["date_posted"])
new_builds = new_builds.dropna(subset=["date_posted"])
district_monthly = district_monthly.dropna(subset=["year_month"])

secondary = secondary[
    (secondary["price_rub"] > 0)
    & (secondary["price_per_sqm"] > 0)
    & (secondary["total_area"] > 0)
    & (secondary["rooms"] >= 0)
    & (secondary["floor"] > 0)
    & (secondary["total_floors"] > 0)
    ]

rentals = rentals[
    (rentals["monthly_rent_rub"] > 0)
    & (rentals["rent_per_sqm"] > 0)
    & (rentals["total_area"] > 0)
    & (rentals["rooms"] >= 0)
    & (rentals["floor"] > 0)
    & (rentals["total_floors"] > 0)
    ]

new_builds = new_builds[
    (new_builds["price_rub"] > 0)
    & (new_builds["price_per_sqm"] > 0)
    & (new_builds["total_area"] > 0)
    & (new_builds["rooms"] >= 0)
    & (new_builds["floor"] > 0)
    & (new_builds["total_floors"] > 0)
    ]

secondary = secondary[secondary["floor"] <= secondary["total_floors"]]
rentals = rentals[rentals["floor"] <= rentals["total_floors"]]
new_builds = new_builds[new_builds["floor"] <= new_builds["total_floors"]]

secondary_price_low = secondary["price_rub"].quantile(0.01)
secondary_price_high = secondary["price_rub"].quantile(0.99)
secondary = secondary[
    secondary["price_rub"].between(secondary_price_low, secondary_price_high)
]

rent_low = rentals["monthly_rent_rub"].quantile(0.01)
rent_high = rentals["monthly_rent_rub"].quantile(0.99)
rentals = rentals[
    rentals["monthly_rent_rub"].between(rent_low, rent_high)
]

new_builds_price_low = new_builds["price_rub"].quantile(0.01)
new_builds_price_high = new_builds["price_rub"].quantile(0.99)
new_builds = new_builds[
    new_builds["price_rub"].between(new_builds_price_low, new_builds_price_high)
]

secondary["listing_year"] = secondary["date_posted"].dt.year
secondary["listing_month"] = secondary["date_posted"].dt.month
secondary["listing_quarter"] = secondary["date_posted"].dt.quarter

rentals["listing_year"] = rentals["date_posted"].dt.year
rentals["listing_month"] = rentals["date_posted"].dt.month
rentals["listing_quarter"] = rentals["date_posted"].dt.quarter

new_builds["listing_year"] = new_builds["date_posted"].dt.year
new_builds["listing_month"] = new_builds["date_posted"].dt.month
new_builds["listing_quarter"] = new_builds["date_posted"].dt.quarter

secondary["price_mln"] = secondary["price_rub"] / 1_000_000
new_builds["price_mln"] = new_builds["price_rub"] / 1_000_000
rentals["monthly_rent_thousand"] = rentals["monthly_rent_rub"] / 1_000

secondary["building_age"] = secondary["listing_year"] - secondary["building_year"]
rentals["building_age"] = rentals["listing_year"] - rentals["building_year"]

secondary["building_age"] = secondary["building_age"].clip(lower=0)
rentals["building_age"] = rentals["building_age"].clip(lower=0)

secondary["is_first_floor"] = secondary["floor"] == 1
secondary["is_last_floor"] = secondary["floor"] == secondary["total_floors"]
secondary["floor_ratio"] = secondary["floor"] / secondary["total_floors"]

rentals["is_first_floor"] = rentals["floor"] == 1
rentals["is_last_floor"] = rentals["floor"] == rentals["total_floors"]
rentals["floor_ratio"] = rentals["floor"] / rentals["total_floors"]

new_builds["is_first_floor"] = new_builds["floor"] == 1
new_builds["is_last_floor"] = new_builds["floor"] == new_builds["total_floors"]
new_builds["floor_ratio"] = new_builds["floor"] / new_builds["total_floors"]

secondary["living_area_share"] = secondary["living_area"] / secondary["total_area"]
secondary["kitchen_area_share"] = secondary["kitchen_area"] / secondary["total_area"]

secondary["living_area_share"] = secondary["living_area_share"].clip(0, 1)
secondary["kitchen_area_share"] = secondary["kitchen_area_share"].clip(0, 1)

secondary["has_balcony_int"] = secondary["has_balcony"].astype(int)
rentals["furnished_int"] = rentals["furnished"].astype(int)
rentals["pets_allowed_int"] = rentals["pets_allowed"].astype(int)
new_builds["subsidized_mortgage_int"] = new_builds["subsidized_mortgage"].astype(int)

rentals["furnished_status"] = rentals["furnished_int"].map({0: "No", 1: "Yes"})
rentals["pets_allowed_status"] = rentals["pets_allowed_int"].map({0: "No", 1: "Yes"})

secondary["metro_is_walk"] = secondary["metro_distance_type"].eq("walk")

secondary["distance_to_center_group"] = pd.cut(
    secondary["to_center_km"],
    bins=[0, 5, 10, 15, 25, 100],
    labels=["0-5 km", "5-10 km", "10-15 km", "15-25 km", "25+ km"],
    include_lowest=True,
)

rentals["distance_to_center_group"] = pd.cut(
    rentals["to_center_km"],
    bins=[0, 5, 10, 15, 25, 100],
    labels=["0-5 km", "5-10 km", "10-15 km", "15-25 km", "25+ km"],
    include_lowest=True,
)

new_builds["distance_to_center_group"] = pd.cut(
    new_builds["to_center_km"],
    bins=[0, 5, 10, 15, 25, 100],
    labels=["0-5 km", "5-10 km", "10-15 km", "15-25 km", "25+ km"],
    include_lowest=True,
)

rooms_group_map = {
    0: "Studio",
    1: "1 room",
    2: "2 rooms",
    3: "3 rooms",
    4: "4 rooms",
    5: "5 rooms",
}

secondary["rooms_group"] = secondary["rooms"].map(rooms_group_map)
rentals["rooms_group"] = rentals["rooms"].map(rooms_group_map)
new_builds["rooms_group"] = new_builds["rooms"].map(rooms_group_map)

secondary["rooms_group"] = secondary["rooms_group"].fillna(
    secondary["rooms"].astype(str) + " rooms"
)

rentals["rooms_group"] = rentals["rooms_group"].fillna(
    rentals["rooms"].astype(str) + " rooms"
)

new_builds["rooms_group"] = new_builds["rooms_group"].fillna(
    new_builds["rooms"].astype(str) + " rooms"
)

rentals["rent_price_group"] = pd.cut(
    rentals["monthly_rent_thousand"],
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

secondary["sale_price_group"] = pd.cut(
    secondary["price_mln"],
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

new_builds["sale_price_group"] = pd.cut(
    new_builds["price_mln"],
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

secondary["market_type"] = "Secondary Market"
new_builds["market_type"] = "New Builds"

sales_common_columns = [
    "id",
    "date_posted",
    "listing_year",
    "listing_month",
    "listing_quarter",
    "market_type",
    "district",
    "okrug",
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

secondary_sales = secondary[sales_common_columns].copy()
new_builds_sales = new_builds[sales_common_columns].copy()

sales = pd.concat(
    [secondary_sales, new_builds_sales],
    ignore_index=True,
)

district_monthly["year"] = district_monthly["year_month"].dt.year
district_monthly["month"] = district_monthly["year_month"].dt.month
district_monthly["quarter"] = district_monthly["year_month"].dt.quarter
district_monthly["year_month_text"] = district_monthly["year_month"].dt.to_period("M").astype(str)

secondary.to_csv(processed_data_dir / "secondary_clean.csv", index=False)
rentals.to_csv(processed_data_dir / "rentals_clean.csv", index=False)
new_builds.to_csv(processed_data_dir / "new_builds_clean.csv", index=False)
metro.to_csv(processed_data_dir / "metro_stations_clean.csv", index=False)
district_monthly.to_csv(processed_data_dir / "district_monthly_clean.csv", index=False)
sales.to_csv(processed_data_dir / "sales_clean.csv", index=False)

cleaning_report = pd.DataFrame(
    [
        {
            "dataset": "secondary_clean",
            "rows": len(secondary),
            "columns": secondary.shape[1],
            "missing_values": secondary.isna().sum().sum(),
        },
        {
            "dataset": "rentals_clean",
            "rows": len(rentals),
            "columns": rentals.shape[1],
            "missing_values": rentals.isna().sum().sum(),
        },
        {
            "dataset": "new_builds_clean",
            "rows": len(new_builds),
            "columns": new_builds.shape[1],
            "missing_values": new_builds.isna().sum().sum(),
        },
        {
            "dataset": "district_monthly_clean",
            "rows": len(district_monthly),
            "columns": district_monthly.shape[1],
            "missing_values": district_monthly.isna().sum().sum(),
        },
        {
            "dataset": "sales_clean",
            "rows": len(sales),
            "columns": sales.shape[1],
            "missing_values": sales.isna().sum().sum(),
        },
    ]
)

cleaning_report.to_csv(
    reports_tables_dir / "cleaning_report.csv",
    index=False,
)

print("=" * 80)
print("Data cleaning and feature engineering completed")
print()
print(cleaning_report)
print()
print(f"Processed data saved to: {processed_data_dir}")
print(f"Cleaning report saved to: {reports_tables_dir / 'cleaning_report.csv'}")
