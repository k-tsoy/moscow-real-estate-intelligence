from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

project_dir = Path(__file__).resolve().parents[1]

raw_data_dir = project_dir / "data" / "raw"
reports_dir = project_dir / "reports"
figures_dir = reports_dir / "figures"
tables_dir = reports_dir / "tables"

figures_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)

secondary = pd.read_csv(raw_data_dir / "secondary_market.csv")
rentals = pd.read_csv(raw_data_dir / "rentals.csv")
new_builds = pd.read_csv(raw_data_dir / "new_builds.csv")
metro = pd.read_csv(raw_data_dir / "metro_stations.csv")
district_monthly = pd.read_csv(raw_data_dir / "district_prices_monthly.csv")

secondary["date_posted"] = pd.to_datetime(secondary["date_posted"], errors="coerce")
rentals["date_posted"] = pd.to_datetime(rentals["date_posted"], errors="coerce")
new_builds["date_posted"] = pd.to_datetime(new_builds["date_posted"], errors="coerce")
district_monthly["year_month"] = pd.to_datetime(district_monthly["year_month"], errors="coerce")

print("=" * 80)
print("Data loaded successfully")
print()
print("Secondary market:", secondary.shape)
print("Rentals:", rentals.shape)
print("New builds:", new_builds.shape)
print("Metro stations:", metro.shape)
print("District monthly prices:", district_monthly.shape)

datasets = {
    "secondary_market": secondary,
    "rentals": rentals,
    "new_builds": new_builds,
    "metro_stations": metro,
    "district_prices_monthly": district_monthly,
}

missing_rows = []

for dataset_name, df in datasets.items():
    for column in df.columns:
        missing_count = df[column].isna().sum()
        missing_percent = missing_count / len(df) * 100

        missing_rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "missing_count": missing_count,
                "missing_percent": round(missing_percent, 2),
            }
        )

missing_report = pd.DataFrame(missing_rows)
missing_report = missing_report.sort_values(
    by=["missing_percent", "missing_count"],
    ascending=False,
)

missing_report.to_csv(
    tables_dir / "missing_values_report.csv",
    index=False,
)

secondary.describe(include="all").to_csv(tables_dir / "secondary_market_summary.csv")
rentals.describe(include="all").to_csv(tables_dir / "rentals_summary.csv")
new_builds.describe(include="all").to_csv(tables_dir / "new_builds_summary.csv")

secondary_by_district = (
    secondary
    .groupby(["okrug", "district"], as_index=False)
    .agg(
        listings_count=("id", "count"),
        avg_price_rub=("price_rub", "mean"),
        median_price_rub=("price_rub", "median"),
        avg_price_per_sqm=("price_per_sqm", "mean"),
        median_price_per_sqm=("price_per_sqm", "median"),
        avg_area=("total_area", "mean"),
    )
    .sort_values("avg_price_per_sqm", ascending=False)
)

secondary_by_district.to_csv(
    tables_dir / "secondary_by_district.csv",
    index=False,
)

rentals_by_district = (
    rentals
    .groupby(["okrug", "district"], as_index=False)
    .agg(
        listings_count=("id", "count"),
        avg_monthly_rent=("monthly_rent_rub", "mean"),
        median_monthly_rent=("monthly_rent_rub", "median"),
        avg_rent_per_sqm=("rent_per_sqm", "mean"),
        median_rent_per_sqm=("rent_per_sqm", "median"),
        avg_area=("total_area", "mean"),
    )
    .sort_values("avg_rent_per_sqm", ascending=False)
)

rentals_by_district.to_csv(
    tables_dir / "rentals_by_district.csv",
    index=False,
)

market_by_month = (
    district_monthly
    .groupby("year_month", as_index=False)
    .agg(
        avg_secondary_price_per_sqm=("secondary_price_per_sqm", "mean"),
        avg_newbuild_price_per_sqm=("newbuild_price_per_sqm", "mean"),
        avg_rental_price_per_sqm=("rental_price_per_sqm_monthly", "mean"),
        avg_key_rate=("cbr_key_rate_pct", "mean"),
        avg_mortgage_rate=("avg_mortgage_rate_pct", "mean"),
    )
)

market_by_month.to_csv(
    tables_dir / "market_by_month.csv",
    index=False,
)

secondary_plot = secondary.copy()
secondary_price_low = secondary_plot["price_rub"].quantile(0.01)
secondary_price_high = secondary_plot["price_rub"].quantile(0.99)

secondary_plot = secondary_plot[
    secondary_plot["price_rub"].between(secondary_price_low, secondary_price_high)
]

secondary_plot["price_mln"] = secondary_plot["price_rub"] / 1_000_000

plt.figure(figsize=(10, 6))
plt.hist(secondary_plot["price_mln"], bins=50)
plt.title("Secondary Market Price Distribution")
plt.xlabel("Price, mln RUB")
plt.ylabel("Listings")
plt.tight_layout()
plt.savefig(figures_dir / "secondary_price_distribution.png", dpi=300)
plt.close()

rentals_plot = rentals.copy()
rent_low = rentals_plot["monthly_rent_rub"].quantile(0.01)
rent_high = rentals_plot["monthly_rent_rub"].quantile(0.99)

rentals_plot = rentals_plot[
    rentals_plot["monthly_rent_rub"].between(rent_low, rent_high)
]

rentals_plot["monthly_rent_thousand"] = rentals_plot["monthly_rent_rub"] / 1_000

plt.figure(figsize=(10, 6))
plt.hist(rentals_plot["monthly_rent_thousand"], bins=50)
plt.title("Monthly Rent Distribution")
plt.xlabel("Monthly rent, k RUB")
plt.ylabel("Listings")
plt.tight_layout()
plt.savefig(figures_dir / "rent_distribution.png", dpi=300)
plt.close()

top_secondary_districts = secondary_by_district.head(15).sort_values(
    "avg_price_per_sqm",
    ascending=True,
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_secondary_districts["district"],
    top_secondary_districts["avg_price_per_sqm"],
)
plt.title("Top 15 Districts by Secondary Market Price per sqm")
plt.xlabel("Avg price per sqm, RUB")
plt.ylabel("District")
plt.tight_layout()
plt.savefig(figures_dir / "top_secondary_districts.png", dpi=300)
plt.close()

top_rental_districts = rentals_by_district.head(15).sort_values(
    "avg_rent_per_sqm",
    ascending=True,
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_rental_districts["district"],
    top_rental_districts["avg_rent_per_sqm"],
)
plt.title("Top 15 Districts by Rent per sqm")
plt.xlabel("Avg rent per sqm, RUB")
plt.ylabel("District")
plt.tight_layout()
plt.savefig(figures_dir / "top_rental_districts.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(
    market_by_month["year_month"],
    market_by_month["avg_secondary_price_per_sqm"],
    label="Secondary Market",
)
plt.plot(
    market_by_month["year_month"],
    market_by_month["avg_newbuild_price_per_sqm"],
    label="New Builds",
)
plt.title("Price per sqm Dynamics")
plt.xlabel("Month")
plt.ylabel("Avg price per sqm, RUB")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(figures_dir / "sales_price_dynamics.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(
    market_by_month["year_month"],
    market_by_month["avg_rental_price_per_sqm"],
    label="Rent",
)
plt.title("Rent per sqm Dynamics")
plt.xlabel("Month")
plt.ylabel("Avg rent per sqm, RUB")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(figures_dir / "rent_price_dynamics.png", dpi=300)
plt.close()

print("=" * 80)
print("Basic EDA completed")
print()
print("Tables saved to:")
print(tables_dir)
print()
print("Figures saved to:")
print(figures_dir)
