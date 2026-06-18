from pathlib import Path

import pandas as pd

project_dir = Path(__file__).resolve().parents[1]

raw_data_dir = project_dir / "data" / "raw"
reports_tables_dir = project_dir / "reports" / "tables"

reports_tables_dir.mkdir(parents=True, exist_ok=True)

files = {
    "secondary_market": "secondary_market.csv",
    "rentals": "rentals.csv",
    "new_builds": "new_builds.csv",
    "metro_stations": "metro_stations.csv",
    "district_prices_monthly": "district_prices_monthly.csv",
}

overview_rows = []

for dataset_name, file_name in files.items():
    file_path = raw_data_dir / file_name

    if not file_path.exists():
        print(f"File not found: {file_path}")
        print("Check that the CSV file is located in data/raw/")
        continue

    df = pd.read_csv(file_path)

    rows_count = df.shape[0]
    columns_count = df.shape[1]
    duplicates_count = df.duplicated().sum()
    missing_values_count = df.isna().sum().sum()

    overview_rows.append(
        {
            "dataset": dataset_name,
            "file_name": file_name,
            "rows": rows_count,
            "columns": columns_count,
            "duplicates": duplicates_count,
            "missing_values_total": missing_values_count,
        }
    )

    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"File: {file_name}")
    print(f"Shape: {rows_count} rows, {columns_count} columns")
    print(f"Duplicates: {duplicates_count}")
    print(f"Total missing values: {missing_values_count}")
    print()

    print("Columns:")
    print(df.columns.tolist())
    print()

    print("Data types:")
    print(df.dtypes)
    print()

    print("Top 15 columns by missing values:")
    print(df.isna().sum().sort_values(ascending=False).head(15))
    print()

    print("First 5 rows:")
    print(df.head())
    print()

overview = pd.DataFrame(overview_rows)

output_path = reports_tables_dir / "dataset_overview.csv"
overview.to_csv(output_path, index=False)

print("=" * 80)
print("Dataset overview")
print(overview)
print()
print(f"Report saved to: {output_path}")
