
import os
import pandas as pd
import numpy as np

from src.shipment_prep import (
    PROMISED_HOURS,
    FEATURES,
    clean_shipments,
    build_features
)


# Create output directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# Load source datasets
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")


print("Raw shipment rows:", len(raw))
print("Hub rows:", len(hubs))

# Capture raw-data checks before cleaning
raw_rows = len(raw)

raw_dedup = raw.drop_duplicates()

duplicate_rows_removed = raw_rows - len(raw_dedup)

missing_distance_before = raw_dedup["distance_km"].isna().sum()

missing_weight_before = raw_dedup["parcel_weight_kg"].isna().sum()

sentinel_weight_rows = (
    raw_dedup["parcel_weight_kg"] > 100
).sum()

missing_declared_value_before = (
    raw_dedup["declared_value"].isna().sum()
)

missing_payment_mode_before = (
    raw_dedup["payment_mode"].isna().sum()
)

service_spellings_before = raw_dedup["service_type"].nunique()


# Clean shipment data
clean_df, medians = clean_shipments(raw)

print("Rows after cleaning:", len(clean_df))
print("Imputation medians:", medians)


# Build features
modelling_df = build_features(clean_df, hubs)


# Create target variable
modelling_df["is_delayed"] = (
    modelling_df["delivery_hours"]
    > modelling_df["promised_hours"]
).astype(int)


# Save modelling table
modelling_df.to_csv(
    "data/modelling_table.csv",
    index=False
)

# --------------------------------------------------
# 5. Data quality checks
# --------------------------------------------------

print("\n--- DATA QUALITY CHECKS ---")

print("Duplicate rows remaining:", modelling_df.duplicated().sum())

print(
    "Missing values in modelling features:",
    modelling_df[FEATURES].isnull().sum().sum()
)

print(
    "Missing delivery_hours:",
    modelling_df["delivery_hours"].isnull().sum()
)

print(
    "Missing is_delayed:",
    modelling_df["is_delayed"].isnull().sum()
)

print(
    "Delayed shipments:",
    modelling_df["is_delayed"].sum()
)

print(
    "Overall delay rate:",
    round(modelling_df["is_delayed"].mean(), 4)
)

# --------------------------------------------------
# 6. Delay analysis by service type
# --------------------------------------------------

print("\n--- DELAY ANALYSIS BY SERVICE TYPE ---")

service_delay = (
    modelling_df
    .groupby("service_type")
    .agg(
        total_shipments=("is_delayed", "count"),
        delayed_shipments=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean")
    )
    .reset_index()
)

service_delay["delay_rate"] = service_delay["delay_rate"].round(4)

print(service_delay)

service_delay.to_csv(
    "outputs/delay_by_service.csv",
    index=False
)

# --------------------------------------------------
# 7. Delay analysis by hub tier
# --------------------------------------------------

print("\n--- DELAY ANALYSIS BY HUB TIER ---")

hub_tier_delay = (
    modelling_df
    .groupby("hub_tier")
    .agg(
        total_shipments=("is_delayed", "count"),
        delayed_shipments=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean")
    )
    .reset_index()
)

hub_tier_delay["delay_rate"] = hub_tier_delay["delay_rate"].round(4)

print(hub_tier_delay)

hub_tier_delay.to_csv(
    "outputs/delay_by_hub_tier.csv",
    index=False
)

# --------------------------------------------------
# 8. Delay analysis by distance band
# --------------------------------------------------

print("\n--- DELAY ANALYSIS BY DISTANCE BAND ---")

distance_bins = [0, 200, 400, 600, 2000]
distance_labels = ["0-200", "201-400", "401-600", "601+"]

modelling_df["distance_band"] = pd.cut(
    modelling_df["distance_km"],
    bins=distance_bins,
    labels=distance_labels,
    right=True
)

distance_delay = (
    modelling_df
    .groupby("distance_band", observed=False)
    .agg(
        total_shipments=("is_delayed", "count"),
        delayed_shipments=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean")
    )
    .reset_index()
)

distance_delay["delay_rate"] = distance_delay["delay_rate"].round(4)

print(distance_delay)

distance_delay.to_csv(
    "outputs/delay_by_distance_band.csv",
    index=False
)

# --------------------------------------------------
# 9. Correlation analysis
# --------------------------------------------------

print("\n--- CORRELATION WITH DELIVERY HOURS ---")

correlation = (
    modelling_df[FEATURES + ["delivery_hours"]]
    .corr(numeric_only=True)["delivery_hours"]
    .drop("delivery_hours")
    .sort_values(key=abs, ascending=False)
)

print(correlation.round(4))

correlation.to_csv(
    "outputs/correlation_with_delivery_hours.csv",
    header=["correlation"]
)

print(
    "\nCorrelation between is_delayed and distance_km:",
    round(modelling_df["is_delayed"].corr(modelling_df["distance_km"]), 4)
)

# --------------------------------------------------
# 10. Outlier analysis for delivery hours
# --------------------------------------------------

print("\n--- DELIVERY HOURS OUTLIER ANALYSIS ---")

q1 = modelling_df["delivery_hours"].quantile(0.25)
q3 = modelling_df["delivery_hours"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

outliers = modelling_df[
    (modelling_df["delivery_hours"] < lower_bound) |
    (modelling_df["delivery_hours"] > upper_bound)
]

print("Q1:", round(q1, 2))
print("Q3:", round(q3, 2))
print("IQR:", round(iqr, 2))
print("Lower bound:", round(lower_bound, 2))
print("Upper bound:", round(upper_bound, 2))
print("Number of outliers:", len(outliers))

# --------------------------------------------------
# 11. Declared value distribution analysis
# --------------------------------------------------

print("\n--- DECLARED VALUE ANALYSIS ---")

print(
    "Declared value mean:",
    round(modelling_df["declared_value_inr"].mean(), 2)
)

print(
    "Declared value median:",
    round(modelling_df["declared_value_inr"].median(), 2)
)

print(
    "Log declared value mean:",
    round(modelling_df["log_declared_value"].mean(), 3)
)

print(
    "Log declared value median:",
    round(modelling_df["log_declared_value"].median(), 3)
)

# --------------------------------------------------
# 12. Save required Part A checks
# --------------------------------------------------

checks = {
    "raw_rows": raw_rows,
    "duplicate_rows_removed": duplicate_rows_removed,
    "rows_after_dedup": len(raw_dedup),
    "missing_distance_km_before": missing_distance_before,
    "missing_parcel_weight_before": missing_weight_before,
    "sentinel_weight_rows": sentinel_weight_rows,
    "missing_declared_value_before": missing_declared_value_before,
    "missing_payment_mode_before": missing_payment_mode_before,
    "service_type_spellings_before": service_spellings_before,
    "service_type_spellings_after": modelling_df["service_type"].nunique(),
    "median_distance_km": round(medians["distance_km"], 2),
    "median_parcel_weight_kg": round(medians["parcel_weight_kg"], 2),
    "median_declared_value_inr": round(medians["declared_value_inr"], 2),
    "rows_after_hub_merge": len(modelling_df),
    "nulls_in_features_and_target": modelling_df[
        FEATURES + ["delivery_hours", "is_delayed"]
    ].isnull().sum().sum(),
    "delayed_count": int(modelling_df["is_delayed"].sum()),
    "delay_rate": round(modelling_df["is_delayed"].mean(), 4),
    "hours_q1": round(q1, 2),
    "hours_q3": round(q3, 2),
    "hours_lower_bound": round(lower_bound, 2),
    "hours_upper_bound": round(upper_bound, 2),
    "hours_outlier_count": len(outliers),
    "declared_value_mean_inr": round(
        modelling_df["declared_value_inr"].mean(), 2
    ),
    "declared_value_median_inr": round(
        modelling_df["declared_value_inr"].median(), 2
    ),
    "log_declared_value_mean": round(
        modelling_df["log_declared_value"].mean(), 3
    ),
    "log_declared_value_median": round(
        modelling_df["log_declared_value"].median(), 3
    ),
}

pd.DataFrame(
    list(checks.items()),
    columns=["metric", "value"]
).to_csv("outputs/part_a_checks.csv", index=False)


# --------------------------------------------------
# 13. Save combined Part A delay-by-group evidence
# --------------------------------------------------

group_results = []

for dimension in ["service_type", "hub_tier", "distance_band"]:
    grouped = modelling_df.groupby(
        dimension,
        observed=False
    ).agg(
        shipments=("is_delayed", "count"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index()

    grouped = grouped.rename(columns={dimension: "group"})
    grouped.insert(0, "dimension", dimension)

    group_results.append(grouped)

part_a_delay_by_group = pd.concat(
    group_results,
    ignore_index=True
)

part_a_delay_by_group["delay_rate"] = (
    part_a_delay_by_group["delay_rate"].round(4)
)

part_a_delay_by_group["mean_hours"] = (
    part_a_delay_by_group["mean_hours"].round(2)
)

part_a_delay_by_group.to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False
)

# --------------------------------------------------
# 14. Save full Part A correlation matrix
# --------------------------------------------------

numeric_columns = modelling_df.select_dtypes(
    include="number"
).columns

correlation_matrix = modelling_df[
    numeric_columns
].corr()

correlation_matrix.to_csv(
    "outputs/part_a_correlations.csv",
    index=True
)

print("\nPart A checks saved successfully.")

print("\nModelling table saved successfully.")
print("Modelling table shape:", modelling_df.shape)
