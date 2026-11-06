
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

print("\nModelling table saved successfully.")
print("Modelling table shape:", modelling_df.shape)
