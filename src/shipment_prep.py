
import numpy as np
import pandas as pd


# Promise hours for each delivery service
PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}


# Exact feature order required by the capstone
FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


# Bookkeeping columns that are expected to carry no useful signal
NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


def clean_shipments(raw, medians=None):
    """
    Clean raw shipment data.

    Parameters
    ----------
    raw : pandas.DataFrame
        Raw shipment dataset.

    medians : dict or None
        Previously calculated median values.

    Returns
    -------
    clean_df : pandas.DataFrame
        Cleaned shipment dataset.

    medians : dict
        Median values used for filling missing data.
    """

    # Work on a copy to avoid modifying the original DataFrame
    df = raw.copy()

    # Step 1: Remove exact duplicate rows
    df = df.drop_duplicates().copy()

    # Step 2: Standardize service type
    df["service_type"] = (
        df["service_type"]
        .str.strip()
        .str.title()
    )

    # Step 3: Handle missing payment modes
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # Step 4: Convert declared value to numeric INR
    df["declared_value_inr"] = (
        df["declared_value"]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    df = df.drop(columns=["declared_value"])

    # Step 5: Convert dispatch time to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # Step 6: Treat scanner-fault weights as missing
    df.loc[
        df["parcel_weight_kg"] > 100,
        "parcel_weight_kg"
    ] = np.nan

    # Step 7: Calculate medians or use previously supplied medians
    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr"
    ]

    if medians is None:
        medians = {
            col: float(df[col].median())
            for col in median_columns
        }

    # Fill missing values using the selected medians
    for col in median_columns:
        df[col] = df[col].fillna(medians[col])

    return df, medians
    
    

def build_features(clean, hubs):
    """
    Attach hub information and create model-ready features.

    Parameters
    ----------
    clean : pandas.DataFrame
        Cleaned shipment dataset.

    hubs : pandas.DataFrame
        Hub master dataset.

    Returns
    -------
    pandas.DataFrame
        Shipment dataset with engineered features and hub information.
    """

    # Work on a copy
    df = clean.copy()

    # Step 1: Attach hub information using a left merge
    hub_columns = [
        "hub_id",
        "hub_city",
        "region",
        "hub_tier",
        "hub_daily_capacity"
    ]

    df = df.merge(
        hubs[hub_columns],
        on="hub_id",
        how="left",
        validate="many_to_one"
    )

    # Step 2: Calculate promised delivery hours
    df["promised_hours"] = (
        df["service_type"].map(PROMISED_HOURS)
    )

    # Step 3: Identify Cash on Delivery shipments
    df["is_cod"] = (
        df["payment_mode"] == "COD"
    ).astype(int)

    # Step 4: Identify weekend dispatches
    df["is_weekend"] = (
        df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    # Step 5: Extract dispatch hour
    df["dispatch_hour"] = (
        df["dispatch_time"].dt.hour
    ).astype(int)

    # Step 6: Log-transform declared value
    df["log_declared_value"] = np.log(
        df["declared_value_inr"]
    )

    return df

