import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import FEATURES, clean_shipments


def main():
    # Load the modelling table and Part C model configuration
    data = pd.read_csv("data/modelling_table.csv")
    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]

    X = data[FEATURES]
    y = data["is_delayed"]

    # Rebuild the exact Part C stratified train/validation/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # Read model settings
    selector_k = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])
    max_depth_value = config["max_depth"]
    max_depth = None if str(max_depth_value).lower() == "unlimited" else int(max_depth_value)
    min_samples_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    # Rebuild and fit the final Part C pipeline
    pipeline = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=selector_k)),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Create output folders
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Save the trained pipeline
    joblib.dump(pipeline, "models/delay_model.pkl")

    # Save the classification threshold
    pd.DataFrame({"threshold": [threshold]}).to_csv(
        "models/delay_threshold.csv", index=False
    )

    # Save the exact feature order
    pd.DataFrame({"feature": FEATURES}).to_csv(
        "models/feature_order.csv", index=False
    )

    # Calculate and save cleaning medians from the raw historical data
    raw = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw)

    impute_values = pd.DataFrame({
        "column": list(medians.keys()),
        "median": list(medians.values())
    })
    impute_values.to_csv("models/impute_values.csv", index=False)

    # Reload the saved pipeline and compare test predictions
    loaded_pipeline = joblib.load("models/delay_model.pkl")

    original_probabilities = pipeline.predict_proba(X_test)[:, 1]
    reloaded_probabilities = loaded_pipeline.predict_proba(X_test)[:, 1]

    max_diff = np.max(
        np.abs(original_probabilities - reloaded_probabilities)
    )

    reload_check = pd.DataFrame({
        "metric": [
            "test_rows_scored",
            "reload_max_abs_probability_diff",
            "threshold"
        ],
        "value": [
            len(X_test),
            round(float(max_diff), 10),
            threshold
        ]
    })

    reload_check.to_csv(
        "outputs/part_d_reload_check.csv", index=False
    )

    print("Model packaging complete.")
    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Test rows scored: {len(X_test)}")
    print(f"Reload maximum probability difference: {max_diff:.12f}")
    print(f"Threshold: {threshold}")
    print("\nSaved model files:")
    print("models/delay_model.pkl")
    print("models/delay_threshold.csv")
    print("models/feature_order.csv")
    print("models/impute_values.csv")


if __name__ == "__main__":
    main()