import joblib
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


def main():
    # Load new shipments and hub reference data
    raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # Load saved model and supporting files
    model = joblib.load("models/delay_model.pkl")

    threshold = float(
        pd.read_csv("models/delay_threshold.csv").iloc[0]["threshold"]
    )

    feature_order = pd.read_csv(
        "models/feature_order.csv"
    )["feature"].tolist()

    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median"]))

    hub_segments = pd.read_csv("outputs/hub_segments.csv")

    # Clean new shipments using the stored historical medians
    clean, _ = clean_shipments(raw, medians=medians)

    # Build features and preserve the saved feature order
    features = build_features(clean, hubs)
    X_new = features[feature_order]

    # Predict delay probabilities
    probabilities = model.predict_proba(X_new)[:, 1]
    delay_probabilities = pd.Series(probabilities).round(4)

    # Flag shipments using the stored threshold
    flagged = (delay_probabilities >= threshold).astype(int)

    # Assemble predictions in the original shipment order
    predictions = pd.DataFrame({
        "shipment_id": features["shipment_id"].values,
        "hub_id": features["hub_id"].values,
        "delay_probability": delay_probabilities.values,
        "flagged": flagged.values
    })

    # Attach operational hub segments
    predictions = predictions.merge(
        hub_segments[["hub_id", "segment"]],
        on="hub_id",
        how="left",
        sort=False,
        validate="many_to_one"
    )

    # Assign the recommended action
    predictions["action"] = "No action"

    flagged_rows = predictions["flagged"] == 1
    escalate_rows = flagged_rows & (predictions["segment"] == "Escalate")
    call_rows = flagged_rows & (predictions["segment"] != "Escalate")

    predictions.loc[escalate_rows, "action"] = "Escalate to hub manager"
    predictions.loc[call_rows, "action"] = "Proactive customer call"

    # Save the final action file
    predictions = predictions[
        [
            "shipment_id",
            "hub_id",
            "segment",
            "delay_probability",
            "flagged",
            "action"
        ]
    ]

    predictions.to_csv(
        "outputs/new_shipment_actions.csv",
        index=False
    )

    print("New shipment scoring complete.")
    print(f"Shipments scored: {len(predictions)}")
    print(f"Flagged shipments: {predictions['flagged'].sum()}")
    print(
        "Escalations: "
        f"{(predictions['action'] == 'Escalate to hub manager').sum()}"
    )
    print(
        "Customer calls: "
        f"{(predictions['action'] == 'Proactive customer call').sum()}"
    )
    print("\nFirst 10 predictions:")
    print(predictions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()