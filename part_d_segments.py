import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from itertools import combinations
import numpy as np


def main():
    # Read the historical modelling table
    df = pd.read_csv("data/modelling_table.csv")

    # Calculate the operational profile for each hub
    hub_profile = (
        df.groupby("hub_id")
        .agg(
            shipments=("shipment_id", "size"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean")
        )
        .reset_index()
        .sort_values("hub_id")
    )

    # Round decimal values to four places
    numeric_columns = hub_profile.select_dtypes(
        include="number"
    ).columns

    hub_profile[numeric_columns] = (
        hub_profile[numeric_columns].round(4)
    )

    # Save the output
    os.makedirs("outputs", exist_ok=True)

    hub_profile.to_csv(
        "outputs/hub_profile.csv",
        index=False
    )

    print("Hub profile created successfully.")
    print(f"Total hubs: {len(hub_profile)}")
    print("\nFirst five hubs:")
    print(hub_profile.head().to_string(index=False))
    
        # Read the saved, rounded hub profile
    profile = pd.read_csv("outputs/hub_profile.csv")

    # Select the four clustering features
    cluster_features = [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops"
    ]

    X = profile[cluster_features]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit K-Means with exactly three clusters
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_scaled)

    # Display cluster sizes
    cluster_sizes = pd.Series(cluster_labels).value_counts()

    print("\nK-Means segmentation results:")
    print("Number of clusters:", len(cluster_sizes))
    print("Cluster sizes:", sorted(
        cluster_sizes.tolist(),
        reverse=True
    ))
    
        # Evaluate clustering separation for K = 2, 3, 4, 5
    separation_results = []

    for k in [2, 3, 4, 5]:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        within_distances = []
        across_distances = []

        # Calculate distances for all 276 unordered hub pairs
        for i, j in combinations(range(len(X_scaled)), 2):

            distance = np.linalg.norm(
                X_scaled[i] - X_scaled[j]
            )

            if labels[i] == labels[j]:
                within_distances.append(distance)
            else:
                across_distances.append(distance)

        ratio = (
            np.mean(across_distances)
            / np.mean(within_distances)
        )

        cluster_sizes = sorted(
            pd.Series(labels).value_counts().tolist(),
            reverse=True
        )

        separation_results.append({
            "K": k,
            "across_over_within": round(ratio, 4),
            "cluster_sizes": "-".join(
                map(str, cluster_sizes)
            )
        })

    # Save separation evidence
    separation_df = pd.DataFrame(separation_results)

    separation_df.to_csv(
        "outputs/part_d_separation.csv",
        index=False
    )

    print("\nSeparation evidence:")
    print(separation_df.to_string(index=False))
    
    
        # Assign K-Means cluster labels to the hub profile
    profile["cluster"] = cluster_labels

    # Calculate average delay rate for each cluster
    cluster_delay_rates = (
        profile.groupby("cluster")["delay_rate"]
        .mean()
        .sort_values()
    )

    # Assign operational segment names by delay rate
    segment_names = {
        cluster_delay_rates.index[0]: "Standard",
        cluster_delay_rates.index[1]: "Watch",
        cluster_delay_rates.index[2]: "Escalate"
    }

    profile["segment"] = profile["cluster"].map(segment_names)

    # Attach hub city and tier information
    hubs = pd.read_csv("data/hubs.csv")

    hub_segments = profile[
        ["hub_id", "segment"]
    ].merge(
        hubs[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left",
        validate="one_to_one"
    )

    hub_segments = hub_segments[
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ].sort_values("hub_id")

    # Save hub segments
    hub_segments.to_csv(
        "outputs/hub_segments.csv",
        index=False
    )

    # Create segment-level profiles
    segment_profile = (
        profile.groupby("segment")
        .agg(
            hubs=("hub_id", "count"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean")
        )
        .reset_index()
    )

    segment_order = ["Standard", "Watch", "Escalate"]

    segment_profile["segment"] = pd.Categorical(
        segment_profile["segment"],
        categories=segment_order,
        ordered=True
    )

    segment_profile = (
        segment_profile.sort_values("segment")
        .reset_index(drop=True)
    )

    numeric_columns = segment_profile.select_dtypes(
        include="number"
    ).columns

    segment_profile[numeric_columns] = (
        segment_profile[numeric_columns].round(4)
    )

    segment_profile["segment"] = (
        segment_profile["segment"].astype(str)
    )

    segment_profile.to_csv(
        "outputs/segment_profile.csv",
        index=False
    )

    # Identify tier-versus-segment mismatches
    expected_segments = {
        1: "Standard",
        2: "Watch",
        3: "Escalate"
    }

    hub_segments["expected_segment"] = (
        hub_segments["hub_tier"].map(expected_segments)
    )

    mismatches = hub_segments[
        hub_segments["segment"] != hub_segments["expected_segment"]
    ][
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ]

    mismatches.to_csv(
        "outputs/tier_vs_segment_mismatch.csv",
        index=False
    )

    # Display results
    print("\nHub segments:")
    print(hub_segments.to_string(index=False))

    print("\nSegment profiles:")
    print(segment_profile.to_string(index=False))

    print("\nTier-versus-segment mismatches:")
    print(mismatches.to_string(index=False))


if __name__ == "__main__":
    main()
