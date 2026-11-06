
import pandas as pd

from sklearn.model_selection import train_test_split

from src.shipment_prep import FEATURES, NOISE_COLUMNS

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# ==========================================
# TASK 1: STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ==========================================

# Load the Part A modelling dataset
df = pd.read_csv("data/modelling_table.csv")

# Select the 17 model input features
X = df[FEATURES]

# Select the classification target
y = df["is_delayed"]


# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# Second split: divide temporary data equally
# into validation and test datasets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


# ==========================================
# VERIFY THE SPLIT
# ==========================================

print("\n========== PART C: TASK 1 ==========")

print("\nDataset sizes:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

print("\nDelayed shipment counts:")
print("Training:", int(y_train.sum()))
print("Validation:", int(y_val.sum()))
print("Test:", int(y_test.sum()))

print("\nClass distribution:")

for name, target in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Test", y_test)
]:
    print(f"\n{name}:")
    print(target.value_counts().sort_index())

print("\n========== SPLIT VALIDATION ==========")

print("Correct training size:", len(X_train) == 840)
print("Correct validation size:", len(X_val) == 180)
print("Correct test size:", len(X_test) == 180)

print("Correct training delayed count:", int(y_train.sum()) == 195)
print("Correct validation delayed count:", int(y_val.sum()) == 42)
print("Correct test delayed count:", int(y_test.sum()) == 41)



# ==========================================
# TASK 2: BASELINE AND THREE CLASSIFIERS
# ==========================================

# Helper function to calculate classification metrics
def evaluate_model(y_true, y_pred):

    # Extract confusion matrix values in fixed order
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        )
    }


# Store model results
results = []


# ------------------------------------------
# MODEL 1: BLIND BASELINE
# ------------------------------------------

# Predict every shipment as on-time (0)
baseline_predictions = [0] * len(y_val)

baseline_metrics = evaluate_model(
    y_val,
    baseline_predictions
)

results.append({
    "model": "blind_all_on_time",
    **baseline_metrics
})


# ------------------------------------------
# MODEL 2: LOGISTIC REGRESSION
# ------------------------------------------

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Train using training data only
logistic_model.fit(X_train, y_train)

# Predict on validation data
logistic_predictions = logistic_model.predict(X_val)

logistic_metrics = evaluate_model(
    y_val,
    logistic_predictions
)

results.append({
    "model": "logistic_regression",
    **logistic_metrics
})


# ------------------------------------------
# MODEL 3: DECISION TREE DEPTH SELECTION
# ------------------------------------------

tree_depths = [2, 3, 4, 5, 6, 8, 10]

tree_results = []
tree_models = {}

for depth in tree_depths:

    tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    # Train on training data
    tree_model.fit(X_train, y_train)

    # Predict on validation data
    tree_predictions = tree_model.predict(X_val)

    tree_metrics = evaluate_model(
        y_val,
        tree_predictions
    )

    tree_results.append({
        "depth": depth,
        "val_f1": tree_metrics["f1"]
    })

    tree_models[depth] = tree_model


# Select highest F1; ties go to smaller depth
best_tree = min(
    tree_results,
    key=lambda item: (-item["val_f1"], item["depth"])
)

best_depth = best_tree["depth"]

print("\nDecision Tree depth comparison:")

for item in tree_results:
    print(
        f"Depth: {item['depth']}, "
        f"Validation F1: {item['val_f1']:.4f}"
    )

print("\nSelected Decision Tree depth:", best_depth)

# Evaluate the selected tree
best_tree_model = tree_models[best_depth]

tree_predictions = best_tree_model.predict(X_val)

tree_metrics = evaluate_model(
    y_val,
    tree_predictions
)

results.append({
    "model": f"decision_tree_depth_{best_depth}",
    **tree_metrics
})


# ------------------------------------------
# MODEL 4: RANDOM FOREST
# ------------------------------------------

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train on training data
random_forest_model.fit(X_train, y_train)

# Predict on validation data
rf_predictions = random_forest_model.predict(X_val)

rf_metrics = evaluate_model(
    y_val,
    rf_predictions
)

results.append({
    "model": "random_forest_default",
    **rf_metrics
})


# ------------------------------------------
# SAVE CLASSIFICATION RESULTS
# ------------------------------------------

results_df = pd.DataFrame(results)

# Round decimal metrics to four places
metric_columns = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]

results_df[metric_columns] = (
    results_df[metric_columns].round(4)
)

# Ensure output directory exists
Path("outputs").mkdir(exist_ok=True)

# Save results without an index column
results_df.to_csv(
    "outputs/part_c_models.csv",
    index=False
)


# ------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------

print("\n========== PART C: TASK 2 ==========")

print("\nClassification model comparison:")

print(results_df.to_string(index=False))

print("\nSaved: outputs/part_c_models.csv")
print("Number of model rows:", len(results_df))



# ==========================================
# TASK 3: FEATURE IMPORTANCE AND SELECTION
# ==========================================

# ------------------------------------------
# EXPERIMENT A: FEATURE IMPORTANCE
# ------------------------------------------

# Train a default Random Forest using all 17 features
feature_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

feature_rf.fit(X_train, y_train)

# Extract Random Forest feature importance
rf_importance = feature_rf.feature_importances_

# Calculate F-scores using training data only
f_scores, p_values = f_classif(X_train, y_train)

# Create a feature score DataFrame
feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importance
})

# Rank features by descending F-score
feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

# Identify the five bookkeeping/noise columns
feature_scores["is_noise"] = (
    feature_scores["feature"].isin(NOISE_COLUMNS)
)

# Arrange columns in the required order
feature_scores = feature_scores[
    [
        "feature",
        "f_score",
        "f_rank",
        "rf_importance",
        "is_noise"
    ]
]

# Round decimal values to four places
feature_scores["f_score"] = feature_scores["f_score"].round(4)
feature_scores["rf_importance"] = (
    feature_scores["rf_importance"].round(4)
)

# Save feature scores
feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False
)


# ------------------------------------------
# EXPERIMENT B: FEATURE COUNT COMPARISON
# ------------------------------------------

k_values = [5, 7, 10, 12, 17]

selectk_results = []

for k in k_values:

    # Create a pipeline with feature selection
    # followed by Random Forest classification
    selectk_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ])

    # Train using training data only
    selectk_model.fit(X_train, y_train)

    # Predict using validation data
    selectk_predictions = selectk_model.predict(X_val)

    # Calculate validation F1
    val_f1 = f1_score(
        y_val,
        selectk_predictions,
        zero_division=0
    )

    selectk_results.append({
        "k": k,
        "val_f1": val_f1
    })


# Create results DataFrame
selectk_df = pd.DataFrame(selectk_results)

# Round F1 values
selectk_df["val_f1"] = selectk_df["val_f1"].round(4)

# Save feature-count comparison
selectk_df.to_csv(
    "outputs/part_c_selectk.csv",
    index=False
)


# ------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------

print("\n========== PART C: TASK 3 ==========")

print("\nFeature importance and F-score ranking:")

print(
    feature_scores.sort_values(
        "f_rank"
    ).to_string(index=False)
)

# Calculate combined importance of noise columns
noise_importance = feature_scores.loc[
    feature_scores["is_noise"],
    "rf_importance"
].sum()

print(
    "\nCombined noise-column importance:",
    round(noise_importance, 4)
)

print("\nFeature count comparison:")

print(selectk_df.to_string(index=False))

print("\nSaved: outputs/part_c_feature_scores.csv")
print("Saved: outputs/part_c_selectk.csv")

print("\nFeature score rows:", len(feature_scores))
print("SelectKBest comparison rows:", len(selectk_df))



# ==========================================
# TASK 4: RANDOM FOREST HYPERPARAMETER TUNING
# ==========================================

# ------------------------------------------
# STEP 1: SELECT THE TOP 7 FEATURES
# ------------------------------------------

# Fit selector using training data only
tuning_selector = SelectKBest(f_classif, k=7)

X_train_selected = tuning_selector.fit_transform(
    X_train,
    y_train
)

# Transform validation data using the same selector
X_val_selected = tuning_selector.transform(X_val)

print("\n========== PART C: TASK 4 ==========")

print("\nSelected feature count:", X_train_selected.shape[1])

print(
    "Selected features:",
    [
        feature
        for feature, selected
        in zip(FEATURES, tuning_selector.get_support())
        if selected
    ]
)


# ------------------------------------------
# STEP 2: DEFINE HYPERPARAMETER GRID
# ------------------------------------------

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}


# ------------------------------------------
# STEP 3: GRID SEARCH
# ------------------------------------------

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),
    param_grid=param_grid,
    cv=5
)

# Fit all 27 combinations using five-fold CV
grid_search.fit(
    X_train_selected,
    y_train
)

# Get the best grid-search model
grid_best_model = grid_search.best_estimator_

# Predict on validation data
grid_predictions = grid_best_model.predict(
    X_val_selected
)

# Calculate validation F1
grid_val_f1 = f1_score(
    y_val,
    grid_predictions,
    zero_division=0
)


# ------------------------------------------
# STEP 4: RANDOMIZED SEARCH
# ------------------------------------------

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

# Fit nine sampled combinations using five-fold CV
random_search.fit(
    X_train_selected,
    y_train
)

# Get the best randomized-search model
random_best_model = random_search.best_estimator_

# Predict on validation data
random_predictions = random_best_model.predict(
    X_val_selected
)

# Calculate validation F1
random_val_f1 = f1_score(
    y_val,
    random_predictions,
    zero_division=0
)


# ------------------------------------------
# STEP 5: DEFAULT RANDOM FOREST
# ------------------------------------------

default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train using the same seven selected features
default_forest.fit(
    X_train_selected,
    y_train
)

# Predict on validation data
default_predictions = default_forest.predict(
    X_val_selected
)

# Calculate validation F1
default_val_f1 = f1_score(
    y_val,
    default_predictions,
    zero_division=0
)


# ------------------------------------------
# STEP 6: COLLECT TUNING RESULTS
# ------------------------------------------

tuning_results = [

    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1
    },

    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_val_f1
    },

    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": str({
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1
        }),
        "best_cv_accuracy": None,
        "val_f1": default_val_f1
    }

]


# ------------------------------------------
# STEP 7: SAVE TUNING RESULTS
# ------------------------------------------

tuning_df = pd.DataFrame(tuning_results)

# Round numerical results
tuning_df["best_cv_accuracy"] = (
    tuning_df["best_cv_accuracy"].round(4)
)

tuning_df["val_f1"] = tuning_df["val_f1"].round(4)

# Save CSV
tuning_df.to_csv(
    "outputs/part_c_tuning.csv",
    index=False
)


# ------------------------------------------
# STEP 8: DISPLAY RESULTS
# ------------------------------------------

print("\nGrid Search best parameters:")
print(grid_search.best_params_)

print(
    "Grid Search best CV accuracy:",
    round(grid_search.best_score_, 4)
)

print("\nRandomized Search best parameters:")
print(random_search.best_params_)

print(
    "Randomized Search best CV accuracy:",
    round(random_search.best_score_, 4)
)

print("\nDefault Forest validation F1:")
print(round(default_val_f1, 4))

print("\nComplete tuning comparison:")
print(tuning_df.to_string(index=False))

print("\nSaved: outputs/part_c_tuning.csv")
print("Number of tuning rows:", len(tuning_df))


# ==========================================
# TASK 5: SELECT AND FIT FINAL MODEL
# ==========================================

# STEP 1: Select the best tuning result using validation F1.
# Tie-breaking order:
# default forest, grid search, randomized search.

tuning_df = pd.read_csv("outputs/part_c_tuning.csv")

priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2
}

tuning_df["priority"] = tuning_df["search"].map(priority)

best_result = tuning_df.sort_values(
    by=["val_f1", "priority"],
    ascending=[False, True]
).iloc[0]

chosen_from = best_result["search"]

print("\n========== PART C: TASK 5 ==========")
print("Chosen model:", chosen_from)
print("Validation F1:", best_result["val_f1"])

# STEP 2: Set the selected model parameters.
if chosen_from == "default_forest_same_7_features":
    chosen_n_estimators = 100
    chosen_max_depth = None
    chosen_min_samples_leaf = 1

elif chosen_from == "grid_search":
    chosen_n_estimators = grid_search.best_params_["n_estimators"]
    chosen_max_depth = grid_search.best_params_["max_depth"]
    chosen_min_samples_leaf = grid_search.best_params_["min_samples_leaf"]

else:
    chosen_n_estimators = random_search.best_params_["n_estimators"]
    chosen_max_depth = random_search.best_params_["max_depth"]
    chosen_min_samples_leaf = random_search.best_params_["min_samples_leaf"]

# STEP 3: Build the final pipeline.
final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("classifier", RandomForestClassifier(
        n_estimators=chosen_n_estimators,
        max_depth=chosen_max_depth,
        min_samples_leaf=chosen_min_samples_leaf,
        random_state=42
    ))
])

# STEP 4: Fit using training data only.
final_model.fit(X_train, y_train)

print("Selected features:",
      final_model.named_steps["selector"].get_feature_names_out(FEATURES).tolist())

print("Final model parameters:")
print({
    "n_estimators": chosen_n_estimators,
    "max_depth": chosen_max_depth,
    "min_samples_leaf": chosen_min_samples_leaf
})


# ==========================================
# TASK 6: ALERT THRESHOLD SELECTION
# ==========================================

from sklearn.metrics import confusion_matrix

# Get delayed-class probabilities on validation data.
val_probabilities = final_model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

    val_predictions = (val_probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        val_predictions,
        labels=[0, 1]
    ).ravel()

    threshold_results.append({
        "threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_val, val_predictions),
        "precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, val_predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    "outputs/part_c_thresholds.csv",
    index=False
)

# Select the highest threshold whose recall is at least 0.80.
eligible_thresholds = threshold_df[
    threshold_df["recall"] >= 0.80
]

if len(eligible_thresholds) > 0:
    chosen_threshold = eligible_thresholds["threshold"].max()
else:
    chosen_threshold = 0.1

print("\n========== PART C: TASK 6 ==========")
print(threshold_df.round(4).to_string(index=False))
print("\nChosen threshold:", chosen_threshold)
print(
    "Recall at chosen threshold:",
    threshold_df.loc[
        threshold_df["threshold"] == chosen_threshold,
        "recall"
    ].iloc[0]
)
print("Saved: outputs/part_c_thresholds.csv")


# ==========================================
# TASK 7: FINAL TEST EVALUATION
# ==========================================

# STEP 1: Predict test probabilities once.
test_probabilities = final_model.predict_proba(X_test)[:, 1]

# STEP 2: Apply the validation-selected threshold.
test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

# STEP 3: Calculate test metrics.
tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

final_test_results = pd.DataFrame([{
    "threshold": chosen_threshold,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "accuracy": accuracy_score(y_test, test_predictions),
    "precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_test, test_predictions, zero_division=0
    )
}])

final_test_results.to_csv(
    "outputs/part_c_final_test.csv",
    index=False
)

# STEP 4: Save final model configuration.
final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_n_estimators,
    "max_depth": (
        "unlimited"
        if chosen_max_depth is None
        else chosen_max_depth
    ),
    "min_samples_leaf": chosen_min_samples_leaf,
    "threshold": chosen_threshold,
    "chosen_from": chosen_from
}])

final_config.to_csv(
    "outputs/part_c_final_config.csv",
    index=False
)

print("\n========== PART C: TASK 7 ==========")
print("\nFinal test results:")
print(final_test_results.round(4).to_string(index=False))

print("\nFinal model configuration:")
print(final_config.to_string(index=False))

print("\nSaved: outputs/part_c_final_test.csv")
print("Saved: outputs/part_c_final_config.csv")

