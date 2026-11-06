
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import os

from src.shipment_prep import FEATURES

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold, cross_val_score


# --------------------------------------------------
# 1. Load the modelling dataset
# --------------------------------------------------

df = pd.read_csv("data/modelling_table.csv")

# Independent variables: 17 pinned features
X = df[FEATURES]

# Dependent variable: delivery hours
y = df["delivery_hours"]


# --------------------------------------------------
# 2. First split: 70% training, 30% temporary
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# --------------------------------------------------
# 3. Second split: 50% validation, 50% test
# --------------------------------------------------

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)


# --------------------------------------------------
# 4. Verify split sizes
# --------------------------------------------------

print("===== TRAINING / VALIDATION / TEST SPLIT =====")

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

print("\n===== FEATURE DIMENSIONS =====")

print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Test features:", X_test.shape)

print("\nTotal rows:", len(X_train) + len(X_val) + len(X_test))



# --------------------------------------------------
# 5. Scaling without data leakage
# --------------------------------------------------

# Create the output directory
os.makedirs("outputs", exist_ok=True)

# Initialize the scaler
scaler = StandardScaler()

# Fit ONLY on training data and transform training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the already fitted scaler
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 6. Verify scaling
# --------------------------------------------------

# Calculate the largest absolute feature mean
train_scaled_max_abs_mean = abs(
    X_train_scaled.mean(axis=0)
).max()

test_scaled_max_abs_mean = abs(
    X_test_scaled.mean(axis=0)
).max()


# --------------------------------------------------
# 7. Write Part B checks file
# --------------------------------------------------

checks = pd.DataFrame({
    "metric": [
        "train_rows",
        "val_rows",
        "test_rows",
        "train_scaled_max_abs_mean",
        "test_scaled_max_abs_mean"
    ],
    "value": [
        len(X_train),
        len(X_val),
        len(X_test),
        round(float(train_scaled_max_abs_mean), 4),
        round(float(test_scaled_max_abs_mean), 4)
    ]
})

checks.to_csv(
    "outputs/part_b_checks.csv",
    index=False
)


# --------------------------------------------------
# 8. Print scaling results
# --------------------------------------------------

print("\n===== SCALING VERIFICATION =====")

print(
    "Training scaled max absolute mean:",
    train_scaled_max_abs_mean
)

print(
    "Test scaled max absolute mean:",
    test_scaled_max_abs_mean
)

print("\nChecks saved to outputs/part_b_checks.csv")



# --------------------------------------------------
# 9. Mean baseline regression
# --------------------------------------------------

# Predict the training target's mean for every validation row
baseline_prediction = [y_train.mean()] * len(y_val)

# Calculate evaluation metrics
baseline_mae = mean_absolute_error(
    y_val,
    baseline_prediction
)

baseline_mse = mean_squared_error(
    y_val,
    baseline_prediction
)

baseline_rmse = baseline_mse ** 0.5

baseline_r2 = r2_score(
    y_val,
    baseline_prediction
)


# Store results for the Part B metrics file
baseline_metrics = {
    "model": "mean_baseline",
    "split": "val",
    "MAE": round(float(baseline_mae), 4),
    "MSE": round(float(baseline_mse), 4),
    "RMSE": round(float(baseline_rmse), 4),
    "R2": round(float(baseline_r2), 4)
}


# --------------------------------------------------
# 10. Print baseline results
# --------------------------------------------------

print("\n===== MEAN BASELINE REGRESSION =====")

print("Training target mean:", round(y_train.mean(), 4))

print("MAE:", round(baseline_mae, 4))
print("MSE:", round(baseline_mse, 4))
print("RMSE:", round(baseline_rmse, 4))
print("R2:", round(baseline_r2, 4))


# 10. Decision tree depth diagnostic

depth_values = [1, 2, 3, 4, 6, 8, 12, None]

depth_results = []

for depth in depth_values:
    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    val_prediction = tree.predict(X_val)

    train_rmse = mean_squared_error(
        y_train, train_prediction
    ) ** 0.5

    val_rmse = mean_squared_error(
        y_val, val_prediction
    ) ** 0.5

    depth_results.append({
        "max_depth": "unlimited" if depth is None else depth,
        "train_rmse": round(float(train_rmse), 4),
        "val_rmse": round(float(val_rmse), 4)
    })

depth_df = pd.DataFrame(depth_results)

depth_df.to_csv(
    "outputs/part_b_tree_depth.csv",
    index=False
)

print("\n===== DECISION TREE DEPTH DIAGNOSTIC =====")
print(depth_df.to_string(index=False))

# Select the depth with the lowest validation RMSE.
# In a tie, the smaller depth is preferred.
best_tree_row = min(
    depth_results,
    key=lambda row: (
        row["val_rmse"],
        float("inf") if row["max_depth"] == "unlimited"
        else row["max_depth"]
    )
)

best_tree_depth = (
    None if best_tree_row["max_depth"] == "unlimited"
    else best_tree_row["max_depth"]
)

print("\nBest tree depth:", best_tree_row["max_depth"])
print("Best validation RMSE:", best_tree_row["val_rmse"])


# 11. Train and compare regression models

models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "decision_tree": DecisionTreeRegressor(
        max_depth=best_tree_depth,
        random_state=42
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

model_results = [baseline_metrics.copy()]

for model_name, model in models.items():

    model.fit(X_train, y_train)

    val_prediction = model.predict(X_val)

    mae = mean_absolute_error(y_val, val_prediction)
    mse = mean_squared_error(y_val, val_prediction)
    rmse = mse ** 0.5
    r2 = r2_score(y_val, val_prediction)

    model_results.append({
        "model": model_name,
        "split": "val",
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4)
    })

metrics_df = pd.DataFrame(model_results)

metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False
)

print("\n===== REGRESSION MODEL COMPARISON =====")
print(metrics_df.to_string(index=False))


# 12. Gradient boosting learning-rate experiment

learning_rates = [0.01, 0.1, 1.0]

gb_results = []

for lr in learning_rates:

    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    val_prediction = gb_model.predict(X_val)

    val_mse = mean_squared_error(y_val, val_prediction)
    val_rmse = val_mse ** 0.5
    val_r2 = r2_score(y_val, val_prediction)

    gb_results.append({
        "learning_rate": lr,
        "val_rmse": round(float(val_rmse), 4),
        "val_r2": round(float(val_r2), 4)
    })

gb_df = pd.DataFrame(gb_results)

gb_df.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False
)

print("\n===== GRADIENT BOOSTING LEARNING-RATE EXPERIMENT =====")
print(gb_df.to_string(index=False))

# 13. Cross-validation for the two best validation models

# Combine training and validation data.
# The test set remains completely separate.
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Rank the four candidate models by validation RMSE.
ranked_models = (
    metrics_df[metrics_df["model"] != "mean_baseline"]
    .sort_values("RMSE", ascending=True)
)

top_two_names = ranked_models["model"].head(2).tolist()

print("\n===== TOP TWO MODELS FOR CROSS-VALIDATION =====")
print(top_two_names)

# Create one KFold object and reuse it for both models.
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for model_name in top_two_names:

    model = models[model_name]

    cv_scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=cv
    )

    cv_results.append({
        "model": model_name,
        "mean_r2": round(float(cv_scores.mean()), 4),
        "std_r2": round(float(cv_scores.std()), 4)
    })

cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    "outputs/part_b_cv.csv",
    index=False
)

print("\n===== FIVE-FOLD CROSS-VALIDATION =====")
print(cv_df.to_string(index=False))


# 14. Final model selection and one-time test evaluation

best_model_name = ranked_models.iloc[0]["model"]
final_model = models[best_model_name]

# The selected model was already fitted on X_train in Task 5.
# Evaluate it on the test set exactly once.
test_prediction = final_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_prediction)
test_mse = mean_squared_error(y_test, test_prediction)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(y_test, test_prediction)

test_metrics = {
    "model": best_model_name,
    "split": "test",
    "MAE": round(float(test_mae), 4),
    "MSE": round(float(test_mse), 4),
    "RMSE": round(float(test_rmse), 4),
    "R2": round(float(test_r2), 4)
}

# Add the single final-test row to the existing validation metrics.
final_metrics_df = pd.concat(
    [metrics_df, pd.DataFrame([test_metrics])],
    ignore_index=True
)

final_metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False
)

print("\n===== FINAL MODEL =====")
print("Selected model:", best_model_name)
print("Test MAE:", test_metrics["MAE"])
print("Test MSE:", test_metrics["MSE"])
print("Test RMSE:", test_metrics["RMSE"])
print("Test R2:", test_metrics["R2"])
