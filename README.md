# delhivery-delay-intelligence
Last-Mile Delivery Delay Intelligence Pipeline

## Part A — Data Foundation and Delay Evidence

### Data preparation

- Generated 1,224 historical shipment records across 24 hubs.
- Removed 24 duplicate records, leaving 1,200 unique shipments.
- Standardized service types and handled missing payment modes.
- Converted declared values from formatted currency strings into numeric INR values.
- Treated parcel weights above 100 kg as invalid and replaced missing numeric values using historical medians.
- Merged hub information into the shipment dataset.
- Created the promised-hours, COD, weekend, dispatch-hour and log-declared-value features.
- Created the delay label using actual delivery hours compared with promised hours.

### Data quality results

- Cleaned shipment records: 1,200
- Duplicate records remaining: 0
- Missing modelling feature values: 0
- Delayed shipments: 278
- Overall delay rate: 23.17%

### Key findings

1. **Service type:** Express shipments had the highest delay rate at 48.97%, followed by Standard at 20.56% and Economy at 8.57%.

2. **Hub tier:** Delay rates increased across hub tiers: Tier 1 at 4.00%, Tier 2 at 21.50% and Tier 3 at 44.00%.

3. **Distance:** Delay rates increased with distance. The 601+ km group had an 85.23% delay rate, compared with 2.19% for shipments in the 0–200 km group.

4. **Correlation:** The strongest correlations with delivery hours were promised hours (+0.8372), distance (+0.4013), number of stops (+0.3247), hub daily capacity (−0.3237) and hub tier (+0.3175).

5. **Noise features:** The five bookkeeping/noise features had correlations close to zero with delivery hours.

6. **Outliers:** The IQR method identified one potential delivery-hours outlier. It was retained rather than automatically removed.

7. **Declared value:** The mean declared value was ₹1,533.05, while the median was ₹1,121.50. The log transformation produced a mean of 7.028 and median of 7.022.

### Conclusion

The exploratory analysis indicates that distance, service type and hub tier are important variables associated with delivery delays. These findings provide a foundation for the regression and classification models developed in the next parts of the capstone.



## Model results (regression)

### Baseline performance

The mean baseline achieved a validation R² of -0.0212 and an RMSE of 18.8552 hours. This indicates that predicting the training target mean for every shipment provides almost no useful predictive performance.

### Model comparison

Linear Regression achieved the lowest validation RMSE of 4.8912 hours, with an R² of 0.9313. Gradient Boosting was a close second, with an RMSE of 4.9165 hours and an R² of 0.9306.

Random Forest achieved an R² of 0.9138, while the Decision Tree achieved an R² of 0.8867.

Linear Regression was selected as the final model because it had the lowest validation RMSE.

### Decision tree depth analysis

A tree with depth 1 showed high bias: its training RMSE was 11.6334 and validation RMSE was 12.1324.

The best depth was 6, with a validation RMSE of 6.2796 hours.

The unlimited-depth tree showed high variance: its training RMSE was 0.0000, but its validation RMSE increased to 7.7890. This indicates overfitting.

### Gradient boosting learning-rate experiment

A learning rate of 0.1 performed best, achieving a validation R² of 0.9306.

A learning rate of 0.01 learned too slowly with 100 estimators, achieving an R² of 0.7127. A learning rate of 1.0 performed worse, with an R² of 0.8440, because the larger learning steps reduced generalization performance.

### Cross-validation and final model

Five-fold cross-validation produced a mean R² of 0.9261 for Linear Regression and 0.9352 for Gradient Boosting.

Although Gradient Boosting achieved a slightly higher cross-validation mean R², Linear Regression was retained because it had the lowest validation RMSE. This follows the validation-based model-selection rule.

The final Linear Regression model achieved the following test results:

- MAE: 3.5671 hours
- MSE: 20.4624
- RMSE: 4.5235 hours
- R²: 0.9493



## Model results (classification)

### Baseline and classification models

The blind baseline predicts every shipment as not delayed. It achieves
76.67% accuracy because most shipments in the dataset are not delayed.
However, its precision, recall, and F1-score are all zero. It fails to
identify any delayed shipments, so accuracy alone is misleading for
this classification problem.

The Logistic Regression model achieved a validation F1-score of 0.8046.
The best Decision Tree used a maximum depth of 6 and achieved an F1-score
of 0.7470. The default Random Forest achieved an F1-score of 0.8000.

### Feature importance and noise

The seven highest-ranked features by ANOVA F-score were:

1. distance_km
2. hub_tier
3. hub_daily_capacity
4. n_stops
5. promised_hours
6. rain_flag
7. sale_event_flag

These features had F-scores above 30. The five known noise features
ranked eighth or lower and had F-scores below 10.

The five noise features together accounted for 22.53% of the Random
Forest's feature importance. This shows that the model still assigns
some importance to irrelevant features, which supports using feature
selection to reduce noise.

### Feature selection

Validation F1-scores for the tested feature counts were:

| Number of features (k) | Validation F1 |
|---:|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.8140 |
| 12 | 0.8049 |
| 17 | 0.8000 |

Using seven features produced the highest validation F1-score, so
k=7 was used for the final model.

### Hyperparameter tuning

Grid Search evaluated 27 candidates using 5-fold cross-validation,
for 135 total fits. Its best cross-validation accuracy was 0.8952
and its validation F1-score was 0.7750.

Randomized Search evaluated 9 candidates using 5-fold cross-validation,
for 45 total fits. Its best cross-validation accuracy was 0.8905
and its validation F1-score was 0.7901.

The untuned Random Forest using the same seven selected features
achieved a validation F1-score of 0.8333. This was higher than both
search results, so the untuned forest was selected.

The validation set contains 180 shipments, including 42 delayed
shipments. Its limited size means that validation scores can vary
with the particular split. Cross-validation provides additional
training-data evaluation, but the final test set remains the
independent evaluation.

### Alert threshold selection

The threshold was selected using validation recall, with a required
minimum recall of 0.80. The highest threshold meeting this requirement
was 0.5.

At this threshold, validation precision was 0.7955 and recall was
0.8333. This means the model identified 35 of the 42 delayed validation
shipments, with 9 false positive alerts.

Increasing the threshold can improve precision but may reduce recall.
The threshold of 0.5 was retained because it was the highest tested
threshold that met the required recall.

### Final test results

The final model used SelectKBest with k=7 followed by a Random Forest
with 100 trees, unlimited depth, and min_samples_leaf=1. The alert
threshold was 0.5.

On the untouched test set, the model achieved:

| Metric | Test result |
|---|---:|
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1-score | 0.7209 |

The test set contained 41 delayed shipments. The model identified
31 of them, with 14 false positive alerts and 10 missed delayed
shipments.

Test recall (0.7561) was lower than validation recall (0.8333).
This difference highlights that performance on a held-out dataset
may differ from validation performance. The test results were
reported without further model or threshold tuning.





## Part D — Hub Segmentation and Operational Delay Actions

### How to run

Run the following commands from the project root in a fresh clone.

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

The scripts run in sequence: data generation, Parts A–C, hub segmentation, model packaging, and new-shipment prediction.

`package_model.py` rebuilds the final classification pipeline using the configuration saved by Part C. It saves the trained model and supporting files in `models/`. The model is serialized with joblib; use the pinned scikit-learn version in `requirements.txt` when loading it to avoid compatibility issues.

Part D outputs are saved in `outputs/`, including hub profiles, separation evidence, hub segments, segment profiles, tier mismatches, model reload checks, and new-shipment actions.

### Operations recommendation

The three hub segments show clear differences in delivery performance:

| Segment | Hubs | Mean delivery time (hours) | Delay rate | Mean distance (km) | Mean stops |
|---|---:|---:|---:|---:|---:|
| Standard | 7 | 47.2396 | 2.57% | 188.9296 | 55.3686 |
| Watch | 8 | 56.8137 | 15.50% | 314.1762 | 70.7175 |
| Escalate | 9 | 64.6504 | 46.00% | 470.4757 | 84.3444 |

Escalate hubs should receive the highest operational attention because their average delay rate is 46.00%. Operations teams should review their longer routes, higher stop counts, and local capacity constraints. Watch hubs should receive closer monitoring, while Standard hubs can continue under routine monitoring.

Four hubs differ from the segment expected from their assigned tier:

- HUB07, Kolkata: Tier 1 but Watch.
- HUB13, Nagpur: Tier 2 but Escalate.
- HUB15, Kochi: Tier 2 but Escalate.
- HUB22, Dehradun: Tier 3 but Watch.

These mismatches show that hub tier alone does not fully describe operational delay risk. Review the actual performance of these hubs rather than relying only on their tier labels.

The selected K-Means solution uses three clusters. Its across-cluster to within-cluster distance ratio is 5.3928. Although K=4 and K=5 produce higher ratios, they create smaller groups, including clusters of four, three, and two hubs. Three segments provide a more practical basis for operational playbooks.

The shipment classifier uses a probability threshold of 0.50. On the Part C test set, precision was 68.89%, meaning roughly 7 of every 10 flagged shipments were delayed. Recall was 75.61%, meaning roughly 8 of every 10 actually delayed shipments were identified.

For the 48 new shipments scored in Part D:

- 8 were flagged.
- 6 were assigned to hub-manager escalation.
- 2 were assigned to proactive customer calls.
- 40 required no action.

Flagged shipments at Escalate hubs should be routed to hub managers. Other flagged shipments should receive proactive customer calls. These actions are decision-support recommendations and should be reviewed by operations teams.

### Production plan

The project includes several elements that support a reliable production workflow:

- **Reproducibility:** Fixed random seeds are used in model training and hub clustering.
- **Consistency:** `shipment_prep.py` provides shared cleaning and feature-building logic.
- **Model packaging:** The trained pipeline, threshold, feature order, and imputation medians are saved separately in `models/`.
- **Reload verification:** `part_d_reload_check.csv` records the test-row count and probability difference after reloading the saved model.
- **Maintainability:** Segmentation, model packaging, and prediction are separated into dedicated scripts.
- **Operational traceability:** Hub segments and shipment actions are saved as CSV outputs for review.

Suggested monitoring metrics and initial investigation triggers:

1. **Flag rate:** Investigate if the share of shipments flagged exceeds 25% over a daily operating period.
2. **Missing input rate:** Investigate if more than 5% of incoming shipments have missing values in important model inputs before imputation.
3. **Delay rate by segment:** Review an Escalate hub if its observed delay rate remains below or above its established profile by more than 10 percentage points over a rolling month.
4. **Prediction distribution:** Review the model if the average predicted delay probability changes by more than 0.10 compared with the previous month's average.

These are proposed starting thresholds and should be calibrated using real operating data.

Retrain the model quarterly as an initial schedule, and also consider retraining after major changes in routes, hub capacity, service types, or shipment patterns. Evaluate the new model against a time-based holdout before deployment.

During the 15–28 September sale window, compare shipment volumes, delay rates, prediction rates, and operational outcomes with the same type of sale event from a previous period. Avoid comparing the sale window only with ordinary weeks. Monitor changes in input distributions and prediction distributions for signs of data drift or changing operational conditions.

### Limitations

- The project uses synthetic shipment data, so the observed patterns may not represent actual delivery operations.
- The final classification test set contains 180 shipments, including only 41 delayed shipments. Performance estimates may therefore vary on a larger or different dataset.
- The alert threshold was selected using one validation split. A different validation period may produce a different threshold.
- Hub segments are based on historical hub-level averages. Conditions can change over time, so segments should be recalculated periodically.
- The monitoring triggers and retraining schedule are proposed starting points, not thresholds validated through live production operations.
- Predictions depend on the availability and consistency of shipment and hub input fields. Changes in data collection or feature definitions may affect model performance.