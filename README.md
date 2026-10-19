# Minor Project: Heart Failure Survival Prediction using Boosting

**Name:** Rohan Jha

## Dataset
Heart Failure Clinical Records (UCI): https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
299 heart failure patients with 12 clinical features (age, anaemia, diabetes, blood pressure, ejection fraction, serum creatinine, serum sodium, etc.). The target `DEATH_EVENT` shows whether the patient died during follow-up (1) or survived (0).

## Problem Statement and Objective
Predict whether a heart failure patient will die during the follow-up period from their clinical records, so high-risk patients can be monitored closely. The objective is to build a Gradient Boosting classifier and find the most important medical factors.

## Data Cleaning Summary
- No missing values and no duplicate rows. All data types were already numeric.
- Capped outliers (IQR method) in `creatinine_phosphokinase`, `platelets` and `serum_creatinine`, keeping all 299 rows because the dataset is small.
- No scaling or encoding needed, since boosting uses decision trees and the 0/1 columns are already encoded.

## Visualizations
1. Survived vs died count (target distribution)
2. Age vs death event
3. Ejection fraction vs death event
4. Serum creatinine vs death event

## Method
Binary classification with `GradientBoostingClassifier` (80/20 stratified train-test split, `random_state = 42`). SMOTE was not used because the classes are about 68% / 32%, which is not severely imbalanced.

## Evaluation
| Measure | Result |
|---|---|
| Accuracy | about 0.82 |
| Recall (died) | about 0.63 |
| Precision (died) | about 0.75 |
| ROC AUC | about 0.84 |

Top features: `time`, `serum_creatinine`, `ejection_fraction`.

## Conclusion
Older age, lower ejection fraction and higher serum creatinine are linked with a higher chance of death. The model is correct for about 8 in 10 test patients but misses about 4 in 10 patients who died. Limitations: very small dataset (60 test patients), `time` is only known after follow-up, and the data is from one hospital. Future work: remove `time`, tune the model, use cross-validation and compare with XGBoost or CatBoost.

## How to Run
Open `Heart_Failure_Boosting_Project.ipynb` in Google Colab or Jupyter and run all cells (the first cell installs `ucimlrepo`; it needs internet to fetch the data). Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn.
