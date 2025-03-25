import os
import random
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error, make_scorer

# -------------------------------------------------------------------
# 1. Define Paths
# -------------------------------------------------------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

train_file = os.path.join(DATA_DIR, "train.csv")             # Main 42-labeled
oligomer_file = os.path.join(DATA_DIR, "OligomerFeatures.csv")# Additional 2200
test_file = os.path.join(DATA_DIR, "test.csv")               # Competition test

# -------------------------------------------------------------------
# 2. Load Data
# -------------------------------------------------------------------
train_main = pd.read_csv(train_file)
oligomer_df = pd.read_csv(oligomer_file)
test_df = pd.read_csv(test_file)

print("Shapes:")
print("  train_main:", train_main.shape)
print("  oligomer_df:", oligomer_df.shape)
print("  test_df:", test_df.shape)

# Identify labeled/unlabeled in oligomer
oligomer_labeled = oligomer_df.dropna(subset=["T80"]).copy()
oligomer_unlabeled = oligomer_df[oligomer_df["T80"].isna()].copy()
print("  oligomer_labeled:", oligomer_labeled.shape)
print("  oligomer_unlabeled:", oligomer_unlabeled.shape)

# Combine labeled sets
combined_labeled = pd.concat([train_main, oligomer_labeled], ignore_index=True)
combined_labeled.drop_duplicates(subset=["Batch_ID"], inplace=True)
print("Combined labeled set size:", combined_labeled.shape)

# -------------------------------------------------------------------
# 3. Pipeline Placeholder
#    Replace with your advanced pipeline (DOS PCA, 3D RDKit descriptors, etc.)
# -------------------------------------------------------------------
def pipeline_process(df, is_train=True):
    """
    Placeholder for your actual pipeline.
    1) Possibly generate advanced descriptors
    2) PCA on DOS
    3) Filter invalid/constant
    4) Return processed df
    We'll just do a log-transform if T80 is present.
    """
    df = df.copy()
    if "T80" in df.columns and not df["T80"].isna().all():
        df["T80_log"] = np.log1p(df["T80"])
    return df

def get_feature_cols(df):
    """
    Only numeric columns. Exclude known non-features (Batch_ID, Smiles, T80, T80_log, etc.)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {"Batch_ID", "Smiles", "T80", "T80_log"}
    final_cols = [c for c in numeric_cols if c not in exclude_cols]
    return final_cols

# -------------------------------------------------------------------
# 4. Process Combined Labeled
# -------------------------------------------------------------------
labeled_processed = pipeline_process(combined_labeled, is_train=True)
feature_cols = get_feature_cols(labeled_processed)

X_labeled = labeled_processed[feature_cols].fillna(0.0)
y_labeled = labeled_processed["T80_log"] if "T80_log" in labeled_processed else np.log1p(labeled_processed["T80"])

print("Feature columns:", feature_cols)

# -------------------------------------------------------------------
# 5. Train Two Models (Ensemble) for Confidence Estimation
# -------------------------------------------------------------------
# We'll train 2 distinct models (RF & GBM) to estimate confidence via std of predictions

# a) Simple hyperparameter search for RandomForest
param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
}
rf_model = RandomForestRegressor(random_state=42)
msle_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)

search_rf = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=5,  # small search for demonstration
    scoring=msle_scorer,
    cv=3,
    random_state=42,
    n_jobs=-1
)
search_rf.fit(X_labeled, y_labeled)
best_rf = search_rf.best_estimator_
print("Best RF params:", search_rf.best_params_)

# b) GradientBoostingRegressor (fixed params for demonstration)
gbm_model = GradientBoostingRegressor(
    random_state=42,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5
)
gbm_model.fit(X_labeled, y_labeled)

# We'll combine them for pseudo-labeling predictions
def ensemble_predict(X):
    p1 = best_rf.predict(X)
    p2 = gbm_model.predict(X)
    # average predictions
    avg_pred = (p1 + p2) / 2.0
    # measure std for confidence
    std_pred = np.std(np.vstack([p1, p2]), axis=0)
    return avg_pred, std_pred

print("Initial ensemble trained on combined labeled set.")

# -------------------------------------------------------------------
# 6. Pseudo-Label the Unlabeled Oligomers
# -------------------------------------------------------------------
unlabeled_processed = pipeline_process(oligomer_unlabeled, is_train=False)
# unify columns
for col in feature_cols:
    if col not in unlabeled_processed.columns:
        unlabeled_processed[col] = 0.0
X_unlabeled = unlabeled_processed[feature_cols].fillna(0.0)

pred_log_unlabeled, std_unlabeled = ensemble_predict(X_unlabeled)
pred_unlabeled = np.expm1(pred_log_unlabeled)

unlabeled_processed["pseudo_T80"] = pred_unlabeled
unlabeled_processed["pred_std"] = std_unlabeled
print("Predicted T80 for unlabeled oligomer set.")

# -------------------------------------------------------------------
# 7. Confidence Filter
#    Only keep pseudo-labeled samples where std is low (0.2?), 
#    and T80 is in [0.5, 200].
# -------------------------------------------------------------------
mask_confident = (
    (unlabeled_processed["pred_std"] < 0.2) &
    (unlabeled_processed["pseudo_T80"] > 0.5) &
    (unlabeled_processed["pseudo_T80"] < 200)
)
pseudo_labeled_subset = unlabeled_processed[mask_confident].copy()
print("Pseudo-labeled subset size:", pseudo_labeled_subset.shape)

pseudo_labeled_subset["T80"] = pseudo_labeled_subset["pseudo_T80"]

# -------------------------------------------------------------------
# 8. Combine Pseudo-Labeled with Labeled
# -------------------------------------------------------------------
expanded_df = pd.concat([combined_labeled, pseudo_labeled_subset], ignore_index=True)
expanded_df.drop_duplicates(subset=["Batch_ID"], inplace=True)
print("Expanded dataset size (labeled + pseudo-labeled):", expanded_df.shape)

# Mark original IDs to evaluate on the original 42-labeled
original_ids = set(train_main["Batch_ID"])
expanded_df["is_original"] = expanded_df["Batch_ID"].isin(original_ids)

# -------------------------------------------------------------------
# 9. Final Model on Expanded Dataset
#     We'll retrain the best RF from above for final predictions.
# -------------------------------------------------------------------
expanded_processed = pipeline_process(expanded_df, is_train=True)
for col in feature_cols:
    if col not in expanded_processed.columns:
        expanded_processed[col] = 0.0

X_expanded = expanded_processed[feature_cols].fillna(0.0)
y_expanded = expanded_processed["T80_log"] if "T80_log" in expanded_processed else np.log1p(expanded_processed["T80"])

best_rf_final = search_rf.best_estimator_
best_rf_final.fit(X_expanded, y_expanded)

# Evaluate on original 42-labeled subset
original_subset = expanded_processed[expanded_processed["is_original"]]
X_original = original_subset[feature_cols].fillna(0.0)
y_original = original_subset["T80"]  # raw T80
pred_log_original = best_rf_final.predict(X_original)
pred_original = np.expm1(pred_log_original)
score_msle = mean_squared_log_error(y_original, pred_original)
print(f"MSLE on original 42-labeled set after pseudo-labeling + ensemble confidence: {score_msle:.4f}")

# -------------------------------------------------------------------
# 10. Predict on Competition Test
# -------------------------------------------------------------------
test_processed = pipeline_process(test_df, is_train=False)
for col in feature_cols:
    if col not in test_processed.columns:
        test_processed[col] = 0.0

X_test = test_processed[feature_cols].fillna(0.0)
pred_test_log = best_rf_final.predict(X_test)
pred_test_t80 = np.expm1(pred_test_log)

submission = pd.DataFrame({
    "Batch_ID": test_processed["Batch_ID"],
    "T80": pred_test_t80
})

submission_file = "submission.csv"
submission.to_csv(submission_file, index=False)
print(f"Saved final submission to: {submission_file}")
print(submission.head())