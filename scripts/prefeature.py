import os
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import loguniform

# XGBoost
from xgboost import XGBRegressor

# -------------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

train_file = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(train_file)

data = train_df.copy()

# -------------------------------------------------------------------
# 2. PCA on DOS Features
# -------------------------------------------------------------------
numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
dos_features = [col for col in numeric_cols if "DOS" in col]

pca = PCA(n_components=0.95, svd_solver='full')
dos_data = data[dos_features].values
dos_pca = pca.fit_transform(dos_data)

pca_columns = [f"DOS_PCA_{i+1}" for i in range(dos_pca.shape[1])]
for i, col_name in enumerate(pca_columns):
    data[col_name] = dos_pca[:, i]

# Drop original DOS features
data.drop(columns=dos_features, inplace=True)

print(f"DOS PCA: Reduced {len(dos_features)} DOS features to {len(pca_columns)} PCA components.\n")

# -------------------------------------------------------------------
# 3. Generate RDKit Descriptors for SMILES
# -------------------------------------------------------------------
desc_list = Descriptors._descList  # list of (descriptor_name, function)
desc_names = [d[0] for d in desc_list]

def generate_rdkit_descriptors(smiles_str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        # Return NaNs for invalid SMILES
        return [np.nan] * len(desc_list)
    values = []
    for _, func in desc_list:
        try:
            val = func(mol)
        except:
            val = np.nan
        values.append(val)
    return values

desc_data = data["Smiles"].apply(generate_rdkit_descriptors)
desc_df = pd.DataFrame(desc_data.tolist(), columns=desc_names)

# Rename descriptor columns with a prefix to avoid collisions
desc_df.columns = [f"rd_{c}" for c in desc_df.columns]

# Concatenate descriptor columns
data = pd.concat([data.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)

# Remove any duplicate columns (just in case)
if data.columns.duplicated().any():
    data = data.loc[:, ~data.columns.duplicated()]

print(f"Added {len(desc_names)} RDKit descriptor columns from SMILES (prefixed 'rd_').\n")

# -------------------------------------------------------------------
# 4. Filter Out Invalid or Constant Descriptor Columns
# -------------------------------------------------------------------
before_filter = data.shape[1]
descriptor_cols = [col for col in data.columns if col.startswith("rd_")]

invalid_cols = []
for col in descriptor_cols:
    col_series = data[col]
    # Check for any NaN/Inf
    any_na = col_series.isna().any()
    any_inf = np.isinf(col_series).any()
    # Check if column is constant
    is_constant = (col_series.nunique() <= 1)
    
    if any_na or any_inf or is_constant:
        invalid_cols.append(col)

data.drop(columns=invalid_cols, inplace=True, errors="ignore")
after_filter = data.shape[1]
print(f"Filtered out {len(invalid_cols)} invalid/constant columns. "
      f"Remaining columns: {after_filter} (was {before_filter}).\n")

# -------------------------------------------------------------------
# 5. Optional: Prune Highly Correlated Descriptors
# -------------------------------------------------------------------
def prune_correlated(df, cols, threshold=0.95):
    """
    Drops columns from 'cols' that have a correlation above 'threshold' 
    with any other column. Returns pruned column list.
    """
    corr_matrix = df[cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []
    for column in upper.columns:
        if any(upper[column] > threshold):
            to_drop.append(column)
    return list(set(to_drop))

# Example usage (uncomment if desired):
prune_threshold = 0.95
desc_cols_now = [c for c in data.columns if c.startswith("rd_") or c.startswith("DOS_PCA_")]
drop_corr = prune_correlated(data, desc_cols_now, threshold=prune_threshold)
if drop_corr:
    data.drop(columns=drop_corr, inplace=True)
    print(f"Pruned {len(drop_corr)} descriptors correlated above {prune_threshold}.\n")

# -------------------------------------------------------------------
# 6. Log-Transform T80
# -------------------------------------------------------------------
data["T80_log"] = np.log1p(data["T80"])

# -------------------------------------------------------------------
# 7. Set Up Features/Target for Modeling
# -------------------------------------------------------------------
numeric_cols_final = data.select_dtypes(include=["float64", "int64"]).columns
# Exclude T80 and T80_log from features
numeric_cols_final = numeric_cols_final.drop(["T80", "T80_log"], errors="ignore")

X = data[numeric_cols_final].values
y = data["T80_log"].values

# We'll define a custom scoring function for MSE
def mse_scorer(estimator, X, y_true):
    y_pred = estimator.predict(X)
    return mean_squared_error(y_true, y_pred)

scorer = make_scorer(mse_scorer, greater_is_better=False)  
# scikit-learn expects higher scores to be better, so we use negative sign.

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------------------------------------------------
# 8. Hyperparameter Search for RandomForest
# -------------------------------------------------------------------
rf = RandomForestRegressor(random_state=42)
# Example parameter distributions for random search
rf_param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5]
}

rf_random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_param_dist,
    n_iter=10,  # increase for a more thorough search
    scoring="neg_mean_squared_error",
    cv=cv,
    random_state=42,
    n_jobs=-1
)

rf_random_search.fit(X, y)

print("RandomForest Best Params:")
print(rf_random_search.best_params_)
print(f"RandomForest Best Score (CV MSE on log scale): {-rf_random_search.best_score_:.3f}\n")

# -------------------------------------------------------------------
# 9. Hyperparameter Search for XGBRegressor
# -------------------------------------------------------------------
xgb = XGBRegressor(random_state=42, eval_metric="rmse")

# Example parameter distributions
# 'loguniform' can be used for continuous hyperparams (requires scikit-learn>=1.0)
xgb_param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": loguniform(1e-3, 1e-1),  # between 0.001 and 0.1
    "colsample_bytree": [0.6, 0.8, 1.0],
    "subsample": [0.6, 0.8, 1.0]
}

xgb_random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=xgb_param_dist,
    n_iter=10,  # increase for a more thorough search
    scoring="neg_mean_squared_error",
    cv=cv,
    random_state=42,
    n_jobs=-1
)

xgb_random_search.fit(X, y)

print("XGBoost Best Params:")
print(xgb_random_search.best_params_)
print(f"XGBoost Best Score (CV MSE on log scale): {-xgb_random_search.best_score_:.3f}\n")

# -------------------------------------------------------------------
# 10. Compare Final Results
# -------------------------------------------------------------------
best_rf_mse = -rf_random_search.best_score_
best_xgb_mse = -xgb_random_search.best_score_

print("--- Final Model Comparison (Lower is Better) ---")
print(f"RandomForest MSE (log T80): {best_rf_mse:.3f}")
print(f"XGBoost    MSE (log T80): {best_xgb_mse:.3f}")

if best_xgb_mse < best_rf_mse:
    print("\nXGBoost performed better on the log-scale MSE metric.")
else:
    print("\nRandomForest performed better on the log-scale MSE metric.")

print("\nNext Steps:")
print("1. Increase 'n_iter' or expand parameter grids for a more thorough search.")
print("2. Consider advanced feature engineering or domain-based descriptor filtering.")
print("3. Evaluate final models on an external validation set or compare MSLE on raw T80.\n")