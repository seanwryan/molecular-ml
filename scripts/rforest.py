import os
import time
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.decomposition import PCA
from sklearn.model_selection import (
    RepeatedKFold, 
    RandomizedSearchCV,
    cross_val_score
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import loguniform

# -------------------------------------------------------------------
# 1. Load Data & Basic Pipeline
# -------------------------------------------------------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
train_file = os.path.join(DATA_DIR, "train.csv")

train_df = pd.read_csv(train_file)
data = train_df.copy()

# 1A. PCA on DOS features
numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
dos_features = [col for col in numeric_cols if "DOS" in col]

pca = PCA(n_components=0.95, svd_solver='full')
dos_data = data[dos_features].values
dos_pca = pca.fit_transform(dos_data)

pca_columns = [f"DOS_PCA_{i+1}" for i in range(dos_pca.shape[1])]
for i, col_name in enumerate(pca_columns):
    data[col_name] = dos_pca[:, i]

data.drop(columns=dos_features, inplace=True)

# 1B. RDKit descriptors
desc_list = Descriptors._descList
desc_names = [d[0] for d in desc_list]

def generate_rdkit_descriptors(smiles_str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
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
desc_df.columns = [f"rd_{c}" for c in desc_df.columns]
data = pd.concat([data.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)

if data.columns.duplicated().any():
    data = data.loc[:, ~data.columns.duplicated()]

# 1C. Filter invalid/constant columns
descriptor_cols = [c for c in data.columns if c.startswith("rd_")]
invalid_cols = []
for col in descriptor_cols:
    s = data[col]
    if s.isna().any() or np.isinf(s).any() or s.nunique() <= 1:
        invalid_cols.append(col)
data.drop(columns=invalid_cols, inplace=True, errors="ignore")

# 1D. Prune correlated descriptors
def prune_correlated(df, cols, threshold=0.95):
    corr_matrix = df[cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []
    for column in upper.columns:
        if any(upper[column] > threshold):
            to_drop.append(column)
    return list(set(to_drop))

desc_cols_now = [c for c in data.columns if c.startswith("rd_") or c.startswith("DOS_PCA_")]
drop_corr = prune_correlated(data, desc_cols_now, threshold=0.95)
data.drop(columns=drop_corr, inplace=True)

# 1E. Log-transform T80
data["T80_log"] = np.log1p(data["T80"])

# -------------------------------------------------------------------
# 2. Placeholder: Domain-Based Filtering (Optional)
# -------------------------------------------------------------------
# If you have chemistry insights about which descriptors are physically relevant,
# you could drop or transform them here. Example:
# domain_relevant = ["rd_MolWt", "rd_TPSA", ...] # hypothetical
# data = data[["T80_log", "Smiles"] + domain_relevant + pca_columns]

# -------------------------------------------------------------------
# 3. Feature Importance Pruning
# -------------------------------------------------------------------
numeric_cols_final = data.select_dtypes(include=["float64", "int64"]).columns.drop(["T80", "T80_log"], errors="ignore")

X = data[numeric_cols_final]
y = data["T80_log"]

# 3A. Quick pass with a small RF to identify top features
temp_rf = RandomForestRegressor(n_estimators=50, random_state=42)
temp_rf.fit(X, y)

importances = pd.Series(temp_rf.feature_importances_, index=numeric_cols_final)
importance_threshold = 0.001  # can be adjusted
important_features = importances[importances > importance_threshold].index.tolist()

X = X[important_features]
print(f"Feature Importance Pruning: Kept {len(important_features)} / {len(importances)} "
      f"features (importance > {importance_threshold}).\n")

# -------------------------------------------------------------------
# 4. Repeated Cross-Validation Setup
# -------------------------------------------------------------------
# Using RepeatedKFold to reduce variance in small datasets
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

def mse_scorer(estimator, X_, y_true):
    y_pred = estimator.predict(X_)
    return mean_squared_error(y_true, y_pred)

neg_mse_scorer = make_scorer(mse_scorer, greater_is_better=False)

# -------------------------------------------------------------------
# 5. Extended Hyperparameter Search for RandomForest
# -------------------------------------------------------------------
rf_param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5, 0.75]
}

rf_model = RandomForestRegressor(random_state=42)

searcher = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_param_dist,
    n_iter=20,  # more iterations for a broader search
    scoring="neg_mean_squared_error",  # or neg_mse_scorer
    cv=cv,
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
searcher.fit(X, y)
elapsed = time.time() - start_time

best_params = searcher.best_params_
best_score = -searcher.best_score_  # negative MSE => MSE

print("--- RandomForest Refinement Results ---")
print(f"Best Params: {best_params}")
print(f"Best CV Score (MSE on log T80): {best_score:.3f}")
print(f"Elapsed time: {elapsed:.1f} seconds\n")

# -------------------------------------------------------------------
# 6. Final Check: Evaluate the Best RF
# -------------------------------------------------------------------
best_rf = searcher.best_estimator_

# We'll do a repeated cross-validation manually to see final MSE
scores = cross_val_score(best_rf, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
final_mse = -scores.mean()
final_std = scores.std()

print(f"Final RF with best params: MSE (log T80) = {final_mse:.3f} ± {final_std:.3f}\n")

print("Next Steps:")
print("1. Consider further domain-based descriptor filtering if relevant.")
print("2. Potentially stack/blend this RF with other models.")
print("3. Evaluate on original T80 scale or MSLE if the competition requires it.")
print("4. Retrain on the entire training set and generate final predictions for the test set.")