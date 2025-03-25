import os
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.utils.fixes import loguniform

# XGBoost
from xgboost import XGBRegressor

# LightGBM
import lightgbm as lgb

# CatBoost
from catboost import CatBoostRegressor

# -------------------------------------------------------------------
# 1. (Option A) Load Processed Data from prefeature.py
# -------------------------------------------------------------------
# If prefeature.py already saves a CSV with processed features, just load that:
# data = pd.read_csv("processed_data.csv")

# -------------------------------------------------------------------
# 1. (Option B) Or re-run partial pipeline here
# -------------------------------------------------------------------
# For demonstration, let's do the entire pipeline again quickly, 
# but you can also copy/paste from prefeature.py or import from a function.

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
train_file = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(train_file)
data = train_df.copy()

# 1B(i). PCA on DOS features
numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
dos_features = [col for col in numeric_cols if "DOS" in col]

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, svd_solver='full')
dos_data = data[dos_features].values
dos_pca = pca.fit_transform(dos_data)

pca_columns = [f"DOS_PCA_{i+1}" for i in range(dos_pca.shape[1])]
for i, col_name in enumerate(pca_columns):
    data[col_name] = dos_pca[:, i]

data.drop(columns=dos_features, inplace=True)

# 1B(ii). RDKit descriptors
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

# 1B(iii). Filter invalid/constant columns
descriptor_cols = [c for c in data.columns if c.startswith("rd_")]
invalid_cols = []
for col in descriptor_cols:
    s = data[col]
    if s.isna().any() or np.isinf(s).any() or s.nunique() <= 1:
        invalid_cols.append(col)
data.drop(columns=invalid_cols, inplace=True, errors="ignore")

# 1B(iv). Prune correlated descriptors
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

# 1B(v). Log transform T80
data["T80_log"] = np.log1p(data["T80"])

# -------------------------------------------------------------------
# 2. Quick Feature Importance Pruning
# -------------------------------------------------------------------
# We'll do a simple approach: train a small RandomForest and drop features with negligible importance
numeric_cols_final = data.select_dtypes(include=["float64", "int64"]).columns.drop(["T80", "T80_log"], errors="ignore")

X = data[numeric_cols_final]
y = data["T80_log"]

temp_rf = RandomForestRegressor(n_estimators=50, random_state=42)
temp_rf.fit(X, y)

importances = pd.Series(temp_rf.feature_importances_, index=numeric_cols_final)
# Let's keep features that have importance above a small threshold, or keep top N
importance_threshold = 0.001  # Adjust as needed
important_features = importances[importances > importance_threshold].index.tolist()

print(f"Feature Importance Pruning: Kept {len(important_features)} / {X.shape[1]} features "
      f"(importance > {importance_threshold}).\n")

X = X[important_features]  # refine X

# -------------------------------------------------------------------
# 3. Define CV and MSE scorer
# -------------------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def mse_scorer(estimator, X_, y_true):
    y_pred = estimator.predict(X_)
    return mean_squared_error(y_true, y_pred)

# scikit-learn expects a "higher is better" metric for maximizing,
# so we pass negative MSE
neg_mse_scorer = make_scorer(mse_scorer, greater_is_better=False)

# -------------------------------------------------------------------
# 4. Expanded Parameter Grids & Model List
# -------------------------------------------------------------------
# Slightly broader random forest param distribution
rf_param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5, 0.75]
}

# Slightly broader XGB param distribution
xgb_param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": loguniform(1e-3, 1e-1),
    "colsample_bytree": [0.6, 0.8, 1.0],
    "subsample": [0.6, 0.8, 1.0]
}

# LightGBM param distribution
lgb_param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [-1, 5, 10, 20],  # -1 means no limit in LightGBM
    "learning_rate": loguniform(1e-3, 1e-1),
    "num_leaves": [31, 63, 127],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "subsample": [0.6, 0.8, 1.0]
}

# CatBoost param distribution
cat_param_dist = {
    "iterations": [50, 100, 200, 300],
    "depth": [4, 6, 8, 10],
    "learning_rate": loguniform(1e-3, 1e-1),
    "subsample": [0.6, 0.8, 1.0]
}

# -------------------------------------------------------------------
# 5. Run RandomizedSearch for each model
# -------------------------------------------------------------------
from sklearn.model_selection import RandomizedSearchCV
import time

models = {
    "RandomForest": (RandomForestRegressor(random_state=42), rf_param_dist),
    "XGBoost": (XGBRegressor(random_state=42, eval_metric="rmse"), xgb_param_dist),
    "LightGBM": (lgb.LGBMRegressor(random_state=42), lgb_param_dist),
    "CatBoost": (CatBoostRegressor(random_state=42, verbose=0), cat_param_dist)
}

results = {}

for model_name, (estimator, param_dist) in models.items():
    print(f"--- Tuning {model_name} ---")
    searcher = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=12,  # slightly expanded, adjust for runtime
        scoring="neg_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    searcher.fit(X, y)
    elapsed = time.time() - start_time
    
    best_params = searcher.best_params_
    best_score = -searcher.best_score_  # negative MSE => MSE
    results[model_name] = (best_params, best_score)
    
    print(f"{model_name} Best Params: {best_params}")
    print(f"{model_name} Best Score (CV MSE on log scale): {best_score:.3f}")
    print(f"Elapsed time: {elapsed:.1f} seconds\n")

# -------------------------------------------------------------------
# 6. Compare Final Results
# -------------------------------------------------------------------
print("--- Final Model Comparison (Lower MSE on Log Scale is Better) ---")
for model_name, (params, score) in results.items():
    print(f"{model_name}: MSE (log T80) = {score:.3f}")

best_model_name = min(results, key=lambda m: results[m][1])
print(f"\nBest model under these settings is: {best_model_name}, "
      f"with MSE (log T80) = {results[best_model_name][1]:.3f}\n")

print("Next Steps:")
print("1. Possibly increase 'n_iter' or expand parameter grids further.")
print("2. Adjust feature-importance threshold or domain-based filtering.")
print("3. Evaluate final chosen model on hold-out set or test set, and consider MSLE if relevant.")