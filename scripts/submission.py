import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------------------------
# 1. Helper Functions for Pipeline Steps
# -------------------------------------------------------------------
def generate_rdkit_descriptors(smiles_str, desc_list):
    """Generate all RDKit descriptors for a given SMILES string."""
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

def prune_correlated(df, cols, threshold=0.95):
    """Drop columns from 'cols' that have correlation above 'threshold'."""
    corr_matrix = df[cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []
    for column in upper.columns:
        if any(upper[column] > threshold):
            to_drop.append(column)
    return list(set(to_drop))

# -------------------------------------------------------------------
# 2. Paths & Data
# -------------------------------------------------------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

train_file = os.path.join(DATA_DIR, "train.csv")
test_file = os.path.join(DATA_DIR, "test.csv")

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# -------------------------------------------------------------------
# 3. Pipeline on TRAIN
# -------------------------------------------------------------------
data_train = train_df.copy()

# 3A. PCA on DOS features
dos_features = [c for c in data_train.columns if "DOS" in c and c not in ["Batch_ID", "Smiles"]]
pca = PCA(n_components=0.95, svd_solver='full')
dos_data_train = data_train[dos_features].values
dos_pca_train = pca.fit_transform(dos_data_train)

pca_columns = [f"DOS_PCA_{i+1}" for i in range(dos_pca_train.shape[1])]
for i, col_name in enumerate(pca_columns):
    data_train[col_name] = dos_pca_train[:, i]

data_train.drop(columns=dos_features, inplace=True)

# 3B. Generate RDKit descriptors
desc_list = Descriptors._descList  # list of (name, function)
desc_names = [d[0] for d in desc_list]

desc_data_train = data_train["Smiles"].apply(lambda s: generate_rdkit_descriptors(s, desc_list))
desc_df_train = pd.DataFrame(desc_data_train.tolist(), columns=desc_names)
desc_df_train.columns = [f"rd_{c}" for c in desc_df_train.columns]

data_train = pd.concat([data_train.reset_index(drop=True),
                        desc_df_train.reset_index(drop=True)], axis=1)

# Remove duplicates if any
if data_train.columns.duplicated().any():
    data_train = data_train.loc[:, ~data_train.columns.duplicated()]

# 3C. Filter invalid/constant columns
descriptor_cols = [c for c in data_train.columns if c.startswith("rd_")]
invalid_cols = []
for col in descriptor_cols:
    s = data_train[col]
    if s.isna().any() or np.isinf(s).any() or s.nunique() <= 1:
        invalid_cols.append(col)
data_train.drop(columns=invalid_cols, inplace=True, errors="ignore")

# 3D. Prune correlated descriptors
desc_cols_now = [c for c in data_train.columns if c.startswith("rd_") or c.startswith("DOS_PCA_")]
drop_corr = prune_correlated(data_train, desc_cols_now, threshold=0.95)
data_train.drop(columns=drop_corr, inplace=True)

# 3E. Log-transform T80
data_train["T80_log"] = np.log1p(data_train["T80"])

# 3F. Feature importance pruning (optional step)
# For simplicity, let's skip or do a quick pass
all_feats = [c for c in data_train.select_dtypes(include=[np.number]).columns
             if c not in ["T80", "T80_log"]]

X_train = data_train[all_feats].values
y_train = data_train["T80_log"].values

temp_rf = RandomForestRegressor(n_estimators=50, random_state=42)
temp_rf.fit(X_train, y_train)

importances = temp_rf.feature_importances_
feat_import_series = pd.Series(importances, index=all_feats)
keep_threshold = 0.001
keep_feats = feat_import_series[feat_import_series > keep_threshold].index.tolist()

# Final train features
X_train = data_train[keep_feats].values

# -------------------------------------------------------------------
# 4. Train Final RF with Best Hyperparams (from your previous search)
# -------------------------------------------------------------------
best_params = {
    'n_estimators': 300,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'log2',
    'max_depth': None,
    'random_state': 42
}

final_rf = RandomForestRegressor(**best_params)
final_rf.fit(X_train, y_train)

print("Trained final RandomForest on the full training set.\n")

# -------------------------------------------------------------------
# 5. Pipeline on TEST
# -------------------------------------------------------------------
data_test = test_df.copy()

# 5A. Apply same PCA transform for DOS features
# NOTE: The test set also has the same DOS columns
dos_data_test = data_test[dos_features].values  # same dos_features from train
dos_pca_test = pca.transform(dos_data_test)     # use the SAME PCA fit

for i, col_name in enumerate(pca_columns):
    data_test[col_name] = dos_pca_test[:, i]

data_test.drop(columns=dos_features, inplace=True)

# 5B. RDKit descriptors
desc_data_test = data_test["Smiles"].apply(lambda s: generate_rdkit_descriptors(s, desc_list))
desc_df_test = pd.DataFrame(desc_data_test.tolist(), columns=desc_names)
desc_df_test.columns = [f"rd_{c}" for c in desc_df_test.columns]

data_test = pd.concat([data_test.reset_index(drop=True),
                       desc_df_test.reset_index(drop=True)], axis=1)

if data_test.columns.duplicated().any():
    data_test = data_test.loc[:, ~data_test.columns.duplicated()]

# 5C. Filter the same invalid/constant columns we removed in train
data_test.drop(columns=invalid_cols, inplace=True, errors="ignore")

# 5D. Prune the same correlated columns
data_test.drop(columns=drop_corr, inplace=True, errors="ignore")

# 5E. Final feature set must match train
# We'll keep only the columns we used for final training
for c in keep_feats:
    if c not in data_test.columns:
        data_test[c] = 0.0  # or np.nan; ideally you want consistent columns
X_test = data_test[keep_feats].fillna(0.0).values  # fillna to handle any missing

# -------------------------------------------------------------------
# 6. Predict & Create Submission
# -------------------------------------------------------------------
pred_log = final_rf.predict(X_test)    # predictions in log scale
pred_t80 = np.expm1(pred_log)          # revert log transform => T80

# IMPORTANT: use "Batch_ID" to match the sample submission's column name
submission_df = pd.DataFrame({
    "Batch_ID": data_test["Batch_ID"],
    "T80": pred_t80
})

submission_file = "submission.csv"
submission_df.to_csv(submission_file, index=False)
print(f"Saved submission file to {submission_file}")

# Optional: Print the final DataFrame head
print("\nSample submission:")
print(submission_df.head())