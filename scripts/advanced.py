import os
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')

# -------------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

train_file = os.path.join(DATA_DIR, "train.csv")
test_file = os.path.join(DATA_DIR, "test.csv")
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# -------------------------------------------------------------------
# 2. 3D Embedding + RDKit Descriptors
# -------------------------------------------------------------------
# We'll define a function that:
#  - Adds hydrogens
#  - Embeds a 3D conformer
#  - Optimizes it with MMFF
#  - Extracts some 2D/3D descriptors
#  - If embedding fails, we revert to 2D descriptors or fill with np.nan

desc_cols = [
    "MolWt", "NumHBA", "NumHBD", "MolLogP", 
    "Asphericity", "RadiusOfGyration", "TPSA",
    "NumRings", "NumRotatableBonds", "NumHeteroatoms"
]
# We can add additional 3D geometry descriptors below
# e.g. Eccentricity, InertialShapeFactor, etc. from rdMolDescriptors

def embed_and_extract_descriptors(smile, random_seed=42):
    """Generate 2D/3D RDKit descriptors with a 3D embedding & optimization."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return [np.nan]*len(desc_cols)
    try:
        # Add hydrogens
        mol = Chem.AddHs(mol)
        # Embed 3D conformer
        AllChem.EmbedMolecule(mol, randomSeed=random_seed)
        # MMFF optimization
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        # If embedding fails, we still try to proceed with 2D descriptors
        pass

    # Now extract descriptors
    # Note: for 3D-based descriptors like Asphericity, RadiusOfGyration, etc., 
    #       we need the 3D conformer. For 2D, standard RDKit calls suffice.
    vals = []
    # 1. MolWt
    vals.append(Descriptors.MolWt(mol))
    # 2. NumHBA
    vals.append(rdMolDescriptors.CalcNumHBA(mol))
    # 3. NumHBD
    vals.append(rdMolDescriptors.CalcNumHBD(mol))
    # 4. MolLogP
    vals.append(Descriptors.MolLogP(mol))
    # 5. Asphericity (3D shape descriptor)
    vals.append(rdMolDescriptors.CalcAsphericity(mol))
    # 6. RadiusOfGyration
    vals.append(rdMolDescriptors.CalcRadiusOfGyration(mol))
    # 7. TPSA (2D measure of polar surface area)
    vals.append(Descriptors.TPSA(mol))
    # 8. NumRings
    vals.append(rdMolDescriptors.CalcNumRings(mol))
    # 9. NumRotatableBonds
    vals.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
    # 10. NumHeteroatoms
    vals.append(rdMolDescriptors.CalcNumHeteroatoms(mol))

    return vals

def generate_3d_descriptors(df):
    """Apply embed_and_extract_descriptors to each SMILES in df."""
    new_desc = []
    for smile in df["Smiles"]:
        vals = embed_and_extract_descriptors(smile, random_seed=random.randint(1, 999999))
        new_desc.append(vals)
    desc_array = np.array(new_desc)
    desc_df = pd.DataFrame(desc_array, columns=[f"rd_3d_{c}" for c in desc_cols])
    return pd.concat([df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)

# Apply to train and test
train_df = generate_3d_descriptors(train_df)
test_df = generate_3d_descriptors(test_df)

# -------------------------------------------------------------------
# 3. PCA on DOS Features
# -------------------------------------------------------------------
numeric_cols = train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
dos_features = [col for col in numeric_cols if "DOS" in col]

pca = PCA(n_components=0.95, svd_solver='full')
dos_data_train = train_df[dos_features].values
dos_pca_train = pca.fit_transform(dos_data_train)
pca_columns = [f"DOS_PCA_{i+1}" for i in range(dos_pca_train.shape[1])]

for i, col_name in enumerate(pca_columns):
    train_df[col_name] = dos_pca_train[:, i]

# We will transform test similarly, but we must handle the possibility
# that the test has the same DOS features
dos_data_test = test_df[dos_features].values
dos_pca_test = pca.transform(dos_data_test)
for i, col_name in enumerate(pca_columns):
    test_df[col_name] = dos_pca_test[:, i]

# Drop original DOS columns
train_df.drop(columns=dos_features, inplace=True, errors="ignore")
test_df.drop(columns=dos_features, inplace=True, errors="ignore")

# -------------------------------------------------------------------
# 4. Filter Invalid/Constant Columns
# -------------------------------------------------------------------
def filter_invalid_constant_cols(train, test):
    numeric_train = train.select_dtypes(include=["float64", "int64"])
    invalid_cols = []
    for col in numeric_train.columns:
        s = train[col]
        if s.isna().any() or np.isinf(s).any() or s.nunique() <= 1:
            invalid_cols.append(col)
    # Drop from train/test
    train.drop(columns=invalid_cols, inplace=True, errors="ignore")
    test.drop(columns=invalid_cols, inplace=True, errors="ignore")
    return train, test

train_df, test_df = filter_invalid_constant_cols(train_df, test_df)

# -------------------------------------------------------------------
# 5. Prune Correlated Descriptors
# -------------------------------------------------------------------
def prune_correlated(df, threshold=0.95):
    """Drop columns that are correlated above 'threshold' with any other column."""
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop

corr_drop_cols = prune_correlated(train_df, threshold=0.95)
train_df.drop(columns=corr_drop_cols, inplace=True, errors="ignore")
test_df.drop(columns=corr_drop_cols, inplace=True, errors="ignore")

# -------------------------------------------------------------------
# 6. Log-Transform T80
# -------------------------------------------------------------------
train_df["T80_log"] = np.log1p(train_df["T80"])

# -------------------------------------------------------------------
# 7. Model Training (RandomForest) + Cross-Validation
# -------------------------------------------------------------------
# For a small dataset, let's do 5-fold CV to see approximate performance
def msle_scorer(model, X, y):
    # We'll treat y as log(T80), so we exponentiate predictions
    pred_log = model.predict(X)
    # Convert back to original scale
    pred_t80 = np.expm1(pred_log)
    # Compare to actual T80
    actual_t80 = np.expm1(y)
    return mean_squared_log_error(actual_t80, pred_t80)

# Features: all numeric except T80, T80_log
final_numeric_cols = train_df.select_dtypes(include=["float64", "int64"]).columns
final_numeric_cols = final_numeric_cols.drop(["T80", "T80_log"], errors="ignore")

X_train = train_df[final_numeric_cols].values
y_train = train_df["T80_log"].values

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    model.fit(X_tr, y_tr)
    cv_scores.append(msle_scorer(model, X_val, y_val))

print("Cross-Validation MSLE:", np.mean(cv_scores), "+/-", np.std(cv_scores))

# Retrain on full training data
model.fit(X_train, y_train)

# -------------------------------------------------------------------
# 8. Predict on Test & Save Submission
# -------------------------------------------------------------------
# We must replicate the final numeric columns
for col in final_numeric_cols:
    if col not in test_df.columns:
        test_df[col] = 0.0  # fill missing columns
X_test = test_df[final_numeric_cols].fillna(0.0).values

pred_test_log = model.predict(X_test)
pred_test_t80 = np.expm1(pred_test_log)

submission = pd.DataFrame({
    "Batch_ID": test_df["Batch_ID"],
    "T80": pred_test_t80
})

# The competition sample submission says the header is "Batch_ID,T80"
submission_file = "submission.csv"
submission.to_csv(submission_file, index=False)
print(f"Submission file saved: {submission_file}")
print(submission.head())