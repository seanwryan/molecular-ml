import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
BASE_DIR = os.getcwd()  # Current working directory
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Create the images directory if it doesn't exist
os.makedirs(IMAGES_DIR, exist_ok=True)

# Define file paths
train_file = os.path.join(DATA_DIR, "train.csv")

# Load dataset
train_df = pd.read_csv(train_file)

# -----------------------------------------------------------------------------
# 1. Correlation with T80: Identify top correlated features
# -----------------------------------------------------------------------------
numeric_cols = train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
corr_matrix = train_df[numeric_cols].corr()

# Sort features by absolute correlation with T80 (excluding T80 itself)
corr_with_T80 = corr_matrix["T80"].drop("T80").abs().sort_values(ascending=False)

# Select top 15 features (example) correlated with T80
top_features = corr_with_T80.head(15).index.tolist()

print("Top features correlated with T80 (by absolute value):")
print(corr_with_T80.head(15))

# Create subset DataFrame for T80 + top features
top_corr_df = train_df[["T80"] + top_features]

# Plot correlation heatmap for top features
plt.figure(figsize=(10, 8))
sns.heatmap(top_corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap: T80 & Top 15 Correlated Features")
plt.tight_layout()
save_path = os.path.join(IMAGES_DIR, "correlation_top_features.png")
plt.savefig(save_path)
plt.show()

# -----------------------------------------------------------------------------
# 2. Grouping Features for Separate Correlation Matrices
# -----------------------------------------------------------------------------
# Example grouping: DOS-like features vs. basic descriptors
dos_features = [col for col in numeric_cols if "DOS" in col]
basic_descriptors = [
    "Mass", "NumHeteroatoms", "HAcceptors", "HDonors", 
    "NumRotatableBonds", "PolarSurfaceArea", "DipoleMoment"
    # Add or remove columns as needed
]

# (a) Correlation among DOS-like features + T80
dos_df = train_df[["T80"] + dos_features]

plt.figure(figsize=(12, 10))
sns.heatmap(dos_df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap: T80 & DOS Features")
plt.tight_layout()
save_path = os.path.join(IMAGES_DIR, "correlation_dos_features.png")
plt.savefig(save_path)
plt.show()

# (b) Correlation among basic descriptors + T80
basic_desc = [col for col in basic_descriptors if col in numeric_cols]  # Ensure column exists
basic_df = train_df[["T80"] + basic_desc]

plt.figure(figsize=(8, 6))
sns.heatmap(basic_df.corr(), cmap="coolwarm", annot=True, fmt=".2f", square=True)
plt.title("Correlation Heatmap: T80 & Basic Descriptors")
plt.tight_layout()
save_path = os.path.join(IMAGES_DIR, "correlation_basic_descriptors.png")
plt.savefig(save_path)
plt.show()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\nEDA Refinements Summary:")
print("- Computed correlation with T80 and identified top correlated features.")
print("- Created separate correlation heatmaps for DOS-like features and basic descriptors.")
print(f"- Saved all plots in '{IMAGES_DIR}' folder.\n")