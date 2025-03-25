import pandas as pd
import os

# Set base directory for the data folder (relative to the project directory)
BASE_DIR = os.path.join(os.getcwd(), "data")

# Define file paths
train_file = os.path.join(BASE_DIR, "train.csv")
test_file = os.path.join(BASE_DIR, "test.csv")
sample_submission_file = os.path.join(BASE_DIR, "sample_submission.csv")

def load_datasets():
    """Load the train, test, and sample submission datasets."""
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    sample_submission_df = pd.read_csv(sample_submission_file)
    return train_df, test_df, sample_submission_df

def explore_dataset(df, name="Dataset"):
    """Print basic info and statistics for a given DataFrame."""
    print(f"--- {name} ---")
    print("Shape:", df.shape)
    print("\nHead:")
    print(df.head())
    
    print("\nData Types & Info:")
    print(df.info())
    
    print("\nDescriptive Statistics:")
    print(df.describe(include="all"))
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # For non-numeric columns, print unique value counts for first few columns (if any)
    non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print("\nUnique value counts for non-numeric columns:")
        for col in non_numeric_cols:
            print(f"Column: {col} - Unique values count: {df[col].nunique()}")
            print(df[col].value_counts().head(), "\n")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    # Load datasets
    train_df, test_df, sample_submission_df = load_datasets()
    
    # Explore each dataset
    explore_dataset(train_df, "Train Dataset")
    explore_dataset(test_df, "Test Dataset")
    explore_dataset(sample_submission_df, "Sample Submission Dataset")