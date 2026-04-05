"""
Initialises Kubernetes storage with heart.csv dataset
Matches YAML path
"""
import os
import pandas as pd

# 1. Path from YOUR Kubernetes YAML
K8S_STORAGE_PATH = "/data/heart-project"
# 2. Path to uploaded heart.csv (team should update this to their local path)
DATASET_PATH = "heart.csv"  # e.g., if file is in Downloads mine is in downloads reasons this path is this way: "/home/user/Downloads/heart.csv"


os.makedirs(K8S_STORAGE_PATH, exist_ok=True)

# Load dataset
try:
    df = pd.read_csv(DATASET_PATH)
    # Save to Kubernetes storage path
    output_file = f"{K8S_STORAGE_PATH}/heart.csv"
    df.to_csv(output_file, index=False)
    
    # Confirm success
    print(f"SUCCESS: Your heart.csv saved to Kubernetes storage")
    print(f"File location: {output_file}")
    print(f"Dataset stats: {len(df)} rows | {len(df.columns)} columns")
    print("Database/code can now access the dataset")
except Exception as e:
    print(f"Error: {str(e)} | Fix: Update 'DATASET_PATH' to where heart.csv is saved")