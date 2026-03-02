import os
import numpy as np
import pandas as pd
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def preprocess():
    # Load the raw dataset
    df = pd.read_csv(f"{RAW_DATA_DIR}/initial_dataset.csv")

    # Handle missing values (as per EDA findings)
    df["RestingBP"] = df["RestingBP"].replace(0, np.nan)
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)

    # Save the cleaned dataset for Nested CV
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    processed_path = f"{PROCESSED_DATA_DIR}/clean_dataset.csv"
    df.to_csv(processed_path, index=False)

    # PREVENTION OF DATA LEAKAGE
    # I deliberately DO NOT perform the following operations in this script:
    # 1. Imputation (e.g., replacing NaNs with median)
    # 2. Feature Scaling (e.g., StandardScaler)
    # 3. Categorical Encoding (e.g., OneHotEncoder)
    #
    # Performing these transformations on the entire dataset globally would 
    # cause Data Leakage during the Nested Cross-Validation phase, as the test 
    # folds would influence the learned parameters (like global median or mean).
    # Instead, these steps are strictly delegated to Scikit-Learn Pipelines 
    # in the modelling phase, ensuring they are computed dynamically ONLY 
    # on the training folds.

if __name__ == "__main__":
    preprocess()