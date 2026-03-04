import joblib
import json
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from confidence_intervals import get_confidence_interval

from config import PROCESSED_DATA_DIR, RESULTS_DIR

def train_and_evaluate():
    data_path = f"{PROCESSED_DATA_DIR}/clean_dataset.csv"
    df = pd.read_csv(data_path)
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # FastingBs is numerical but has only 2 unique values (0 and 1). 
    # Discard it now and use passthrough to keep it in the pipeline without transformation, as it is already clean and binary. 
    # This prevents unnecessary imputation or scaling that could distort its meaning.
    numerical_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    categorical_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

    # CATEGORICAL PIPELINE
    # Imputation strategy for categorical features is set to most_frequent (for future cases where missing values might be present)
    base_cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    # NUMERICAL PIPELINE
    # StandardScaler must be applied before KNNImputer. If we impute raw data, Euclidean distances will be heavily dominated by features with large scales (like Cholesterol), 
    # distorting the nearest neighbors. StandardScaler handles NaNs safely by ignoring them during the mean/variance calculation.
    base_num_pipeline = Pipeline([
        ("scaler", StandardScaler()), 
        # For numerical features, KNN imputation is applied for missing values with default parameters (n_neighbors=5).
        ("imputer", KNNImputer())
    ])

    #
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", base_num_pipeline, numerical_features),
            ("cat", base_cat_pipeline, categorical_features),
        ],
        remainder="passthrough"
    )

    models = {
        "Logistic_Regression": {
            "estimator": Pipeline([
                ("prep", preprocessor), 
                ("clf", LogisticRegression(max_iter=2000, random_state=42, solver='saga'))
            ]),
            "param_grid": {
                "clf__C": [0.01, 0.1, 1.0, 10.0], 
                "clf__l1_ratio": [0, 0.5, 1], # 0=Ridge(L2), 1=Lasso(L1), 0.5=ElasticNet
                "clf__class_weight": [None, "balanced"]
            }
        },
        "KNN": {
            "estimator": Pipeline([
                ("prep", preprocessor), 
                # At this point, numericals are scaled but categoricals (OHE) have a max variance of 0.25.
                # To prevent categorical features from losing weight in the KNN Euclidean distance computation, a global StandardScaler is applied. 
                # It scales categoricals to unit variance and acts as a neutral operation on the already-scaled numericals.
                ("scaler_all", StandardScaler()),
                ("clf", KNeighborsClassifier())
            ]),
            "param_grid": {
                "clf__n_neighbors": [3, 5, 7, 9, 11], 
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"]
            }
        },
        "Random_Forest": {
            "estimator": Pipeline([
                ("prep", preprocessor), 
                ("clf", RandomForestClassifier(random_state=42))
            ]),
            "param_grid": {
                "clf__n_estimators": [50, 100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5],
                "clf__class_weight": [None, "balanced"]
            }
        }
    }

    # Repeated Nested Cross-Validation
    n_split = 10
    n_repeats = 10
    total_folds = n_split * n_repeats
    # 10-Times Repeated Stratified 10-Fold Nested CV: 100 total train/test evaluations per model
    outer_cv = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeats, random_state=42)
    # 10-Times Repeated Stratified 5-Fold CV for inner hyperparameter tuning: 50 total evaluations per fold
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    all_results = {}

    for model_name, config in models.items():
        print(f"\n{'='*50}\nTraining Model: {model_name}\n{'='*50}")
        
        outer_accuracies, outer_f1, outer_kappa = [], [], []
        y_true_all, y_prob_all = [], []
        total_instances = 0
        total_successes = 0
        
        for i, (train_ix, test_ix) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

            # Inner CV for Hyperparameter Tuning
            clf_search = GridSearchCV(config["estimator"], config["param_grid"], cv=inner_cv, scoring="f1", n_jobs=-1)
            clf_search.fit(X_train, y_train)

            best_model_fold = clf_search.best_estimator_
            y_pred = best_model_fold.predict(X_test)
            y_prob = best_model_fold.predict_proba(X_test)[:, 1]
            
            # Metrics Evaluation
            fold_acc = accuracy_score(y_test, y_pred)
            outer_accuracies.append(fold_acc)
            outer_f1.append(f1_score(y_test, y_pred))
            outer_kappa.append(cohen_kappa_score(y_test, y_pred))
            
            # Exact counts for Confidence Intervals
            fold_instances = len(y_test)
            fold_successes = accuracy_score(y_test, y_pred, normalize=False)
            total_instances += fold_instances
            total_successes += fold_successes
            
            y_true_all.extend(y_test.values)
            y_prob_all.extend(y_prob)

            print(f"Fold {i+1}/{total_folds} | F1: {outer_f1[-1]:.4f} | Acc: {outer_accuracies[-1]:.4f} | Kappa: {outer_kappa[-1]:.4f}")
            print(f"Params: {clf_search.best_params_}\n")

        # Aggregate Results
        mean_acc = np.mean(outer_accuracies)
        mean_f1 = np.mean(outer_f1)
        mean_kappa = np.mean(outer_kappa)
        ci_acc_lower, ci_acc_upper = get_confidence_interval(total_successes, total_instances, confidence=0.99)

        print(f"\n--- {model_name} Results ---")
        print(f"Total Instances Evaluated (N): {total_instances}")
        print(f"Accuracy:    {mean_acc:.4f} (99% CI: {ci_acc_lower:.4f} - {ci_acc_upper:.4f})")
        print(f"F1-Score:    {mean_f1:.4f}")
        print(f"Kappa Stat:  {mean_kappa:.4f}")

        # Save results to JSON file
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_file = f"{RESULTS_DIR}/{model_name.lower().replace('_', '')}.json"
        
        results = {
            "model": model_name,
            "total_instances": int(total_instances),
            "total_successes": int(total_successes),
            "metrics": {
                "accuracy": float(mean_acc),
                "f1_score": float(mean_f1),
                "kappa_stat": float(mean_kappa),
                "confidence_interval_99": {
                    "lower": float(ci_acc_lower),
                    "upper": float(ci_acc_upper)
                }
            },
            "folds": []
        }
        
        # Add results for each fold
        for i in range(len(outer_accuracies)):
            fold_result = {
                "fold": i + 1,
                "accuracy": float(outer_accuracies[i]),
                "f1_score": float(outer_f1[i]),
                "kappa_stat": float(outer_kappa[i])
            }
            results["folds"].append(fold_result)
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")

        # Store for PKL dump
        all_results[model_name] = {
            'accuracies': outer_accuracies,
            'f1_scores': outer_f1,
            'kappa_scores': outer_kappa,
            'y_true': y_true_all,
            'y_prob': y_prob_all
        }

    joblib.dump(all_results, f"{RESULTS_DIR}/cv_evaluation_results.pkl")

if __name__ == "__main__":
    train_and_evaluate()