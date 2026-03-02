import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from config import MODELS_DIR, PROCESSED_DATA_DIR

def train_and_evaluate():
    # Load dataset
    data_path = f"{PROCESSED_DATA_DIR}/clean_dataset.csv"
    df = pd.read_csv(data_path)
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # Pre-process pipeline: imputation with median for numerical, one-hot encoding for categorical, scaling for numerical
    # This preprocessor is used in all models to ensure a fair comparison, and is fitted only on the training data in each fold to prevent data leakage.
    numerical_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    categorical_features = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numerical_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough"
    )

    # Setup models and their hyperparameter grids for GridSearchCV
    models = {
        "1R_Baseline": {
            "estimator": Pipeline([("prep", preprocessor), ("clf", DecisionTreeClassifier(max_depth=1, random_state=42))]),
            "param_grid": {"clf__criterion": ["gini", "entropy"]}
        },
        "Logistic_Regression": {
            "estimator": Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=2000, random_state=42))]),
            "param_grid": {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]}
        },
        "KNN": {
            "estimator": Pipeline([("prep", preprocessor), ("clf", KNeighborsClassifier())]),
            "param_grid": {"clf__n_neighbors": [3, 5, 7, 9, 11], "clf__weights": ["uniform", "distance"]}
        },
        "Random_Forest": {
            "estimator": Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(random_state=42))]),
            "param_grid": {
                "clf__n_estimators": [50, 100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5],
                "clf__class_weight": [None, "balanced"]
            }
        }
    }

    # Fixed cross validation splits for reproducibility and to ensure fair comparison across models
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

    all_results = {}

    for model_name, config in models.items():
        print(f"\n{'='*50}\nTraining Model: {model_name}\n{'='*50}")
        
        outer_accuracies, outer_f1, outer_kappa = [], [], []
        y_true_all, y_prob_all = [], []
        
        # Nested CV: Outer Loop
        for i, (train_ix, test_ix) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

            clf_search = GridSearchCV(config["estimator"], config["param_grid"], cv=inner_cv, scoring="accuracy", n_jobs=-1)
            clf_search.fit(X_train, y_train)

            best_model_fold = clf_search.best_estimator_
            y_pred = best_model_fold.predict(X_test)
            y_prob = best_model_fold.predict_proba(X_test)[:, 1]
            
            # Metrics
            outer_accuracies.append(accuracy_score(y_test, y_pred))
            outer_f1.append(f1_score(y_test, y_pred, average="weighted"))
            outer_kappa.append(cohen_kappa_score(y_test, y_pred))
            
            y_true_all.extend(y_test.values)
            y_prob_all.extend(y_prob)

            print(f"Fold {i+1}/10 | Acc: {outer_accuracies[-1]:.4f} | Params: {clf_search.best_params_}")

        # Mean and 99% CI for Accuracy
        mean_acc = np.mean(outer_accuracies)
        std_acc = np.std(outer_accuracies, ddof=1)
        ci_acc = stats.t.interval(0.99, df=len(outer_accuracies)-1, loc=mean_acc, scale=std_acc/np.sqrt(len(outer_accuracies)))
        
        mean_kappa = np.mean(outer_kappa)

        print(f"\n--- {model_name} Results ---")
        print(f"Accuracy:  {mean_acc:.4f} (99% CI: {ci_acc[0]:.4f} - {ci_acc[1]:.4f})")
        print(f"F1-Score:  {np.mean(outer_f1):.4f}")
        print(f"Kappa Stat:{mean_kappa:.4f}")

        all_results[model_name] = {
            'accuracies': outer_accuracies,
            'y_true': y_true_all,
            'y_prob': y_prob_all
        }

        # Final training on the entire dataset with the best hyperparameters found in the nested CV
        final_search = GridSearchCV(config["estimator"], config["param_grid"], cv=outer_cv, scoring="accuracy", n_jobs=-1)
        final_search.fit(X, y)
        best_final_model = final_search.best_estimator_

        # Save pipeline with the best model
        model_path = f"{MODELS_DIR}/{model_name}_final_pipeline.pkl"
        joblib.dump(best_final_model, model_path)

    # Save all results for later analysis
    joblib.dump(all_results, f"{PROCESSED_DATA_DIR}/cv_evaluation_results.pkl")

if __name__ == "__main__":
    import os
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_and_evaluate()