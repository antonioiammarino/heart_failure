import json
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import RepeatedStratifiedKFold
from confidence_intervals import get_nadeau_bengio_ci

from config import PROCESSED_DATA_DIR, RESULTS_DIR

class OneRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, min_bucket_size=6, numerical_feature_mask=None):
        self.min_bucket_size = min_bucket_size
        self.numerical_feature_mask = numerical_feature_mask

    def _discretise_numerical(self, values, y):
        # Missing values are handled by treating them as a separate category
        valid_mask = ~pd.isna(values)
        valid_vals = values[valid_mask]
        valid_y = y[valid_mask]
        
        nan_y = y[~valid_mask]
        nan_error = 0
        nan_class = self.default_class_
        
        # Majority class among missing values (if any) and count errors
        if len(nan_y) > 0:
            classes, counts = np.unique(nan_y, return_counts=True)
            nan_class = classes[np.argmax(counts)]
            nan_error = np.sum(nan_y != nan_class)
            
        if len(valid_vals) == 0:
             return [], [self.default_class_], nan_error, nan_class

        # Sort by feature values (non-missing only)
        order = np.argsort(valid_vals)
        sorted_vals = valid_vals[order]
        sorted_y = valid_y[order]

        # Create bins based on changes in class distribution and feature values, ensuring minimum bin size
        bins = []
        bin_starts = [0]
        current_bin = {}

        for idx in range(len(sorted_y)):
            cls = sorted_y[idx]
            current_bin[cls] = current_bin.get(cls, 0) + 1

            majority_count = max(current_bin.values())
            majority_class = max(current_bin, key=current_bin.get)

            # Check if bin dimension is sufficient and if a split can be made at the next point
            if majority_count >= self.min_bucket_size and idx < len(sorted_y) - 1:
                next_cls = sorted_y[idx + 1]
                
                # Only split if the next class is different and the feature value changes (to avoid splitting on identical values)
                if next_cls != majority_class and sorted_vals[idx] != sorted_vals[idx + 1]:
                    bins.append(dict(current_bin))
                    bin_starts.append(idx + 1)
                    current_bin = {}

        # Add last bin
        if current_bin:
            bins.append(dict(current_bin))

        # Merge adjacent bins with the same majority class
        merged_bins = [bins[0]]
        merged_starts = [bin_starts[0]]
        for i in range(1, len(bins)):
            prev_majority = max(merged_bins[-1], key=merged_bins[-1].get)
            curr_majority = max(bins[i], key=bins[i].get)
            if curr_majority == prev_majority:
                for cls, cnt in bins[i].items():
                    merged_bins[-1][cls] = merged_bins[-1].get(cls, 0) + cnt
            else:
                merged_bins.append(bins[i])
                merged_starts.append(bin_starts[i])

        # Build breakpoints
        breakpoints = []
        for i in range(1, len(merged_starts)):
            end_prev = merged_starts[i] - 1
            start_cur = merged_starts[i]
            bp = (sorted_vals[end_prev] + sorted_vals[start_cur]) / 2.0
            breakpoints.append(bp)

        bin_classes = []
        # Calculate total error including the error from missing values
        total_error = nan_error
        for b in merged_bins:
            majority_class = max(b, key=b.get)
            bin_classes.append(majority_class)
            total_error += sum(cnt for cls, cnt in b.items() if cls != majority_class)

        return breakpoints, bin_classes, total_error, nan_class

    def _evaluate_categorical(self, values, y):
        rules = {}
        error = 0
        for val in np.unique(values):
            mask = values == val
            subset_y = y[mask]
            classes, counts = np.unique(subset_y, return_counts=True)
            majority_class = classes[np.argmax(counts)]
            rules[val] = majority_class
            error += np.sum(subset_y != majority_class)
        return rules, error

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        n_features = X.shape[1]
        best_error = float('inf')

        classes, counts = np.unique(y, return_counts=True)
        self.default_class_ = classes[np.argmax(counts)]
        self.classes_ = classes

        if self.numerical_feature_mask is None:
            num_mask = np.zeros(n_features, dtype=bool)
        else:
            num_mask = np.asarray(self.numerical_feature_mask, dtype=bool)

        for feat_idx in range(n_features):
            col = X[:, feat_idx]

            if num_mask[feat_idx]:
                breakpoints, bin_classes, error, nan_class = self._discretise_numerical(col, y)
                rule = {
                    "type": "numerical", 
                    "breakpoints": breakpoints, 
                    "bin_classes": bin_classes,
                    # Save rule for NaN values to use during prediction
                    "nan_class": nan_class 
                }
            else:
                cat_rules, error = self._evaluate_categorical(col, y)
                rule = {"type": "categorical", "mapping": cat_rules}

            if error < best_error:
                best_error = error
                self.best_feature_idx_ = feat_idx
                self.best_rule_ = rule

        return self

    def _predict_single(self, value):
        rule = self.best_rule_
        if rule["type"] == "categorical":
            return rule["mapping"].get(value, self.default_class_)
        else:
            # For numerical features, if the value is NaN, use the class assigned to NaN values during training
            if pd.isna(value):
                return rule.get("nan_class", self.default_class_)
            
            bps = rule["breakpoints"]
            bcs = rule["bin_classes"]
            for i, bp in enumerate(bps):
                if value <= bp:
                    return bcs[i]
            return bcs[-1]

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        col = X[:, self.best_feature_idx_]
        return np.array([self._predict_single(v) for v in col])
    
    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(preds), 2))
        proba[np.arange(len(preds)), preds] = 1.0
        return proba

def train_1r_baseline():
    data_path = f"{PROCESSED_DATA_DIR}/clean_dataset.csv"
    df = pd.read_csv(data_path)
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    numerical_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    # FastingBs is numerical but has only 2 unique values (0 and 1). In 1R, it can be treated as categorical
    categorical_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope", "FastingBS"]

    preprocessor_1r = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),  
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]), categorical_features),
        ]
    )

    n_num = len(numerical_features)
    n_cat = len(categorical_features)
    numerical_mask = [True] * n_num + [False] * n_cat

    baseline_pipeline = Pipeline([
        ("prep", preprocessor_1r),
        ("clf", OneRClassifier(min_bucket_size=6, numerical_feature_mask=numerical_mask))
    ])

    n_split = 10
    n_repeats = 10
    total_folds = n_split * n_repeats
    # 10-Times Repeated Stratified 10-Fold CV
    outer_cv = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeats, random_state=42)

    print(f"\n{'='*50}\nTraining Model: 1R Baseline\n{'='*50}")
    
    outer_accuracies, outer_f1, outer_kappa = [], [], []
    y_true_all, y_prob_all = [], []
    
    for i, (train_ix, test_ix) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        baseline_pipeline.fit(X_train, y_train)

        y_pred = baseline_pipeline.predict(X_test)
        y_prob = baseline_pipeline.predict_proba(X_test)[:, 1]
        y_true_all.extend(y_test.values)
        y_prob_all.extend(y_prob)
        
        # Metrics for each fold
        fold_acc = accuracy_score(y_test, y_pred)
        outer_accuracies.append(fold_acc)
        outer_f1.append(f1_score(y_test, y_pred))
        outer_kappa.append(cohen_kappa_score(y_test, y_pred))

        best_feat_idx = baseline_pipeline.named_steps["clf"].best_feature_idx_
        best_rule = baseline_pipeline.named_steps["clf"].best_rule_
        feat_names = baseline_pipeline.named_steps["prep"].get_feature_names_out()
        best_feat_name = feat_names[best_feat_idx].split("__")[-1]

        if best_rule["type"] == "numerical":
            n_bins = len(best_rule["breakpoints"]) + 1
            rule_desc = f"{n_bins} bins, breakpoints={[round(b, 2) for b in best_rule['breakpoints']]}"
        else:
            rule_desc = f"{len(best_rule['mapping'])} values"

        print(f"Fold {i+1}/{total_folds} | F1: {outer_f1[-1]:.4f} | Acc: {outer_accuracies[-1]:.4f} | Kappa: {outer_kappa[-1]:.4f} | Rule based on: {best_feat_name} ({rule_desc})")
    
    mean_acc = np.mean(outer_accuracies)
    mean_f1 = np.mean(outer_f1)
    mean_kappa = np.mean(outer_kappa)

    # Calculate Nadeau-Bengio Confidence Interval for Accuracy
    n_total = len(X)
    n_test = n_total / n_split
    n_train = n_total - n_test
    
    ci_acc_lower, ci_acc_upper = get_nadeau_bengio_ci(
        accuracies=outer_accuracies, 
        n_train=n_train, 
        n_test=n_test, 
        confidence=0.99
    )

    print(f"\n--- 1R Baseline Results ---")
    print(f"Dataset Size (N): {n_total}")
    print(f"Accuracy:  {mean_acc:.4f} (99% CI: {ci_acc_lower:.4f} - {ci_acc_upper:.4f})")
    print(f"F1-Score:  {mean_f1:.4f}")
    print(f"Kappa Stat:{mean_kappa:.4f}")

    # Save results to JSON file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = f"{RESULTS_DIR}/1r_baseline.json"
    
    results = {
        "model": "1R Baseline",
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
    
    # Results for each fold
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
    
    # Save full results for PKL dump
    results_1r = {
        "1R_Baseline": {
            'accuracies': outer_accuracies,
            'f1_scores': outer_f1,
            'kappa_scores': outer_kappa,
            'y_true': y_true_all,
            'y_prob': y_prob_all
        }
    }
    
    joblib.dump(results_1r, f"{RESULTS_DIR}/1r_baseline_results.pkl")

if __name__ == "__main__":
    train_1r_baseline()