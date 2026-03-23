# Beyond Accuracy: A Statistically Comparison of Machine Learning Models for Heart Disease Prediction

This project focuses on predicting heart failure using clinical data. Unlike standard predictive modeling pipelines, this repository emphasizes methodological rigor, statistical validation, and clinical interpretability.

We benchmark three multivariate machine learning models (Logistic Regression, KNN, Random Forest, SVM) against a heuristic baseline (1-Rule classifier). The evaluation is performed using 10-Times Repeated 10-Fold Nested Cross-Validation, and the performance differences are rigorously tested using the Corrected Resampled Paired t-test (Nadeau & Bengio, 2003), specifically adopting the formulation for Repeated k-fold Cross-Validation (Bouckaert & Frank 2004), and ROC Convex Hull (ROCCH) analysis.

## Key Findings
*Methodological Validation*: All multivariate models significantly outperformed the 1R Baseline (p<0.05).

*Model Selection*: Logistic Regression, Random Forest, SVM, and KNN showed no statistically significant differences in predictive performance. `Logistic Regression` was therefore selected as the preferred model because its white-box structure enables direct interpretation of feature effects, supporting transparent clinical reasoning and easier communication with medical professionals.

## Project Structure

```
.
├── data/
│   ├── raw/                 # Original dataset (initial_dataset.csv)
│   └── processed/           # Cleaned and engineered data (clean_dataset.csv)
├── results/                 # JSON/PKL files containing model metrics and predictions
├── eda.ipynb                # 1. Exploratory Data Analysis & statistical testing
├── preprocess.py            # 2. Data cleaning & pipeline generation
├── baseline_1R.py           # 3. Custom implementation of the 1-Rule baseline
├── train_models.py          # 4. Nested CV training for LogReg, KNN, and Random Forest
├── confidence_intervals.py  # Utility module for exact CI calculation (imported during training)
├── model_comparison.ipynb   # 5. Statistical testing, ROCCH, and final model selection
├── config.py                # Global directory configurations
├── environment.yml          # Conda environment definition
├── requirements.txt         # Pip dependencies
└── README.md
```

## Environment Setup

### Using Conda (Recommended)

You can use the provided `environment.yml` to set up your environment (e.g., `heart-failure`):

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate heart-failure
```

### Using pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Pipeline Execution Steps
To reproduce the analysis, execute the scripts and notebooks in the following logical order:

### 1. Exploratory Data Analysis (EDA)

Open the Jupyter notebook to explore the raw dataset, analyze clinical distributions, and perform univariate/bivariate statistical testing
```bash
jupyter notebook eda.ipynb
```

### 2. Data Preprocessing

Run the preprocessing script to clean the data (handling missing/impossible physiological values) and generate the finalized dataset for modeling.

```bash
python preprocess.py
```

### 3. Baseline Model Validation (1R)

Establish a rigorous minimum performance threshold by training the univariate 1-Rule baseline classifier. Results are saved to results/1r_baseline.json.

```bash
python baseline_1R.py
```

### 4. Multivariate Model Training (Nested CV)

Execute the core training pipeline. This script utilizes 10-Times Repeated 10-Fold Nested Cross-Validation for unbiased hyperparameter tuning and evaluation. (Note: This script automatically imports confidence_intervals.py to compute 99% CIs).

```bash
python train_models.py
```

### 5. Statistical Comparison & Model Selection

Finally, open the comparison notebook to evaluate the models geometrically (ROC Convex Hull) and statistically (Corrected Resampled t-test).

```bash
jupyter notebook model_comparison.ipynb
```
