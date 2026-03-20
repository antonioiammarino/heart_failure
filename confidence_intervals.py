import numpy as np
from scipy import stats

def get_nadeau_bengio_ci(accuracies, n_train, n_test, confidence=0.99):
    n_folds = len(accuracies)
    mean_acc = np.mean(accuracies)
    
    # Classic variance of accuracies (ddof=1 for sample variance)
    var_acc = np.var(accuracies, ddof=1)
    
    # Correction factor for Nadeau-Bengio
    correction_factor = (1 / n_folds) + (n_test / n_train)
    
    # Corrected Standard Error
    se_corrected = np.sqrt(correction_factor * var_acc)
    
    # Calculation of t-statistic for the requested confidence
    alpha = 1 - confidence
    t_stat = stats.t.ppf(1 - alpha / 2, df=n_folds - 1)
    
    # Margin of error and interval
    margin = t_stat * se_corrected
    lower_bound = mean_acc - margin
    upper_bound = mean_acc + margin
    
    return lower_bound, upper_bound