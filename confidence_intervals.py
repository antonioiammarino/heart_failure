import numpy as np
from scipy import stats

def get_confidence_interval(successes, n, confidence=0.99):
    f = successes / n
    # z per 99% di confidenza -> (1-0.99)/2 = 0.005 -> z = 2.576
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    center = f + z**2 / (2 * n)
    margin = z * np.sqrt(f/n - f**2/n + z**2/(4*n**2))
    denominator = 1 + z**2 / n
    
    lower = (center - margin) / denominator
    upper = (center + margin) / denominator
    return lower, upper