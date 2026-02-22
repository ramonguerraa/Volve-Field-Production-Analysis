import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

def hyperbolic_decline(t, qi, di, b):
    return qi / (1 + b * di * t)**(1/b)

def calculate_eur(qi, di, b, q_limit=50):
    # Integral of the curve to obtain cumulative volume
    eur = (qi**b / (di * (1 - b))) * (qi**(1 - b) - q_limit**(1 - b))
    return eur * 30.4 # Approximate monthly bbl 