import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from arch import arch_model # Asegúrate de tener 'arch' en tu requirements.txt

# --- 1. MODELO DE LEY DE POTENCIA (Power Law Loss-of-Ratio) ---
def power_law_decline(t, qi, n):
    """
    Modelo útil para declinaciones que no son constantes en el tiempo.
    q = qi * (t + 1)^-n
    """
    return qi * (t + 1)**(-n)

# --- 2. MODELO SEPD (Stretched Exponential Production Decline) ---
def sepd_decline(t, qi, tau, n):
    """
    Modelo avanzado para flujos transigentes (como el inicio del F-11).
    q = qi * exp(-(t/tau)^n)
    """
    return qi * np.exp(-(t / tau)**n)

# --- 3. MODELO GARCH (Análisis de Volatilidad) ---
def fit_garch_model(series):
    """
    Ajusta un modelo GARCH(1,1) para entender la varianza del error.
    Retorna la volatilidad condicional.
    """
    # Escalamos la serie para facilitar la convergencia numérica
    scaling_factor = 100 / series.std()
    scaled_series = series * scaling_factor
    
    try:
        model = arch_model(scaled_series, vol='Garch', p=1, q=1, dist='normal')
        res = model.fit(disp='off')
        # Devolvemos la volatilidad des-escalada
        return res.conditional_volatility / scaling_factor
    except:
        return None

# --- 4. FUNCIÓN MAESTRA DE AJUSTE AVANZADO ---
def fit_advanced_model(model_type, x, y):
    """
    Wrapper para ajustar SEPD o Power Law y devolver métricas.
    """
    try:
        if model_type == 'sepd':
            # p0: qi, tau, n (n suele estar entre 0.1 y 1.0)
            popt, _ = curve_fit(sepd_decline, x, y, p0=[y[0], 100, 0.5], 
                                bounds=(0, [y[0]*1.5, 10000, 1]))
            y_pred = sepd_decline(x, *popt)
        
        elif model_type == 'power_law':
            # p0: qi, n
            popt, _ = curve_fit(power_law_decline, x, y, p0=[y[0], 0.5], 
                                bounds=(0, [y[0]*1.5, 2]))
            y_pred = power_law_decline(x, *popt)
            
        r2 = r2_score(y, y_pred)
        return popt, r2, y_pred
    except Exception as e:
        return None, 0, None