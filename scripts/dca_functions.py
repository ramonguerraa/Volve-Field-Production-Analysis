import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

def hyperbolic_decline(t, qi, di, b):
    return qi / (1 + b * di * t)**(1/b)

def calculate_eur(qi, di, b, q_limit=50):
    # Integral of the curve to obtain cumulative volume
    eur = (qi**b / (di * (1 - b))) * (qi**(1 - b) - q_limit**(1 - b))
    return eur * 30.4 # Approximate monthly bbl 

def fit_arps_model(x, y):
    if len(x) < 5: # Si hay menos de 5 puntos, no vale la pena modelar
        return None, 0, None
    try:
        # Aumentamos maxfev para dar más intentos al algoritmo
        popt, _ = curve_fit(hyperbolic_decline, x, y, 
                            p0=[y[0], 0.001, 0.5], 
                            bounds=(0, [y[0]*1.5, 1, 1]),
                            maxfev=5000) 
        y_pred = hyperbolic_decline(x, *popt)
        r2 = r2_score(y, y_pred)
        return popt, r2, y_pred
    except:
        return None, 0, None

def detect_multiple_breaks(df_well, window=30, log_diff_threshold=0.15, min_dist=180):
    """
    Detecta múltiples puntos de quiebre significativos usando cambios en la 
    pendiente logarítmica y persistencia temporal.
    """
    df_well = df_well.sort_values('dateprd').copy()
    
    # 1. Suavizado para evitar falsos positivos por ruido diario
    df_well['q_smooth'] = df_well['bore_oil_vol'].rolling(window=window, center=True).mean()
    
    # 2. Transformada logarítmica y diferencia (gradiente)
    # Agregamos 1 para evitar log(0)
    df_well['ln_q'] = np.log(df_well['q_smooth'] + 1)
    df_well['log_diff'] = df_well['ln_q'].diff() 
    
    # 3. Identificar cambios bruscos (Rupturas)
    # Buscamos donde el cambio logarítmico supere el umbral
    breaks = df_well[np.abs(df_well['log_diff']) > log_diff_threshold].copy()
    
    if breaks.empty:
        return []

    # 4. Refinamiento: Evitar clusters (agrupamiento de fechas cercanas)
    # Solo tomamos quiebres que estén separados por al menos 'min_dist' días
    refined_breaks = []
    last_break = None
    
    for current_date in breaks['dateprd']:
        if last_break is None or (current_date - last_break).days >= min_dist:
            refined_breaks.append(current_date)
            last_break = current_date
            
    return refined_breaks