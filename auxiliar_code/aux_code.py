
from auxiliar_code.config_set import *
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Any

# ============================================================
# Unit-root / stationarity test functions
# ============================================================

def run_adf(x, trend='c', autolag='AIC'):
    """
    Run the Augmented Dickey–Fuller (ADF) unit root test.

    Parameters
    ----------
    x : array-like
        One-dimensional time series (levels or returns).
    trend : {'c', 'ct', 'nc'}, optional
        Deterministic component in the ADF regression:
        'c'  : constant (default),
        'ct' : constant + linear trend,
        'nc' : no deterministic terms.
    autolag : {'AIC', 'BIC', 't-stat', None}, optional
        Criterion used to select the number of lags automatically.

    Null hypothesis (H0)
    --------------------
    The series has a unit root (is non-stationary).

    Returns
    -------
    dict
        If the test runs successfully:
        - 'stat'     : test statistic.
        - 'pvalue'   : p-value.
        - 'lags'     : number of lags used.
        - 'nobs'     : number of observations used.
        - 'crit_1%'  : 1% critical value.
        - 'crit_5%'  : 5% critical value.
        - 'crit_10%' : 10% critical value.
        If an error occurs:
        - 'stat'   : np.nan
        - 'pvalue' : np.nan
        - 'error'  : error message as string.
    """
    # Try to run the ADF test
    try:
        # Call adfuller → (stat, pvalue, usedlag, nobs, crit_values, icbest)
        res = adfuller(x, regression=trend, autolag=autolag)
        # Unpack ADF output tuple
        stat, pval, usedlag, nobs, crit, icbest = res
        
        # Return test statistic, p-value, lags, sample size and critical values
        return {
            'stat': stat,
            'pvalue': pval,
            'lags': usedlag,
            'nobs': nobs,
            'crit_1%': crit.get('1%'),
            'crit_5%': crit.get('5%'),
            'crit_10%': crit.get('10%')
        }
    # On any error, return NaNs plus the error message
    except Exception as e:
        return {'stat': np.nan, 'pvalue': np.nan, 'error': str(e)}

def run_kpss(x, regression='c', nlags='auto'):
    """
    Run the KPSS stationarity test.

    Parameters
    ----------
    x : array-like
        One-dimensional time series.
    regression : {'c', 'ct'}, optional
        Type of deterministic component:
        'c'  : level stationarity (default),
        'ct' : trend stationarity.
    nlags : {'auto', int}, optional
        Number of lags in the Newey–West estimator.
        'auto' lets the function choose lags automatically.

    Null hypothesis (H0)
    --------------------
    The series is stationary (in level or trend, depending on `regression`).

    Returns
    -------
    dict
        If the test runs successfully:
        - 'stat'     : test statistic.
        - 'pvalue'   : p-value.
        - 'lags'     : number of lags used.
        - 'crit_1%'  : 1% critical value.
        - 'crit_5%'  : 5% critical value.
        - 'crit_10%' : 10% critical value.
        If an error occurs:
        - 'stat'   : np.nan
        - 'pvalue' : np.nan
        - 'error'  : error message as string.
    """
    # Try to run the KPSS test
    try:
        # Temporarily suppress warnings (e.g. p-value interpolation warnings)
        with warnings.catch_warnings():
            # Ignore warnings inside this context
            warnings.simplefilter("ignore")
            # Call kpss and get statistic, p-value, number of lags and critical values
            stat, pval, lags, crit = kpss(x, regression=regression, nlags=nlags)
            
        # Return test statistic, p-value, lags and critical values
        return {
            'stat': stat,
            'pvalue': pval,
            'lags': lags,
            'crit_1%': crit.get('1%'),
            'crit_5%': crit.get('5%'),
            'crit_10%': crit.get('10%')
        }
    # On any error, return NaNs plus the error message
    except Exception as e:
        return {'stat': np.nan, 'pvalue': np.nan, 'error': str(e)}

def run_pp(x, trend='c'):
    """
    Run the Phillips–Perron (PP) unit root test using the `arch` library.

    Parameters
    ----------
    x : array-like
        One-dimensional time series.
    trend : {'c', 'ct', 'nc'}, optional
        Deterministic component:
        'c'  : constant (default),
        'ct' : constant + trend,
        'nc' : no deterministic terms.

    Null hypothesis (H0)
    --------------------
    The series has a unit root (is non-stationary).

    Returns
    -------
    dict
        Dictionary with:
        - 'stat'     : test statistic.
        - 'pvalue'   : p-value.
        - 'lags'     : effective number of lags.
        - 'nobs'     : number of observations.
        - 'crit_1%'  : 1% critical value.
        - 'crit_5%'  : 5% critical value.
        - 'crit_10%' : 10% critical value.
        - 'backend'  : string 'arch'.

    Notes
    -----
    This implementation assumes that `PhillipsPerron` from `arch.unitroot`
    is available and does not catch errors. If you want it to fail gracefully
    (e.g. when `arch` is missing), you can wrap this body in a try/except
    similar to `run_adf` and `run_kpss`.
    """
    # Instantiate the PhillipsPerron test from the arch.unitroot module
    pp = PhillipsPerron(x, trend=trend)
    
    # Return test output, converting to native Python types when needed
    # Note: arch selects the Newey–West lag length automatically if not specified
    return {
        'stat': float(pp.stat),
        'pvalue': float(pp.pvalue),
        'lags': int(pp.lags),
        'nobs': int(pp.nobs),
        'crit_1%': pp.critical_values.get('1%'),
        'crit_5%': pp.critical_values.get('5%'),
        'crit_10%': pp.critical_values.get('10%'),
        'backend': 'arch'
    }

def stationarity_battery(df, trend_adf='c', regression_kpss='c',
                         trend_pp='c', alpha=0.05):
    """
    Run ADF, KPSS and PP tests for each column in a DataFrame
    and build a majority-based stationarity decision.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose columns are time series.
    trend_adf : {'c', 'ct', 'nc'}, optional
        Deterministic component in the ADF regression.
    regression_kpss : {'c', 'ct'}, optional
        Deterministic component in the KPSS regression (level or trend).
    trend_pp : {'c', 'ct', 'nc'}, optional
        Deterministic component in the PP regression.
    alpha : float, optional
        Significance level for each individual test (default 0.05).

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by series name with columns:
        - ADF_stat, ADF_p, ADF_decisión
        - KPSS_stat, KPSS_p, KPSS_decisión
        - PP_stat, PP_p, PP_decisión
        - Mayoría : global stationarity decision with vote count.
    """
    # Initialize list to collect per-series result dictionaries
    rows = []
    
    # Loop over each column (series) in the input DataFrame
    for col in df.columns:
        # Convert series to numeric, coerce errors to NaN, and drop missing values
        series_clean = pd.to_numeric(df[col], errors='coerce').dropna()
        # Extract numeric values as a 1D NumPy array
        x = series_clean.values
        
        # Skip this series if there are too few observations
        if len(x) < 20:
            continue

        # --- RUN INDIVIDUAL TESTS ---

        # Run ADF test on the series
        adf = run_adf(x, trend=trend_adf)
        # Run KPSS test on the series
        kps = run_kpss(x, regression=regression_kpss)
        # Run PP test on the series
        pp = run_pp(x, trend=trend_pp)

        # --- INDIVIDUAL DECISIONS ---

        # 1. ADF (H0: unit root / non-stationary)
        # If there is no p-value, mark as error and exclude from voting
        if pd.isna(adf.get('pvalue')):
            # ADF decision label when the test failed
            adf_dec = 'Error'
            # ADF vote is None to indicate "no vote"
            adf_vote = None
        else:
            # True if p < alpha (reject H0 → stationary)
            is_stat_adf = adf['pvalue'] < alpha
            # Human-readable decision for ADF
            adf_dec = 'Stationary' if is_stat_adf else 'Non-stationary'
            # Boolean vote used in the majority rule
            adf_vote = is_stat_adf

        # 2. KPSS (H0: stationary)
        # If there is no p-value, mark as error and exclude from voting
        if pd.isna(kps.get('pvalue')):
            # KPSS decision label when the test failed
            kps_dec = 'Error'
            # KPSS vote is None to indicate "no vote"
            kps_vote = None
        else:
            # True if p >= alpha (do not reject H0 → stationary)
            is_stat_kps = kps['pvalue'] >= alpha
            # Human-readable decision for KPSS
            kps_dec = 'Stationary' if is_stat_kps else 'Non-stationary'
            # Boolean vote used in the majority rule
            kps_vote = is_stat_kps

        # 3. PP (H0: unit root / non-stationary)
        # If there is no p-value, mark as not computed and exclude from voting
        if pd.isna(pp.get('pvalue')):
            # PP decision label when the test failed
            pp_dec = 'Not computed'
            # PP vote is None to indicate "no vote"
            pp_vote = None
        else:
            # True if p < alpha (reject H0 → stationary)
            is_stat_pp = pp['pvalue'] < alpha
            # Human-readable decision for PP
            pp_dec = 'Stationary' if is_stat_pp else 'Non-stationary'
            # Boolean vote used in the majority rule
            pp_vote = is_stat_pp

        # --- MAJORITY DECISION ---

        # Collect all non-None votes from ADF, KPSS and PP
        votes = [v for v in [adf_vote, kps_vote, pp_vote] if v is not None]
        # Count how many tests provided a valid vote
        n_valid = len(votes)
        
        # If we have at least one valid vote
        if n_valid > 0:
            # Number of "True" votes (stationary)
            n_yes = sum(votes)  # True counts as 1
            # Require a strict majority (> 50% of valid votes)
            is_majority_stat = n_yes > (n_valid / 2)
            # Build a string summarizing the majority decision and vote count
            majority_str = f"{'Stationary' if is_majority_stat else 'Non-stationary'} ({n_yes}/{n_valid})"
        # If no test produced a valid vote, label as insufficient information
        else:
            majority_str = "Insufficient test information"

        # Append all test statistics and decisions for this series
        rows.append({
            'serie': col,
            'ADF_stat': adf.get('stat'),
            'ADF_p': adf.get('pvalue'),
            'ADF_decisión': adf_dec,
            'KPSS_stat': kps.get('stat'),
            'KPSS_p': kps.get('pvalue'),
            'KPSS_decisión': kps_dec,
            'PP_stat': pp.get('stat'),
            'PP_p': pp.get('pvalue'),
            'PP_decisión': pp_dec,
            'Mayoría': majority_str
        })

    # Build the final DataFrame from the list of row dictionaries
    res = pd.DataFrame(rows)
    # If the result is not empty, set the series name as index
    if not res.empty:
        res = res.set_index('serie')
        
    # Return the DataFrame with the stationarity battery results
    return res

# ====================================================================
# 2. Función Robusta para Evaluar SDP
# ====================================================================

def evaluar_sdp(matriz, tol_simetria=1e-8, tol_eigen=1e-10):
    """
    Evalúa si una matriz es Simétrica Definida Positiva.
    - tol_simetria: Tolerancia para diferencias entre A y A^T
    - tol_eigen: Tolerancia para considerar un eigenvalor como > 0
    """
    # 1. Comprobar Simetría
    es_simetrica = np.allclose(matriz, matriz.T, atol=tol_simetria)
    
    # Si no es simétrica por un margen amplio, falla directamente
    if not es_simetrica:
        return False, "Falla Simetría"
        
    # 2. Comprobar Eigenvalores (Definida Positiva)
    # Forzamos simetría exacta para la función eigh por estabilidad numérica
    matriz_sim = (matriz + matriz.T) / 2.0 
    
    try:
        # eigh es mucho más estable y rápido para matrices simétricas
        eigenvalores = np.linalg.eigvalsh(matriz_sim)
        # Verificamos que el eigenvalor más pequeño sea estrictamente > 0 (considerando la tolerancia)
        es_pos_def = np.min(eigenvalores) > -tol_eigen
        
        if es_pos_def:
            return True, "SDP"
        else:
            return False, f"Falla Eigen (Min Eigen: {np.min(eigenvalores):.2e})"
    except np.linalg.LinAlgError:
        return False, "Error Álgebra Lineal"

# -------------------------- Vectorized erf --------------------------
try:
    _erf = np.erf
except AttributeError:
    from math import erf as _erf_scalar
    _erf = np.vectorize(_erf_scalar, otypes=[float])

# -------------------------- Basic Distribution Helpers --------------------------
def _norm_cdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))

def _norm_cdf_scalar(z: float) -> float:
    return float(_norm_cdf(np.array([z]))[0])

def _norm_sf(z: float) -> float:
    return float(1.0 - _norm_cdf_scalar(z))

def _chi2_sf_wilson_hilferty(stat: float, df: int) -> float:
    if df <= 0: return 1.0
    z = (stat / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))
    z /= sqrt(2.0 / (9.0 * df))
    return float(np.clip(_norm_sf(z), 0.0, 1.0))

# -------------------------- Momentos --------------------------
def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    m2 = np.mean(x ** 2)
    m3 = np.mean(x ** 3)
    if m2 <= EPS: return 0.0
    return float(m3 / (m2 ** 1.5))

def excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    m2 = np.mean(x ** 2)
    m4 = np.mean(x ** 4)
    if m2 <= EPS: return 0.0
    return float(m4 / (m2 ** 2) - 3.0)

# -------------------------- Pruebas Univariadas --------------------------
def jarque_bera_test(x: np.ndarray) -> Tuple[float, int, float]:
    x = np.asarray(x, float)
    T = x.size
    S = skewness(x)
    Kex = excess_kurtosis(x)
    JB = (T / 6.0) * (S ** 2 + 0.25 * Kex ** 2)
    p = _chi2_sf_wilson_hilferty(JB, df=2)
    return float(JB), 2, float(p)

def anderson_darling_normal(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, float)
    T = x.size
    if T < 8: return 0.0, 1.0

    mu = x.mean()
    sd = max(x.std(ddof=1), EPS)
    z = (x - mu) / sd
    z_sorted = np.sort(z)

    Fi = np.clip(_norm_cdf(z_sorted), EPS, 1.0 - EPS)
    i = np.arange(1, T + 1)
    A2 = -T - np.mean((2 * i - 1) * (np.log(Fi) + np.log(1.0 - Fi[::-1])))
    A2_star = A2 * (1.0 + 4.0 / T - 25.0 / (T ** 2))

    if A2_star < 0.2: p = 1.0 - np.exp(-13.436 + 101.14 * A2_star - 223.73 * A2_star ** 2)
    elif A2_star < 0.34: p = 1.0 - np.exp(-8.318 + 42.796 * A2_star - 59.938 * A2_star ** 2)
    elif A2_star < 0.6: p = np.exp(0.9177 - 4.279 * A2_star - 1.38 * A2_star ** 2)
    else: p = np.exp(1.2937 - 5.709 * A2_star + 0.0186 * A2_star ** 2)
    return float(A2_star), float(np.clip(p, 0.0, 1.0))

def ks_normal_test(x: np.ndarray, K_terms: int = 100) -> Tuple[float, float]:
    x = np.asarray(x, float)
    T = x.size
    if T < 5: return 0.0, 1.0

    mu = x.mean()
    sd = max(x.std(ddof=1), EPS)
    z_sorted = np.sort((x - mu) / sd)

    Fi = np.clip(_norm_cdf(z_sorted), EPS, 1.0 - EPS)
    i = np.arange(1, T + 1)
    D = float(max(np.max(i/T - Fi), np.max(Fi - (i-1)/T)))

    lam = (sqrt(T) + 0.12 + 0.11 / sqrt(T)) * D
    k = np.arange(1, K_terms + 1)
    p = 2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * (k ** 2) * (lam ** 2)))
    return D, float(np.clip(p, 0.0, 1.0))

def cramer_von_mises_normal(x: np.ndarray, B: int = 500, seed: int = 1234) -> Tuple[float, float]:
    x = np.asarray(x, float)
    T = x.size
    if T < 5: return 0.0, 1.0

    mu = x.mean()
    sd = max(x.std(ddof=1), EPS)
    z_sorted = np.sort((x - mu) / sd)

    Fi = np.clip(_norm_cdf(z_sorted), EPS, 1.0 - EPS)
    ui = (2.0 * np.arange(1, T + 1) - 1.0) / (2.0 * T)
    W2 = float(1.0 / (12.0 * T) + np.sum((Fi - ui) ** 2))

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(B, T))
    Zs = (Z - Z.mean(axis=1, keepdims=True)) / np.maximum(Z.std(axis=1, ddof=1, keepdims=True), EPS)
    Fi_b = np.clip(_norm_cdf(np.sort(Zs, axis=1)), EPS, 1.0 - EPS)
    
    W2_b = 1.0 / (12.0 * T) + np.sum((Fi_b - ui.reshape(1, -1)) ** 2, axis=1)
    return W2, float(np.mean(W2_b >= W2))

def erfinv(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    a = 0.147
    sgn = np.sign(y)
    ln = np.log(1.0 - y ** 2)
    t = 2.0 / (np.pi * a) + ln / 2.0
    x0 = sgn * np.sqrt(np.sqrt(t ** 2 - ln / a) - t)
    return x0 - (_erf(x0) - y) / (2.0 / np.sqrt(np.pi) * np.exp(-x0 ** 2))

# -------------------------- Prueba Mardia (Multivariada) --------------------------
def mardia_tests(W: np.ndarray) -> Dict[str, float]:
    W = np.asarray(W, float)
    T, p = W.shape
    if T < 5: return {"skew_stat": 0.0, "skew_p": 1.0, "kurt_z": 0.0, "kurt_p": 1.0}

    Wc = W - W.mean(axis=0, keepdims=True)
    A = Wc @ np.linalg.pinv((Wc.T @ Wc) / T) @ Wc.T

    skew_stat = T * np.mean(A ** 3) / 6.0
    skew_df = int(p * (p + 1) * (p + 2) / 6)
    
    mean_b2p = p * (p + 2.0)
    var_b2p = (8.0 * p * (p + 2.0)) / T
    kurt_z = (np.mean(np.diag(A) ** 2) - mean_b2p) / sqrt(max(var_b2p, EPS))
    Phi = _norm_cdf_scalar(kurt_z)

    return {
        "skew_stat": float(skew_stat), "skew_df": skew_df,
        "skew_p": _chi2_sf_wilson_hilferty(skew_stat, skew_df),
        "kurt_z": float(kurt_z), "kurt_p": 2.0 * min(Phi, 1.0 - Phi)
    }


# -------------------------- Gráficas en Cuadrícula --------------------------
def plot_normality_grid(W_df: pd.DataFrame, ruta_img=None):
    """
    Dibuja los diagnósticos dividiéndolos en lotes (ej. 5 variables por figura)
    con tipografía EXTRA GRANDE, proporciones anchas y guardado en PDF.
    """
    # Configuramos la tipografía extrema para que resalte
    plt.rcParams.update({
        'font.size': 22, 
        'axes.titlesize': 26, 
        'axes.labelsize': 22,
        'xtick.labelsize': 20, 
        'ytick.labelsize': 20, 
        'legend.fontsize': 22
    })
    
    n_total = W_df.shape[1]
    vars_por_figura = 5  # Lotes de 5 variables
    num_figuras = int(np.ceil(n_total / vars_por_figura))
    
    for fig_idx in range(num_figuras):
        inicio = fig_idx * vars_por_figura
        fin = min(inicio + vars_por_figura, n_total)
        df_lote = W_df.iloc[:, inicio:fin]
        n_lote = df_lote.shape[1]
        
        # Figura ancha y con altura proporcional al lote (4.0 de alto por renglón)
        fig, axes = plt.subplots(n_lote, 4, figsize=(32, 4.0 * n_lote), constrained_layout=True)
        if n_lote == 1: axes = np.array([axes])

        for i, col in enumerate(df_lote.columns):
            x = df_lote[col].dropna().values
            T = x.size
            mu, sd = x.mean(), max(x.std(ddof=1), EPS)
            z = (x - mu) / sd
            z_sorted = np.sort(z)

            # 1. Histograma + Normal
            axes[i, 0].hist(z, bins=30, density=True, alpha=0.6, edgecolor='white', color='#1f77b4')
            grid = np.linspace(z.min(), z.max(), 400)
            axes[i, 0].plot(grid, (1.0 / sqrt(2.0 * np.pi)) * np.exp(-0.5 * grid**2), color='red', lw=3.0)
            axes[i, 0].set_title(f"{col}\nHistograma vs Normal", fontweight='bold')

            # 2. Q-Q Plot
            probs = (np.arange(1, T + 1) - 0.5) / T
            q_theory = np.sqrt(2.0) * erfinv(2.0 * probs - 1.0)
            axes[i, 1].scatter(q_theory, z_sorted, alpha=0.5, color='#1f77b4', s=60)
            lo, hi = min(q_theory.min(), z_sorted.min()), max(q_theory.max(), z_sorted.max())
            axes[i, 1].plot([lo, hi], [lo, hi], 'r--', lw=3.0)
            axes[i, 1].set_title(f"{col}\nQ-Q Plot", fontweight='bold')

            # 3. P-P Plot
            Fi = np.clip(_norm_cdf(z_sorted), EPS, 1.0 - EPS)
            axes[i, 2].scatter(Fi, probs, alpha=0.5, color='#1f77b4', s=60)
            axes[i, 2].plot([0, 1], [0, 1], 'r--', lw=3.0)
            axes[i, 2].set_title(f"{col}\nP-P Plot", fontweight='bold')

            # 4. CDF Empírica vs Normal
            axes[i, 3].plot(z_sorted, probs, label="Empírica", lw=3.0, color='#1f77b4')
            axes[i, 3].plot(z_sorted, Fi, label="Teórica", lw=3.0, color='red', linestyle='--')
            axes[i, 3].set_title(f"{col}\nCDF Empírica vs Normal", fontweight='bold')
            axes[i, 3].legend(loc='best')

        # --- GUARDAR EN PDF POR CADA PARTE ---
        if ruta_img is not None:
            os.makedirs(ruta_img, exist_ok=True)
            archivo_salida = os.path.join(ruta_img, f"diagnosticos_normalidad_tvp_var_parte{fig_idx + 1}.pdf")
            plt.savefig(archivo_salida, format='pdf', bbox_inches='tight')
            print(f"\n[+] Gráfica de Normalidad (Parte {fig_idx + 1}) exportada en PDF: {archivo_salida}")

        plt.show()

# -------------------------- Reporte y API Principal --------------------------
def normality_report_W(W: np.ndarray, names: List[str], cvm_B: int = 500) -> pd.DataFrame:
    rows = []
    for j in range(W.shape[1]):
        x = W[:, j]
        jb_stat, _, jb_p = jarque_bera_test(x)
        ad_stat, ad_p = anderson_darling_normal(x)
        ks_stat, ks_p = ks_normal_test(x)
        cvm_stat, cvm_p = cramer_von_mises_normal(x, B=cvm_B)

        rows.append({
            "Variable": names[j],
            "Asimetría": round(skewness(x), 4),
            "Curtosis Ex.": round(excess_kurtosis(x), 4),
            "JB (p-val)": round(jb_p, 4),
            "AD (p-val)": round(ad_p, 4),
            "KS (p-val)": round(ks_p, 4),
            "CvM (p-val)": round(cvm_p, 4)
        })
    return pd.DataFrame(rows).set_index("Variable")

def validate_normality_of_whitened_innovations_tvp(Y, X, fit, var_names, make_plots=True, cvm_B=1000, ruta_img=None):
    T, n = Y.shape
    k = X.shape[1]

    # Calcular residuales (Y - Y_hat)
    B_all = fit["beta_filt"].reshape(T, n, k)
    y_hat = np.einsum("tik,tk->ti", B_all, X, optimize=True)
    
    # Blanquear las innovaciones
    W = whiten_by_Sigma(Y - y_hat, fit["S_t"], method="zca", center=True)
    df_W = pd.DataFrame(W, columns=var_names)

    # Generar tabla univariada
    df_report = normality_report_W(W, var_names, cvm_B=cvm_B)
    
    # Pruebas Multivariadas de Mardia
    mardia = mardia_tests(W)

    print("\n" + "="*85)
    print("DIAGNÓSTICO COMPLETO DE NORMALIDAD EN INNOVACIONES BLANQUEADAS (W)".center(85))
    print("="*85)

    print("\n[PRUEBAS UNIVARIADAS DE NORMALIDAD] Valores-p (H0: Distribución Normal)")
    print("-" * 85)
    print(df_report)
    print("-" * 85)

    print("\n[PRUEBA MULTIVARIADA DE MARDIA]")
    print(f"Asimetría Multivariada: χ²({mardia['skew_df']}) = {mardia['skew_stat']:.2f} | Valor-p = {mardia['skew_p']:.4f}")
    print(f"Curtosis Multivariada:  Z = {mardia['kurt_z']:.2f} | Valor-p = {mardia['kurt_p']:.4f}")
    print("\nNota: Valores-p menores a 0.05 indican rechazo de la normalidad.")

    if make_plots:
        print("\nGenerando gráficas de diagnóstico de normalidad en cuadrícula...")
        plot_normality_grid(df_W, ruta_img)

    return df_W, df_report, mardia


# -------------------------- 1. HELPERS & MATH --------------------------
def plot_ccf_pairs(resid, names, max_lag=24, pairs=None, ruta_img=None, title="Cross-Correlations"):
    """
    Grafica correlaciones cruzadas en una cuadrícula de 4x5 (20 por panel)
    y guarda CADA PANEL en un archivo PDF INDEPENDIENTE.
    """
    n = resid.shape[1]
    T = resid.shape[0]
    
    # Generar todos los pares si no se especifican
    if pairs is None:
        pairs = list(itertools.combinations(range(n), 2))
    
    # Configuración de 4 filas x 5 columnas
    rows = 4
    cols = 5
    plots_per_page = rows * cols
    
    total_pairs = len(pairs)
    total_pages = math.ceil(total_pairs / plots_per_page)
    
    print(f"\nTotal de pares a graficar: {total_pairs}")
    print(f"Generando {total_pages} paneles (cuadrícula de {rows}x{cols})...")
    
    conf = 1.96 / np.sqrt(T)
    
    # Configuramos la tipografía base más grande
    plt.rcParams.update({
        'font.size': 20, 
        'axes.titlesize': 24, 
        'axes.labelsize': 20,
        'xtick.labelsize': 16, 
        'ytick.labelsize': 16
    })
    
    # Bucle por paneles (páginas)
    for page in range(total_pages):
        start_idx = page * plots_per_page
        end_idx = start_idx + plots_per_page
        
        current_batch = pairs[start_idx:end_idx]
        n_batch = len(current_batch)
        
        # Figura amplia para 4x5 (Aumenté un poco el alto para dar espacio a los textos grandes)
        fig, axes = plt.subplots(rows, cols, figsize=(26, 16), constrained_layout=True)
        fig.suptitle(f"", fontsize=28, fontweight='bold')
        
        axes_flat = np.array(axes).flat
        lags = np.arange(-max_lag, max_lag + 1)
        
        for i, (idx_i, idx_j) in enumerate(current_batch):
            ax = axes_flat[i]
            l, ccf = ccf_1d(resid[:, idx_i], resid[:, idx_j], max_lag)
            
            # Stem plot (líneas y puntos un poco más gruesos para balancear la letra grande)
            markerline, stemlines, baseline = ax.stem(l, ccf, basefmt=" ")
            plt.setp(markerline, markersize=5, color='black')
            plt.setp(stemlines, linewidth=1.8, color='#1f77b4')
            plt.setp(baseline, linewidth=1.2, color='black')
            
            # Bandas de confianza
            ax.axhline(conf, c='red', ls='--', lw=1.8, alpha=0.7)
            ax.axhline(-conf, c='red', ls='--', lw=1.8, alpha=0.7)
            ax.fill_between(l, -conf, conf, color='red', alpha=0.08)
            
            # Estética (Fuentes gigantes)
            name_i = names[idx_i][:12] 
            name_j = names[idx_j][:12]
            ax.set_title(f"{name_i} vs {name_j}", fontsize=18, fontweight='bold')
            ax.tick_params(axis='both', labelsize=14)
            
            # Etiquetas X solo en la parte inferior o en los últimos activos
            if i >= (plots_per_page - cols) or i >= (n_batch - cols):
                ax.set_xlabel("Lag", fontsize=16, fontweight='bold')
        
        # Apagar ejes vacíos
        for k in range(n_batch, len(axes_flat)):
            axes_flat[k].axis('off')
            
        # --- GUARDAR EL PANEL ACTUAL COMO PDF INDEPENDIENTE ---
        if ruta_img is not None:
            os.makedirs(ruta_img, exist_ok=True)
            archivo_salida = os.path.join(ruta_img, f"cross_correlations_panel_{page + 1}.pdf")
            fig.savefig(archivo_salida, format='pdf', bbox_inches='tight')
            print(f"[+] Exportado: {archivo_salida}")
            
        plt.show()

EPS = 1e-12

def _chi2_sf_wilson_hilferty(stat: float, df: int) -> float:
    if df <= 0: return 1.0
    if stat < 0: return 1.0
    z = (stat / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))
    z /= sqrt(2.0 / (9.0 * df))
    return 1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))

def acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom < EPS: return np.zeros(max_lag)
    acf = np.empty(max_lag)
    for k in range(1, max_lag + 1):
        acf[k - 1] = np.dot(x[:-k], x[k:]) / denom
    return acf

def ccf_1d(x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    x = x - x.mean()
    y = y - y.mean()
    denom = sqrt(np.dot(x, x) * np.dot(y, y))
    denom = max(denom, EPS)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.empty(lags.size)
    for i, k in enumerate(lags):
        if k < 0:   out[i] = np.dot(x[-k:], y[:x.size + k]) / denom
        elif k > 0: out[i] = np.dot(x[:x.size - k], y[k:]) / denom
        else:       out[i] = np.dot(x, y) / denom
    return lags, out

def _vec_outer_products(W: np.ndarray, use_vech: bool = False) -> np.ndarray:
    T, n = W.shape
    if use_vech:
        idx = np.tril_indices(n)
        d = len(idx[0])
        S_stack = np.empty((T, d), float)
        for t in range(T):
            S_stack[t] = np.outer(W[t], W[t])[idx]
    else:
        d = n * n
        S_stack = np.empty((T, d), float)
        for t in range(T):
            S_stack[t] = np.outer(W[t], W[t]).reshape(-1)
    return S_stack

# -------------------------- 2. STATISTICAL TESTS --------------------------

def ljung_box(x: np.ndarray, h: int) -> Tuple[float, int, float]:
    T = x.size
    rho = acf_1d(x, h)
    Q = T * (T + 2.0) * np.sum((rho ** 2) / (T - np.arange(1, h + 1)))
    p = _chi2_sf_wilson_hilferty(Q, h)
    return float(Q), int(h), float(p)

def multivariate_portmanteau(W: np.ndarray, m: int) -> Tuple[float, int, float]:
    T, n = W.shape
    Wc = W - W.mean(axis=0)
    Gamma0_inv = np.linalg.pinv((Wc.T @ Wc) / T)
    Qm = 0.0
    for k in range(1, m + 1):
        Gamma_k = (Wc[k:].T @ Wc[:-k]) / (T - k)
        Rk = Gamma_k @ Gamma0_inv
        Qm += (T**2 / (T - k)) * np.trace(Rk.T @ Rk)
    df = n * n * m
    p = _chi2_sf_wilson_hilferty(Qm, df)
    return float(Qm), int(df), float(p)

def arch_lm(x: np.ndarray, L: int) -> Tuple[float, int, float]:
    T = x.size
    y = x ** 2
    Z = np.column_stack([np.ones(T - L)] + [y[L-l:T-l] for l in range(1, L+1)])
    yL = y[L:]
    beta = np.linalg.pinv(Z) @ yL
    yhat = Z @ beta
    SSR = ((yhat - yL.mean())**2).sum()
    TSS = ((yL - yL.mean())**2).sum()
    R2 = 0.0 if TSS < EPS else SSR/TSS
    LM = (T - L) * R2
    p = _chi2_sf_wilson_hilferty(LM, L)
    return float(LM), int(L), float(p)

def multivariate_arch_lm(W: np.ndarray, L: int, use_vech: bool = True) -> Tuple[float, int, float]:
    T, n = W.shape
    if L < 1 or T <= L + 2: return 0.0, 0, 1.0
    
    S_stack = _vec_outer_products(W, use_vech=use_vech)
    d = S_stack.shape[1]
    
    Y = S_stack[L:]
    Z_list = [np.ones((T - L, 1))]
    for l in range(1, L + 1):
        Z_list.append(S_stack[L-l:T-l])
    Z = np.hstack(Z_list)
    
    beta = np.linalg.pinv(Z) @ Y
    Yhat = Z @ beta
    Ybar = Y.mean(axis=0, keepdims=True)
    ESS = np.sum((Yhat - Ybar) ** 2)
    TSS = np.sum((Y - Ybar) ** 2)
    R2 = 0.0 if TSS <= EPS else ESS / TSS
    
    LM = (T - L) * R2
    df = d * L
    p = _chi2_sf_wilson_hilferty(LM, df)
    return float(LM), int(df), float(p)

def suggest_block_lengths(W: np.ndarray, max_lag=24) -> Dict[str, Any]:
    T, n = W.shape
    acfs = np.array([np.abs(acf_1d(W[:, j], max_lag)) for j in range(n)])
    acf_mean = acfs.mean(axis=0)
    
    thresh = 2.0 / sqrt(T)
    L_eff = max_lag
    for k in range(3, max_lag + 1):
        if np.all(acf_mean[k-3:k] < thresh):
            L_eff = k
            break
            
    b1 = max(5, int(np.ceil(1.5 * L_eff)))
    b2 = max(5, int(np.ceil(T**(1/3))))
    return {"b1_acf": min(b1, T//2), "b2_T13": min(b2, T//2), "acf_mean": acf_mean}

# -------------------------- 4. MAIN VALIDATOR --------------------------

def validate_mbb_assumptions_for_tvp(
    fit: Dict[str, Any],
    p: int,
    var_names: Optional[List[str]] = None,
    lb_lag: int = 7,
    arch_lags: int = 7,
    show_summary: bool = True,
    make_plots: bool = False  
) -> Dict[str, Any]:
    n = fit["meta"]["n"]
    u_hat = np.array(fit["innovations"])
    Sigma_t_use = np.array(fit["S_t"])
    
    W = whiten_by_Sigma(u_hat, Sigma_t_use, method="zca")
    min_T = W.shape[0]
    
    b_sugg = suggest_block_lengths(W, max_lag=24)
    
    out = {
        "W": W,
        "block_suggestions": b_sugg
    }
    
    if show_summary:
        print("\n=== Preparación de Residuos (Robust ZCA Whitening) ===")
        print(f">>> Recommended Block Lengths: {b_sugg}")
        
    return out
