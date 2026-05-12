###############################################################
####### Libraries and environment #######
###############################################################
from auxiliar_code.config_set import *
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Literal
from tqdm import tqdm
###############################################################
####### Auxiliar code for the TVP-VAR connectedness #######
###############################################################

def _eigh_clip(
    S: np.ndarray,
    jitter: float = 1e-10,
    min_eig: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetrize, jitter, eigendecompose, and clip eigenvalues of a covariance-like matrix.

    This helper is designed for robust spectral operations (e.g., whitening).
    Given a (possibly slightly non-symmetric or ill-conditioned) matrix S, it:

      1. Symmetrizes S as (S + S.T) / 2.
      2. Adds a small diagonal "jitter" for numerical stability.
      3. Computes the eigen-decomposition S = Q diag(w) Q'.
      4. Clips eigenvalues from below by `min_eig` to avoid zero or negative values.

    Parameters
    ----------
    S : np.ndarray of shape (n, n)
        Symmetric (or nearly symmetric) matrix, e.g. a covariance matrix.
    jitter : float, optional
        Small diagonal term added to S before eigen-decomposition for stability.
    min_eig : float, optional
        Minimum allowed eigenvalue; all eigenvalues are clipped to be >= min_eig.

    Returns
    -------
    w : np.ndarray of shape (n,)
        Clipped eigenvalues of the regularized S.
    Q : np.ndarray of shape (n, n)
        Orthogonal matrix of eigenvectors, so that S ≈ Q diag(w) Q'.
    """
    # Enforce symmetry numerically: S := (S + Sᵀ) / 2
    S = 0.5 * (S + S.T)

    # Get the dimension n of the matrix S
    n = S.shape[0]

    # Add small jitter on the diagonal: S := S + jitter * I
    S = S + jitter * np.eye(n, dtype=S.dtype)

    # Eigen-decompose the regularized matrix S
    w, Q = np.linalg.eigh(S)

    # Clip eigenvalues from below to ensure they are at least min_eig
    w = np.clip(w, min_eig, None)

    # Return the clipped eigenvalues and eigenvectors
    return w, Q

def whiten_by_Sigma(
    u: np.ndarray,
    Sigma_t: np.ndarray,
    method: Literal["zca", "pca"] = "zca",
    jitter: float = 1e-10,
    min_eig: float = 1e-12,
    center: bool = True
) -> np.ndarray:
    """
    Spectral whitening per time t: W_t = Σ_t^{-1/2} u_t.

    For each t, this function:
      1. Takes the covariance matrix Σ_t (n × n).
      2. Eigen-decomposes Σ_t ≈ Q diag(w) Q'.
      3. Clips / regularizes eigenvalues via `_eigh_clip`.
      4. Applies Σ_t^{-1/2} to u_t using either:
         - PCA whitening  : x_t = diag(w^{-1/2}) Q' u_t
         - ZCA whitening  : x_t = Q diag(w^{-1/2}) Q' u_t
      5. Optionally re-centers the whitened series over t.

    Parameters
    ----------
    u : np.ndarray of shape (T, n)
        Time series of residuals/innovations u_t stacked over T periods.
    Sigma_t : np.ndarray of shape (T, n, n)
        Time-varying covariance matrices Σ_t corresponding to each u_t.
    method : {"zca", "pca"}, optional
        Whitening method:
        - "pca": rotates to eigenbasis and rescales (Q' u), then keeps that basis.
        - "zca": rotates to eigenbasis, rescales, then rotates back (Q x).
    jitter : float, optional
        Small diagonal term added inside `_eigh_clip` for numerical stability.
    min_eig : float, optional
        Minimum eigenvalue enforced in `_eigh_clip` to avoid division by zero.
    center : bool, optional
        If True, subtracts the time-average of W across t from each W_t.

    Returns
    -------
    W : np.ndarray of shape (T, n)
        Whitened series W_t with (approximately) identity covariance.
    """
    # Ensure u is a NumPy array
    u = np.asarray(u)
    # Ensure Sigma_t is a NumPy array
    Sigma_t = np.asarray(Sigma_t)
    # Extract number of time points T and dimension n from u
    T, n = u.shape
    # Allocate output array W with same shape and dtype=float
    W = np.empty_like(u, dtype=float)

    # Loop over time t = 0, ..., T-1
    for t in range(T):
        # Eigen-decompose Σ_t with clipping/regularization for numerical stability
        w, Q = _eigh_clip(Sigma_t[t], jitter=jitter, min_eig=min_eig)
        # Compute inverse square roots of eigenvalues (λ^{-1/2})
        inv_sqrt_w = 1.0 / np.sqrt(w)

        # Rotate u_t into eigenbasis: Q' u_t
        x = Q.T @ u[t]
        # Rescale by inverse sqrt eigenvalues: diag(λ^{-1/2}) (Q' u_t)
        x = inv_sqrt_w * x

        # If PCA whitening, keep x in eigenbasis
        if method == "pca":
            W[t] = x
        # If ZCA whitening, rotate back: Q x
        elif method == "zca":
            W[t] = Q @ x
        # Reject invalid method names
        else:
            raise ValueError("method must be 'zca' or 'pca'")

    # Optionally remove the time mean from the whitened series
    if center:
        W = W - W.mean(axis=0, keepdims=True)

    # Return whitened series
    return W

# 1. HELPER DE NORMALIDAD (JARQUE-BERA MULTIVARIADO SIMPLE)
def joint_jarque_bera_test(z_matrix: np.ndarray) -> float:
    """
    Compute the Joint Jarque–Bera normality test for multivariate residuals.

    This follows the approach described in:
        Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
        Springer — section on residual diagnostics.

    Method:
    -------
    1. Orthogonalized residuals (e.g., whitened innovations) `z_matrix`
       should have columns that are uncorrelated under the null.
    2. For each variable i, compute the univariate Jarque–Bera statistic JB_i.
    3. Form the joint statistic:     JB_joint = Σ_i JB_i
    4. Under the null of joint normality:
           JB_joint ~ χ²(df=2n),
       where each series contributes 2 df (skewness and kurtosis).

    Parameters
    ----------
    z_matrix : np.ndarray of shape (T, n)
        Orthogonalized residual matrix. Each column is a univariate residual series.

    Returns
    -------
    p_value : float
        The p-value of the joint chi-square test of normality.
    """
    # Extract number of time observations T and number of series n
    T, n = z_matrix.shape

    # Store each univariate JB statistic
    jb_stats = []

    # Loop over each dimension (column) of z_matrix
    for i in range(n):
        # Compute univariate JB statistic for component i
        # stats.jarque_bera returns (statistic, pvalue)
        jb_val, _ = stats.jarque_bera(z_matrix[:, i])
        jb_stats.append(jb_val)

    # Sum all univariate JB statistics to form the joint statistic
    joint_jb = np.sum(jb_stats)

    # Degrees of freedom: 2 per variable (skewness + kurtosis)
    df = 2 * n

    # Compute joint p-value using the chi-square CDF
    p_value = 1 - stats.chi2.cdf(joint_jb, df)

    # Return p-value of the joint test
    return p_value

# 2. TEST DE HOSKING MULTIVARIADO
def hosking_multivariate_test(
    z_matrix: np.ndarray,
    max_lag: int = 10
) -> Tuple[float, float, int]:
    """
    Multivariate Portmanteau Test of Hosking (1980).

    This implements the Hosking multivariate portmanteau statistic for testing
    the joint absence of serial correlation up to lag `max_lag` in a
    multivariate series z_t.

    Reference
    ---------
    Hosking, J. R. M. (1980).
    "The Multivariate Portmanteau Statistic."
    Journal of the American Statistical Association, 75(371), 602–608.

    Parameters
    ----------
    z_matrix : np.ndarray of shape (T, n)
        Multivariate time series (e.g. whitened residuals), with T time points
        and n variables/series in the columns.
    max_lag : int, optional
        Maximum lag to include in the portmanteau statistic. Default is 10.

    Returns
    -------
    hosking_stat : float
        Value of the Hosking multivariate portmanteau statistic.
    p_value : float
        p-value of the chi-square test under the null of no serial correlation.
    df : int
        Degrees of freedom for the chi-square distribution, df = n^2 * max_lag.

    Notes
    -----
    - If the contemporaneous covariance C0 is singular and cannot be inverted,
      the function returns (np.nan, np.nan, np.nan).
    """
    # Extract sample size T and dimension n from the input matrix
    T, n = z_matrix.shape
    
    # Center the series column-wise by subtracting the mean of each variable
    z_centered = z_matrix - z_matrix.mean(axis=0)

    # Compute contemporaneous covariance matrix C0 = E[z_t z_t']
    C0 = (z_centered.T @ z_centered) / T
    
    # Try to invert C0; if singular, return NaNs
    try:
        C0_inv = np.linalg.inv(C0)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

    # Initialize Hosking's statistic accumulator
    hosking_stat = 0.0

    # Loop over lags j = 1, ..., max_lag
    for j in range(1, max_lag + 1):
        # Compute lag-j cross-covariance C_j = E[z_t z_{t-j}']
        C_j = (z_centered[j:].T @ z_centered[:-j]) / T

        # Compute trace(C_j' C0^{-1} C_j C0^{-1}) term
        term = np.trace(C_j.T @ C0_inv @ C_j @ C0_inv)

        # Add the scaled contribution for lag j
        hosking_stat += (1 / (T - j)) * term

    # Multiply accumulated sum by T^2, as per Hosking's formula
    hosking_stat *= (T * T)

    # Degrees of freedom: n^2 per lag, up to max_lag
    df = (n * n) * max_lag

    # p-value from chi-square distribution with df degrees of freedom
    p_value = 1 - stats.chi2.cdf(hosking_stat, df)
    
    # Return statistic, p-value, and degrees of freedom
    return hosking_stat, p_value, df

def mardia_multivariate_test(X: np.ndarray) -> Tuple[float, float, float, float]:
    
    """
    Mardia's multivariate skewness and kurtosis tests (Mardia, 1970).

    This function computes:
      - Mardia's multivariate skewness statistic and its chi-square p-value.
      - Mardia's multivariate kurtosis statistic and its asymptotic normal p-value.

    It assumes that X are (approximately) centered or whitened/orthogonal residuals.

    Reference
    ---------
    Mardia, K. V. (1970).
    "Measures of Multivariate Skewness and Kurtosis with Applications."
    Biometrika, 57(3), 519–530.

    Parameters
    ----------
    X : np.ndarray of shape (T, n)
        Data matrix (e.g., whitened residuals), where T is the sample size
        and n is the dimension (number of variables).

    Returns
    -------
    skew_stat : float
        Mardia's multivariate skewness test statistic.
    skew_p : float
        p-value of the skewness test based on a chi-square approximation.
    kurt_stat : float
        Mardia's multivariate kurtosis test statistic (standard normal).
    kurt_p : float
        Two-sided p-value for the kurtosis test based on N(0, 1).
    """
    # Extract sample size T and dimension n
    T, n = X.shape
    
    # Center the data column-wise (crucial if X is not exactly centered already)
    X_c = X - X.mean(axis=0)
    
    # Compute the sample covariance matrix (MLE version with T in the denominator)
    S = (X_c.T @ X_c) / T

    # Invert the covariance matrix S
    S_inv = np.linalg.inv(S)
    
    # Compute the matrix of pairwise Mahalanobis products:
    # D_ij = (x_i - mean)' S^-1 (x_j - mean)
    # Efficient vectorized form: D = X_c @ S_inv @ X_c.T
    # Note: This is T x T and may be heavy in RAM if T is very large.
    D = X_c @ S_inv @ X_c.T
    
    # --- 1. MARDIA MULTIVARIATE SKEWNESS ---
    
    # b_1,n = (1 / T^2) * sum_i sum_j (D_ij)^3
    b1 = (1.0 / (T**2)) * np.sum(np.power(D, 3))
    
    # Skewness test statistic: skew_stat = (T * b1) / 6
    skew_stat = (T * b1) / 6.0

    # Degrees of freedom: n(n + 1)(n + 2) / 6 for skewness
    df_skew = n * (n + 1) * (n + 2) / 6.0

    # p-value from chi-square distribution with df_skew degrees of freedom
    skew_p = 1.0 - stats.chi2.cdf(skew_stat, df_skew)
    
    # --- 2. MARDIA MULTIVARIATE KURTOSIS ---
    
    # b_2,n = (1 / T) * sum_i (D_ii)^2,
    # where D_ii is the i-th diagonal element of D
    b2 = (1.0 / T) * np.sum(np.diag(D)**2)
    
    # Expected value of b2 under multivariate normality: n(n + 2)
    expected_b2 = n * (n + 2)

    # Asymptotic variance of b2 under normality: 8n(n + 2)/T
    var_b2 = (8.0 * n * (n + 2)) / T

    # Kurtosis test statistic (approximately standard normal)
    kurt_stat = (b2 - expected_b2) / np.sqrt(var_b2)
    
    # Two-sided p-value based on the standard normal distribution
    kurt_p = 2.0 * (1.0 - stats.norm.cdf(abs(kurt_stat)))
    
    # Return skewness statistic, skewness p-value, kurtosis statistic, kurtosis p-value
    return skew_stat, skew_p, kurt_stat, kurt_p

def multivariate_lm_test(residuals, max_lag=5):
    """
    Test LM (Lagrange Multiplier) de Breusch-Godfrey Multivariado.
    Evalúa autocorrelación serial mediante una regresión auxiliar VAR.
    
    H0: No hay autocorrelación serial hasta el rezago 'max_lag'.
    """
    T, n = residuals.shape
    
    # 1. Preparar regresión auxiliar: u_t contra u_{t-1}, ..., u_{t-p}
    # Y_aux = u_t (cortado)
    # X_aux = [u_{t-1}, ..., u_{t-p}]
    
    # Recortar datos para alinear
    Y_aux = residuals[max_lag:]
    T_eff = Y_aux.shape[0]
    
    # Construir matriz de rezagos
    X_lags = []
    for p in range(1, max_lag + 1):
        X_lags.append(residuals[max_lag - p : -p])
    
    X_aux = np.hstack(X_lags) # (T_eff, n * max_lag)
    
    # Agregar constante a la regresión auxiliar (recomendado Breusch-Godfrey)
    X_aux = np.hstack([X_aux, np.ones((T_eff, 1))])
    
    # 2. Estimar el modelo auxiliar (VAR restringido vs no restringido)
    # Modelo restringido (H0): u_t = error (RSS0 es simplemente la covarianza de u)
    # Modelo no restringido (H1): u_t = lags + error
    
    # Covarianza de residuos bajo H0 (sin lags)
    Sigma_0 = (Y_aux.T @ Y_aux) / T_eff
    
    # Estimar coeficientes auxiliares: B = (X'X)^-1 X'Y
    # Usamos lstsq para estabilidad
    B_aux = np.linalg.lstsq(X_aux, Y_aux, rcond=None)[0]
    
    # Residuos de la regresión auxiliar
    E_aux = Y_aux - X_aux @ B_aux
    
    # Covarianza de residuos bajo H1 (con lags)
    Sigma_aux = (E_aux.T @ E_aux) / T_eff
    
    # 3. Calcular estadístico LM
    # LM = T * (n - trace(Sigma_1 * Sigma_0^-1)) ??? 
    # Versión más robusta: T * (ln|S0| - ln|S_aux|) (Likelihood Ratio approx)
    # O la versión clásica de Breusch-Godfrey para VAR:
    # LM = T_eff * [ n - tr( Sigma_aux * Sigma_0^-1 ) ] (Edgerton & Shukur small sample)
    # Usemos la aproximación asintótica Chi2 estándar:
    
    try:
        det_0 = np.linalg.det(Sigma_0)
        det_aux = np.linalg.det(Sigma_aux)
        
        if det_aux <= 0 or det_0 <= 0:
            return np.nan, np.nan # Error numérico
            
        # Estadístico LR (asintóticamente equivalente a LM)
        lm_stat = (T_eff - (n * max_lag) - n - 1) * (np.log(det_0) - np.log(det_aux))
        
        # Grados de libertad: n^2 * lags
        df = (n**2) * max_lag
        p_value = 1.0 - stats.chi2.cdf(lm_stat, df)
        
        return lm_stat, p_value
        
    except np.linalg.LinAlgError:
        return np.nan, np.nan

# MATRIX BUILDING FOR TVP-VAR
def build_var_lagged_jax_ready(
    df: pd.DataFrame,
    cols: Sequence[str],
    p: int,
    add_const: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, pd.Index, int, int]:
    """
    Build lagged VAR design matrices (Z, Y) in a JAX-ready format.

    This function:
      - Extracts the selected columns from a DataFrame.
      - Builds a standard VAR(p) regression setup:
          Y_t   = [y_t]        for t = p, ..., T-1
          Z_t   = [y_{t-1}', ..., y_{t-p}'] (and optional constant)
      - Returns Y and Z as JAX arrays, plus useful meta information.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a DateTime (or numeric) index and series in columns.
    cols : list of str
        Column names from `df` to include in the VAR system.
    p : int
        VAR lag order (number of lags). Must be >= 1 and < number of rows.
    add_const : bool, optional
        If True, appends a column of ones to Z as a constant term. Default is False.

    Returns
    -------
    Y_jax : jax.numpy.ndarray, shape (T_eff, n)
        Matrix of dependent variables, where T_eff = T_full - p, n = number of series.
    X_jax : jax.numpy.ndarray, shape (T_eff, k)
        Matrix of regressors (stacked lags, and optional constant).
    idx : pandas.Index, length T_eff
        Index (aligned with Y and Z), corresponding to df.index[p:].
    n : int
        Number of variables in the VAR (len(cols)).
    k : int
        Number of regressors in each row of Z (n * p [+ 1 if add_const]).

    Raises
    ------
    KeyError
        If any column in `cols` is missing from `df`.
    ValueError
        If p < 1 or if p >= number of rows in `df`.
    """
    # Check that all requested columns exist in the DataFrame
    if not all(c in df.columns for c in cols):
        raise KeyError("Some requested columns are not present in the DataFrame.")
    
    # Check that the lag order p is a positive integer 
    if not isinstance(p, int) or p < 1:
        raise ValueError("p must be a positive integer (p >= 1).")
    
    # Get the full number of observations and number of variables
    T_full = df.shape[0]
    
    # Ensure that we have enough observations to support p lags
    if p >= T_full:
        raise ValueError("p must be smaller than the number of rows in df.")
    
    # Extract the numeric data as a NumPy array (T_full, n)
    data = df[cols].values.astype(np.float64)
    
    # Get dimensions after extraction
    T_full, n = data.shape
    
    # Build the dependent variable matrix Y (drop the first p observations)
    Y = data[p:]
    
    # Initialize a list to hold lagged regressors for each lag
    X_list = []
    
    # Construct lagged regressors for lags 1 to p
    for lag in range(1, p + 1):
        # For lag l, align y_{t-l} with Y_t (t = p,...)
        X_list.append(data[p - lag : -lag])
    
    # Horizontally stack all lag matrices to form the full regressor matrix Z
    X = np.hstack(X_list)
    
    # Optionally append a constant column of ones to Z
    if add_const:
        X = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
        
    # Build the aligned index (same length as Y and Z)
    idx = df.index[p:]
    
    # Number of regressors per observation (including constant if present)
    k = X.shape[1]
    
    # Convert Y and Z to JAX arrays for downstream JAX-based algorithms
    Y_jax = jnp.array(Y)
    X_jax = jnp.array(X)
    
    # Return JAX arrays, index, and dimension meta information
    return Y_jax, X_jax, idx, n, k

# =============================================================================
# 2. CORE DE KALMAN (JAX SCAN - OPTIMIZADO)
# =============================================================================
# TVP-VAR KALMAN FILTER WITH JAX SCAN
@partial(jit, static_argnames=['return_covariances'])
def run_tvp_kalman_jax(Y, X, Sigma0, a0, P0, kappa1, kappa2, return_covariances=False):
    """
    Time-varying parameter VAR Kalman filter (TVP-VAR) implemented in JAX with lax.scan.

    Parameters
    ----------
    Y : jax.numpy.ndarray, shape (T, n)
        Observed time series (dependent variables).
    Z : jax.numpy.ndarray, shape (T, k)
        Regressor matrix (e.g. stacked lags and constant) aligned with Y.
    Sigma0 : jax.numpy.ndarray, shape (n, n)
        Initial covariance matrix of the observation noise.
    a0 : jax.numpy.ndarray, shape (n*k,)
        Initial state mean (vectorized coefficient matrix B_0).
    P0 : jax.numpy.ndarray, shape (n*k, n*k)
        Initial state covariance matrix for β_0.
    kappa1 : float
        State discount factor for the covariance of a_t (0 < κ₁ <= 1).
    kappa2 : float
        EWMA discount factor for Sigma_t (0 < k2 <= 1).
    n : int
        Number of dependent variables in Y (dimension of y_t).
    k : int
        Number of regressors in x_t (columns of X, and columns of B_t).

    Returns
    -------
    history : dict of jax.numpy.ndarray
        Dictionary of filtered objects over time t = 0,...,T-1 with keys:
        - 'beta_filt' : filtered a_t, shape (T, n*k)
        - 'Sigma_t'   : Sigma_t, shape (T, n, n)
        - 'y_pred'    : one-step-ahead predictions ŷ_t = B_{t-1} x_t, shape (T, n)
        - 'loglik'    : per-period Gaussian log-likelihood contributions, shape (T,)
        - 'S_t'       : one-step-ahead innovation covariance matrices, shape (T, n, n)
        - 'innovations' : v_t = y_t - y_ajustada_t, shape (T, n)
        - 'K_t'       : Kalman gains, shape (T, n*k, n)
        - 'P_filt'    : filtered state covariances P_t, shape (T, n*k, n*k)
        - 'P_pred'    : predicted state covariances P_{t|t-1}, shape (T, n*k, n*k)
    """
    # Initialize the scan carrier with initial state mean, covariance, and Sigma
    initial_carrier = (a0, P0, Sigma0)
    
    n = Y.shape[1]
    k = X.shape[1]
    
    # Define one Kalman filter step to be used inside lax.scan
    def kalman_step(carrier, inputs):
        # Unpack previous state mean, state covariance, and Sigma
        a_prev, P_prev, Sigma_prev = carrier

        # Unpack current observation and regressor
        y_t, z_t = inputs
        
        # --- 1. State-based prediction of y_t ---
        # Reshape previous state vector into coefficient matrix B_{t-1} of shape (n, k)
        B_prev = a_prev.reshape(n, k)
        # Compute one-step-ahead prediction y_gorro_t = B_{t-1} z_t
        y_pred = B_prev @ z_t
        # Innovation (prediction error) v_t = y_t - y_gorro_t
        v_t = y_t - y_pred 
        
        # --- 2. Innovation covariance S_t ---
        # Reshape state covariance P into 4D tensor (n, k, n, k)
        P_4d = P_prev.reshape(n, k, n, k)
        # Compute z_t′ P z_t in matrix form using tensor contraction
        HPHt = jnp.einsum('a, iajb, b -> ij', z_t, P_4d, z_t)
        # Add observation noise covariance Σ_{t-1} to get S_t
        S_t = HPHt + Sigma_prev
        S_t = 0.5 * (S_t + S_t.T)
        S_t = S_t + jnp.eye(n)* (1e-10 * jnp.trace(S_t) / n + 1e-12)
        # --- 3. Kalman gain K_t ---
        # Compute P H′ in tensor form: contraction over regressor index
        PHt_tensor = jnp.einsum('iajb, b -> iaj', P_4d, z_t)
        # Flatten (i, a, j) into (n*k, n) for PHt
        PHt = PHt_tensor.reshape(n * k, n)
        
        # Solve for K_t in K_t S_t = PHt instead of inverting S_t
        # Here we solve S_t′ K_t′ = PHt′ and transpose back
        K_t = jnp.linalg.solve(S_t.T, PHt.T).T 
        
        # --- 4. Posterior state mean update ---
        # Update state mean: a_t = a_{t-1} + K_t v_t
        a_post = a_prev + K_t @ v_t
        
        # --- 5. Posterior state covariance update ---
        # Update state covariance: P_t = P_{t|t-1} - K_t S_t K_t′
        P_post = P_prev - K_t @ S_t @ K_t.T
        # Enforce symmetry in P_t to control numerical asymmetry
        P_post = 0.5 * (P_post + P_post.T)
        
        # --- 6. Observation covariance update (EWMA) ---
        # Reshape updated state into B_t (n × k)
        B_post = a_post.reshape(n, k)
        # Recompute prediction with updated B_t (for covariance updating)
        y_pred_post = B_post @ z_t
        # Post-update residual e_t = y_t - B_t x_t
        e_post = y_t - y_pred_post
        
        # Outer product of residuals for the new information in Σ_t
        Gf = jnp.outer(e_post, e_post)
        # EWMA update: Σ_t = k₂ Sigma_{t-1} + (1 - k₂) e_t e_t′
        Sigma_post = kappa2 * Sigma_prev + (1.0 - kappa2) * Gf
        # Enforce symmetry in Σ_t
        Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)
        
        # --- 7. Per-period Gaussian log-likelihood ---
        # Compute log determinant of S_t robustly via slogdet
        sign, logdet = jnp.linalg.slogdet(S_t)
        # Mahalanobis term v_t′ S_t^{-1}` v_t (solve instead of inverse)
        mahalanobis = v_t @ jnp.linalg.solve(S_t, v_t)
        # Log-density of N(0, S_t) evaluated at v_t
        ll_t = -0.5 * (n * jnp.log(2 * jnp.pi) + logdet + mahalanobis)

        # --- 8. Time update for next step (state discounting) ---
        # Inflate state covariance by discount factor κ₁: P_{t|t} → P_{t+1|t}
        P_next = P_post / kappa1
        # The next prior mean equals the current posterior mean
        a_next = a_post
        
        # Collect outputs for this time step in a dictionary
        step_out = {
                    'beta_filt': a_post,
                    'Sigma_t': Sigma_post,
                    'y_pred': y_pred,
                    'loglik': ll_t,
                    'S_t': S_t,
                    'innovations': v_t,
                    'K_t':K_t
                }
        
        if return_covariances:
            step_out['P_filt'] = P_post
            step_out['P_pred'] = P_prev
        
        # Return updated carrier and the step output
        return (a_next, P_next, Sigma_post), step_out

    # Run the compiled scan over time with initial state and inputs (Y, X)
    last_carrier, history = lax.scan(kalman_step, initial_carrier, (Y, X))

    # Return the full history (each entry stacked over time)
    return history

# FEVD / CONECTIVIDAD (JAX VMAP)
# ====================================================================
# MODIFICACIÓN 1: compute_girf_single_step (Añade sub_mask y calcula sub-red)
# ====================================================================
@partial(jit, static_argnames=['p', 'H', 'n', 'k'])
def compute_girf_single_step(B_flat, Sigma,sub_mask, p, H, n, k):
    """
    Compute generalized FEVD and connectedness indices for a single time t
    using an optimized, JAX-friendly implementation.

    Parameters
    ----------
    B_flat : jnp.ndarray of shape (n * k,)
        Flattened VAR coefficient matrix at time t (row-stacked).
    Sigma : jnp.ndarray of shape (n, n)
        Covariance matrix of shocks at time t.
    p : int
        VAR lag order.
    H : int
        GIRF horizon (number of steps ahead for FEVD).
    n : int
        Number of variables in the VAR.
    k : int
        Number of regressors per equation (n * p [+ 1 if constant]).

    Returns
    -------
    theta_norm : jnp.ndarray of shape (n, n)
        Row-normalized generalized FEVD matrix.
    tci : jnp.ndarray (scalar)
        Total Connectedness Index, in percentage points.
    to_idx : jnp.ndarray of shape (n,)
        TO connectedness for each variable.
    from_idx : jnp.ndarray of shape (n,)
        FROM connectedness for each variable.
    net_idx : jnp.ndarray of shape (n,)
        NET connectedness (TO − FROM) per variable.
    pdc : jnp.ndarray of shape (n, n)
        Pairwise directional connectedness matrix, PDC_ij = θ_ij − θ_ji.
    """
    # 1. Reshape flat coefficient vector into (n × k) matrix
    B = B_flat.reshape(n, k)
    has_const = (k > n * p)
    # Si hay constante, la descartamos omitiendo la última columna, ya que no afecta las respuestas a impulsos
    A_coeffs = B[:, :-1] if has_const else B
    # Reorganizamos los coeficientes en un tensor 3D de forma (rezagos, variables, variables) -> (p, n, n)
    A_3d = A_coeffs.reshape(n, p, n).transpose(1, 0, 2)
    
    # Definimos el paso recursivo de medias móviles (VMA) para utilizarlo en el bucle lax.scan
    def step_fn(Phi_window, _):
        # Calculamos el siguiente Phi sumando los productos de cada rezago con su Phi histórico correspondiente
        new_phi = jnp.einsum('lij,ljk->ik', A_3d, Phi_window)
        # Desplazamos la ventana histórica insertando el nuevo Phi al inicio y descartando el más antiguo
        new_window = jnp.concatenate([new_phi[None, ...], Phi_window[:-1]], axis=0)
        # Retornamos la nueva ventana (para la siguiente iteración) y el nuevo Phi (para guardarlo en el historial)
        return new_window, new_phi
    # Inicializamos la ventana histórica de tamaño (p, n, n) llena de ceros
    init_window = jnp.zeros((p, n, n))
    # Establecemos la matriz identidad en el primer elemento de la ventana (Phi_0 = I)
    init_window = init_window.at[0].set(jnp.eye(n))
    # Guardamos Phi_0 como un tensor independiente para concatenarlo más tarde
    Phi_0 = jnp.eye(n)[None, ...]
    # Ejecutamos el bucle optimizado de JAX H-1 veces para obtener los impulsos respuesta futuros
    _, Phi_rest = lax.scan(step_fn, init_window, None, length=H - 1)
    # Unimos Phi_0 con el resto de las proyecciones para obtener el tensor completo de respuestas (H, n, n)
    Phi = jnp.concatenate([Phi_0, Phi_rest], axis=0)
    # Extraemos las varianzas individuales (elementos de la diagonal) de la matriz de covarianza de choques
    sigma_diag = jnp.diag(Sigma)

    # Definimos la función que calcula el numerador y denominador de la GFEVD para un solo paso 'h'
    def compute_metrics_optimized(G):
        G_Sig = G @ Sigma  
        num = (G_Sig) ** 2 / sigma_diag
        den = jnp.sum(G_Sig * G, axis=1)
        return num, den
    # Vectorizamos la función anterior para aplicarla en paralelo a todos los horizontes (h=0 hasta H-1)
    nums, dens = vmap(compute_metrics_optimized)(Phi)
    # Sumamos todas las contribuciones a lo largo del horizonte temporal H para obtener el numerador total
    numer = jnp.sum(nums, axis=0)
    # Sumamos todas las contribuciones a lo largo del horizonte temporal H para obtener el denominador total
    denom = jnp.sum(dens, axis=0)
    
    # -----------------------------------------------------
    # RED COMPLETA (Original)
    # -----------------------------------------------------
    # Dividimos el numerador entre el denominador para obtener la matriz cruda de descomposición de varianza
    theta = numer / denom[:, None]
    # Calculamos la suma total de cada fila para el proceso de normalización
    theta_row_sum = jnp.sum(theta, axis=1, keepdims=True)
    # Dividimos cada elemento de la fila por su suma para que los pesos de contagio sumen 1 (100%)
    theta_norm = theta / theta_row_sum
    # TO (Emisión): Suma de la columna (cuánto afecta a otros) menos su propia inercia (diagonal), expresado en %
    to_idx = 100 * (jnp.sum(theta_norm, axis=0) - jnp.diag(theta_norm))/n
    # FROM (Recepción): Suma de la fila (cuánto recibe de otros) menos su propia inercia (diagonal), expresado en %
    from_idx = 100 * (jnp.sum(theta_norm, axis=1) - jnp.diag(theta_norm))/n
    # NET (Direccional Neto): Diferencia entre lo que emite y lo que recibe el nodo
    net_idx = to_idx - from_idx
    # TCI (Índice de Conectividad Total): Suma de todos los contagios TO (porcentaje de la varianza del sistema)
    tci = jnp.sum(to_idx) 
    # PDC (Contagio Bilateral): Diferencia neta entre los shocks intercambiados directamente por cada par de nodos
    pdc = theta_norm - theta_norm.T 
    
    # -----------------------------------------------------
    # SUB-RED AISLADA (USA y México)
    # -----------------------------------------------------
    # 1. Filtramos los numeradores usando la máscara
    numer_sub = numer * sub_mask[:, None] * sub_mask[None, :]
    
    # 2. Re-normalizamos la suma de las filas DENTRO de la sub-red
    denom_sub = jnp.sum(numer_sub, axis=1)
    theta_sub = numer_sub / jnp.where(denom_sub > 1e-12, denom_sub, 1.0)[:, None]
    
    own_sub = jnp.diag(theta_sub)
    n_sub = jnp.maximum(jnp.sum(sub_mask), 1.0)
    
    to_sub = 100 * (jnp.sum(theta_sub, axis=0) - own_sub) / n_sub
    from_sub = 100 * (jnp.sum(theta_sub, axis=1) - own_sub) / n_sub
    net_sub = to_sub - from_sub
    tci_sub = jnp.sum(to_sub)
    
    return theta_norm, theta, tci, to_idx, from_idx, net_idx, pdc, tci_sub, net_sub

# ====================================================================
# MODIFICACIÓN 2: compute_connectedness_jax (Pasa el sub_mask)
# ====================================================================
@partial(jit, static_argnames=['p', 'H', 'n', 'k'])
def compute_connectedness_jax(betas_filt, Sigmas_filt, sub_mask, p, H, n, k):
    return vmap(
        compute_girf_single_step,
        in_axes=(0, 0, None, None, None, None, None) # sub_mask entra como None (estático para todo t)
    )(betas_filt, Sigmas_filt, sub_mask, p, H, n, k)

###############################################################
####### Selecting the best model #######
###############################################################

# Light version to compute ONLY log-likelihood
#******CORROBORADA********#
def get_kalman_loglik(Y, X, Sigma0, a0, P0, kappa1, kappa2):
    """
    Compute ONLY the total log-likelihood of the TVP-VAR Kalman filter.
    """
    # Run the Kalman filter (JAX-compiled)
    history = run_tvp_kalman_jax(Y, X, Sigma0, a0, P0, kappa1, kappa2)
    return jnp.sum(history['loglik'])

# Batched version to compute log-likelihoods for multiple (kappa1, kappa2) pairs
# in_axes: (None...) para datos fijos, (0, 0) para los arrays de kappas
batch_loglik_calculator = jit(vmap(get_kalman_loglik, in_axes=(None, None, None, None, None, 0, 0)))

# Main model selection function with two-stage funnel approach
def select_best_model_aic_parallel(
    series: pd.DataFrame,
    cols: Sequence[str],
    p_grid: Sequence[int],
    k1_grid: Sequence[float],
    k2_grid: Sequence[float],
    add_const: bool = False,
    top_candidates: int = 20
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Funnel strategy (two-stage search) for TVP-VAR discount factors using AIC + diagnostics.

    Stage 1 (Fast grid search)
    --------------------------
    For each p in p_grid and each (kappa1, kappa2) combination in the cartesian
    product of k1_grid x k2_grid:
      - Build JAX-ready VAR lag matrices (Y, Z).
      - Trim all models to a common sample (max_p alignment).
      - Compute OLS priors (a0, P0, Sigma0).
      - Evaluate the log-likelihood via a batched JAX function
        `batch_loglik_calculator(...)`.
      - Store AIC values and basic metadata for each combination.

    Stage 2 (Deep diagnostics)
    --------------------------
    For the best `top_candidates` models according to AIC:
      - Rebuild Y, Z, and OLS priors.
      - Run the full Kalman filter `run_tvp_kalman_jax(...)`.
      - Whiten innovations and run multivariate Hosking tests (mean + ARCH)
        and a joint normality test.
      - Store p-values (mean, ARCH, normality) and mark models as "Tested".

    Final selection
    ---------------
    - Prefer models that pass both Hosking tests (p > 0.05) and then pick the
      lowest AIC among them.
    - If none pass, fall back to the global minimum AIC model.

    Parameters
    ----------
    series : pandas.DataFrame
        Original data (levels or returns) with a time index and variables in columns.
    cols : Sequence[str]
        Names of the columns in `series` to include in the VAR system.
    p_grid : Sequence[int]
        Grid of VAR lag orders to consider (e.g., [1, 2, 3, 4]).
    k1_grid : Sequence[float]
        Grid of state discount factors kappa1 to search over.
    k2_grid : Sequence[float]
        Grid of observation covariance discount factors kappa2 to search over.
    add_const : bool, optional
        If True, include a constant in the VAR regressors. Default is False.
    top_candidates : int, optional
        Number of top AIC models to send to Stage 2 diagnostics. Default is 20.

    Returns
    -------
    best_model : pandas.Series
        Row of the results DataFrame corresponding to the selected model
        (contains p, kappa1, kappa2, AIC, p-values, etc.).
    df_results : pandas.DataFrame
        DataFrame with all models and their AIC and diagnostic statistics.
    """
    # Initialize a list to collect results for every (p, kappa1, kappa2) combination
    results_list = []
    
    # --- STAGE 1: FAST SWEEP (AIC/BIC) ---
    
    # Compute the maximum lag order across the grid (for common trimming)
    max_p = max(p_grid)
    
    # Prepare the kappa grids as JAX arrays for vectorized evaluation
    k1_mesh, k2_mesh = jnp.meshgrid(jnp.array(k1_grid), jnp.array(k2_grid))
    
    # Flatten the kappa grids to 1D arrays
    k1_flat = k1_mesh.flatten()
    k2_flat = k2_mesh.flatten()
    
    # Count total combinations in the grid search (p x all kappa pairs)
    total_combinations = len(p_grid) * k1_flat.size
    print(f"🚀 ETAPA 1: Grid Search Rápido en {total_combinations} combinaciones...")
    
    # Loop over each lag order p in the grid
    for p in tqdm(p_grid, desc="Calculando AIC/BIC"):
        # 1. Build JAX-ready VAR lagged design matrices for this p
        Y, X, _, n, k = build_var_lagged_jax_ready(series, cols, p, add_const)
        
        # 2. Enforce a strict common sample across all p by trimming to max_p
        trim_rows = max_p - p
        if trim_rows > 0:
            # Use only the last T_eff = T_full - max_p observations
            Y_eff, X_eff = Y[trim_rows:], X[trim_rows:]
        else:
            # If p == max_p, do not trim
            Y_eff, X_eff = Y, X
        
        # Effective number of time points after trimming
        T_eff = len(Y_eff)
        
        # 3. Compute OLS priors using NumPy (on CPU)
        X_np, Y_np = np.array(X_eff), np.array(Y_eff)
        
        # OLS estimate for coefficients: B_ols has shape (k, n)
        B_ols = np.linalg.lstsq(X_np, Y_np, rcond=None)[0]
        
        # Residuals E_ols = Y - X B̂
        E_ols = Y_np - X_np @ B_ols
        
        # Observation covariance Sigma_0 estimated from OLS residuals
        Sigma0 = (E_ols.T @ E_ols) / T_eff
        
        # Flatten coefficient matrix by rows (n*k,) for state mean a_0
        a0 = B_ols.T.flatten()
        
        # Compute (X'X)^(-1) as a pseudo-inverse for diffuse priors
        XtX_inv = np.linalg.pinv(X_np.T @ X_np)
        
        # State covariance P_0 via Kronecker product, scaled to be diffuse
        P0 = np.kron(Sigma0, XtX_inv) * 10.0
        
        # 4. Massive log-likelihood computation using the batch JAX calculator
        # Convert inputs to JAX arrays only once per p
        logliks = batch_loglik_calculator(
            jnp.array(Y_eff),
            jnp.array(X_eff),
            jnp.array(Sigma0),
            jnp.array(a0),
            jnp.array(P0),
            k1_flat,
            k2_flat
        )
        
        # Convert log-likelihoods back to NumPy for AIC computation
        logliks_np = np.array(logliks)
        
        # Approximate total number of parameters: VAR coefficients + covariance params
        n_params = (n * k) + (n * (n + 1) // 2)
        
        # 5. Compute AIC for each (k1, k2) in the flattened grid
        aics = -2.0 * logliks_np + 2.0 * n_params
        
        # Store results for each combination of (p, kappa1, kappa2)
        for i in range(len(aics)):
            results_list.append({
                'p': p,
                'kappa1': float(k1_flat[i]),
                'kappa2': float(k2_flat[i]),
                'AIC': float(aics[i]),
                'T_eff': T_eff,
                'Prob(H_Mean)': np.nan,
                'Prob(H_ARCH)': np.nan,
                'Prob(Norm)': np.nan,
                'Prob(M_Skew)': np.nan, # Mardia Skewness (NUEVO)
                'Prob(M_Kurt)': np.nan, # Mardia Kurtosis (NUEVO)
                'Prob(LM)': np.nan,     # LM Test (NUEVO)
                'Tested': False
            })

    # Convert the list of result dicts into a DataFrame
    df_results = pd.DataFrame(results_list)
    
    # --- STAGE 2: DEEP DIAGNOSTICS (ONLY TOP CANDIDATES) ---
    
    # Identify indices of the top `top_candidates` models by AIC
    candidates_idx = df_results.sort_values(by="AIC").head(top_candidates).index
    print(f"\n🔬 ETAPA 2: Ejecutando Filtro Kalman Completo en los mejores {top_candidates} modelos...")
    
    # Loop over each candidate model for detailed diagnostics
    for idx in tqdm(candidates_idx, desc="Diagnosticando"):
        # Extract the row corresponding to this candidate
        row = df_results.loc[idx]
        
        # Current lag order and discount factors
        p_curr = int(row['p'])
        k1_curr = row['kappa1']
        k2_curr = row['kappa2']
        
        # Rebuild VAR design matrices for this specific p
        Y, X, _, n, k = build_var_lagged_jax_ready(series, cols, p_curr, add_const)
        
        # Apply the same trimming logic as in Stage 1
        trim_rows = max_p - p_curr
        if trim_rows > 0:
            Y_eff, X_eff = Y[trim_rows:], X[trim_rows:]
        else:
            Y_eff, X_eff = Y, X
            
        # Recompute OLS priors (NumPy) for this effective sample
        X_np, Y_np = np.array(X_eff), np.array(Y_eff)
        
        # OLS coefficients for initialization
        B_ols = np.linalg.lstsq(X_np, Y_np, rcond=None)[0]
        
        # Residuals for this candidate
        E_ols = Y_np - X_np @ B_ols
        
        # Observation covariance Σ₀ for this candidate
        Sigma0 = (E_ols.T @ E_ols) / len(Y_np)
        
        # Flatten coefficient matrix as initial state mean a₀
        a0 = B_ols.T.flatten()
        
        # State covariance P₀ via Kronecker of Σ₀ and (X'X)^(-1)
        P0 = np.kron(Sigma0, np.linalg.pinv(X_np.T @ X_np)) * 10.0
        
        try:
            # ==========================================================
            # RUN USER'S TVP KALMAN FILTER (run_tvp_kalman_jax)
            # ==========================================================
            
            # Convert inputs to JAX arrays for the filter
            Y_jax = jnp.array(Y_eff)
            X_jax = jnp.array(X_eff)
            Sigma0_jax = jnp.array(Sigma0)
            a0_jax = jnp.array(a0)
            P0_jax = jnp.array(P0)
            
            # IMPORTANT: pass n, k, and p_curr (static args) to the filter
            history = run_tvp_kalman_jax(
                Y_jax,
                X_jax,
                Sigma0_jax,
                a0_jax,
                P0_jax,
                float(k1_curr),
                float(k2_curr)
            )
            
            # Extract innovations and S_t, converting back to NumPy for tests
            v_t = np.array(history['innovations'])  # shape (T_eff, n)
            S_t = np.array(history['S_t'])          # shape (T_eff, n, n)
            
            # ==========================================================
            # STATISTICAL TESTS (WHITENING + HOSKING)
            # ==========================================================
            
            # Whiten residuals using the time-varying covariance S_t
            W = whiten_by_Sigma(v_t, S_t, method="zca")
            
            # Hosking test on mean (first moments)
            _, p_mean, _ = hosking_multivariate_test(W, max_lag=10)
            
            # Hosking test on ARCH effects (squared residuals)
            _, p_arch, _ = hosking_multivariate_test(W**2, max_lag=10)
            
            # Joint Jarque–Bera-type test for multivariate normality
            p_norm = joint_jarque_bera_test(W)
            
            # --- NEW TESTS (Mardia + LM) ---
            _, p_mardia_skew, _, p_mardia_kurt = mardia_multivariate_test(W)
            
            _, p_lm = multivariate_lm_test(W, max_lag=5)
            
            # Update the results DataFrame with diagnostic p-values
            df_results.at[idx, 'Prob(H_Mean)'] = p_mean
            df_results.at[idx, 'Prob(H_ARCH)'] = p_arch
            df_results.at[idx, 'Prob(Norm)'] = p_norm
            df_results.at[idx, 'Prob(M_Skew)'] = p_mardia_skew
            df_results.at[idx, 'Prob(M_Kurt)'] = p_mardia_kurt
            df_results.at[idx, 'Prob(LM)'] = p_lm
            df_results.at[idx, 'Tested'] = True
            
        except Exception as e:
            # Report any error for this candidate without stopping the entire search
            print(f"⚠️ Fallo en modelo p={p_curr}, k1={k1_curr}: {e}")

    # --- FINAL MODEL SELECTION ---
    
    # Filter models that were actually tested and pass both Hosking tests
    valid_models = df_results[
        (df_results['Tested'] == True) &
        (df_results['Prob(H_Mean)'] > 0.05) &
        (df_results['Prob(H_ARCH)'] > 0.05)
    ]
    
    # If at least one model passes the diagnostic criteria
    if not valid_models.empty:
        # Select the valid model with the minimum AIC
        best_model = valid_models.sort_values(by="AIC").iloc[0]
        print(f"\n✅ GANADOR ROBUSTO ENCONTRADO (Pasó tests estadísticos).")
    else:
        # Otherwise, fall back to the global AIC minimum across all models
        print(f"\n⚠️ ADVERTENCIA: Ningún candidato pasó los tests rigurosos. Se retorna el mejor AIC simple.")
        best_model = df_results.sort_values(by="AIC").iloc[0]

    # Pretty-print basic information about the selected model
    print(f"🏆 Modelo Seleccionado: p={int(best_model['p'])}, k1={best_model['kappa1']:.4f}, k2={best_model['kappa2']:.4f}")
    print(f"   AIC: {best_model['AIC']:.2f}")
    print(f"   P-Values: Mean={best_model['Prob(H_Mean)']:.4f}, ARCH={best_model['Prob(H_ARCH)']:.4f}, Norm={best_model['Prob(Norm)']:.4f}")

    # Return the winning row and the full results table
    return best_model, df_results

def select_best_model_aic_continuous_square(
    series: pd.DataFrame,
    cols: Sequence[str],
    p_grid: Sequence[int],
    kappa_low: float = 0.94,
    kappa_high: float = 0.99,
    add_const: bool = False,
    top_candidates: int = 20,
    n_starts: int = 20,
    seed: int = 123,
    maxiter: int = 200,
    tol: float = 1e-8,
    include_corners: bool = True
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Búsqueda tipo funnel (2 etapas) para TVP-VAR:
    - ETAPA 1: Optimización continua de (kappa1, kappa2) en el cuadrado [kappa_low,kappa_high]^2
              minimizando AIC para cada p (con multi-start).
    - ETAPA 2: Diagnósticos profundos (Kalman completo + whitening + tests) sobre top_candidates por AIC.
    Selección final: prefiere los que pasan Hosking mean y Hosking ARCH (p>0.05) y luego menor AIC.

    Requiere que existan en tu entorno:
      - build_var_lagged_jax_ready(...)
      - batch_loglik_calculator(...)
      - run_tvp_kalman_jax(...)
      - whiten_by_Sigma(...)
      - hosking_multivariate_test(...)
      - joint_jarque_bera_test(...)
      - mardia_multivariate_test(...)
      - multivariate_lm_test(...)
    """

    # ------------------------- helpers -------------------------

    def _make_starts(rng: np.random.Generator) -> np.ndarray:
        """
        Genera puntos iniciales dentro del cuadrado.
        """
        starts = rng.uniform(kappa_low, kappa_high, size=(n_starts, 2))

        if include_corners:
            corners = np.array([
                [kappa_low,  kappa_low],
                [kappa_low,  kappa_high],
                [kappa_high, kappa_low],
                [kappa_high, kappa_high],
                [(kappa_low + kappa_high) / 2, (kappa_low + kappa_high) / 2],
            ], dtype=float)
            starts = np.vstack([corners, starts])

        return starts

    def _safe_float(x: float, big: float = 1e30) -> float:
        if not np.isfinite(x):
            return big
        return float(x)

    # ------------------------- stage 1 -------------------------

    results_list = []
    rng = np.random.default_rng(seed)

    max_p = max(p_grid)
    bounds = [(kappa_low, kappa_high), (kappa_low, kappa_high)]

    total_targets = len(p_grid) * (n_starts + (5 if include_corners else 0))
    print(f"🚀 ETAPA 1: Optimización continua en {len(p_grid)} valores de p "
          f"(~{total_targets} arranques totales).")

    for p in tqdm(p_grid, desc="Optimizando AIC (continuo)"):
        # 1) Matrices VAR (JAX-ready)
        Y, X, _, n, k = build_var_lagged_jax_ready(series, cols, p, add_const)

        # 2) Trim común a max_p
        trim_rows = max_p - p
        if trim_rows > 0:
            Y_eff, X_eff = Y[trim_rows:], X[trim_rows:]
        else:
            Y_eff, X_eff = Y, X

        T_eff = len(Y_eff)

        # 3) Priors OLS (NumPy)
        X_np, Y_np = np.array(X_eff), np.array(Y_eff)
        B_ols = np.linalg.lstsq(X_np, Y_np, rcond=None)[0]
        E_ols = Y_np - X_np @ B_ols
        Sigma0 = (E_ols.T @ E_ols) / T_eff
        a0 = B_ols.T.flatten()
        XtX_inv = np.linalg.pinv(X_np.T @ X_np)
        P0 = np.kron(Sigma0, XtX_inv) * 10.0

        # #params (igual que tu aproximación)
        n_params = (n * k) + (n * (n + 1) // 2)

        # 4) Preparar constantes JAX una vez
        Y_jax = jnp.array(Y_eff)
        X_jax = jnp.array(X_eff)
        Sigma0_jax = jnp.array(Sigma0)
        a0_jax = jnp.array(a0)
        P0_jax = jnp.array(P0)

        # 5) Objetivo AIC(k1,k2) (JAX) + gradiente (si es diferenciable)
        def aic_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
            k1, k2 = theta[0], theta[1]
            ll = batch_loglik_calculator(
                Y_jax, X_jax, Sigma0_jax, a0_jax, P0_jax,
                jnp.array([k1]), jnp.array([k2])
            )[0]
            aic = -2.0 * ll + 2.0 * n_params
            return aic

        aic_jit = jax.jit(aic_from_theta)
        grad_jit = jax.jit(jax.grad(aic_from_theta))

        def fun_np(theta_np: np.ndarray) -> float:
            val = float(aic_jit(jnp.array(theta_np, dtype=jnp.float64)))
            return _safe_float(val)

        def jac_np(theta_np: np.ndarray) -> np.ndarray:
            g = np.array(grad_jit(jnp.array(theta_np, dtype=jnp.float64)), dtype=float)
            g[~np.isfinite(g)] = 0.0
            return g

        # 6) Multi-start
        starts = _make_starts(rng)
        best_local = None

        for x0 in starts:
            try:
                res = minimize(
                    fun=fun_np,
                    x0=np.array(x0, dtype=float),
                    jac=jac_np,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": int(maxiter), "ftol": float(tol)}
                )

                if not res.success and not np.isfinite(res.fun):
                    continue

                k1_hat, k2_hat = res.x
                aic_hat = _safe_float(res.fun)

                # Guarda este óptimo local como una "combinación" candidata
                results_list.append({
                    "p": int(p),
                    "kappa1": float(k1_hat),
                    "kappa2": float(k2_hat),
                    "AIC": float(aic_hat),
                    "T_eff": int(T_eff),
                    "Prob(H_Mean)": np.nan,
                    "Prob(H_ARCH)": np.nan,
                    "Prob(Norm)": np.nan,
                    "Prob(M_Skew)": np.nan,
                    "Prob(M_Kurt)": np.nan,
                    "Prob(LM)": np.nan,
                    "Tested": False,
                    "Opt_Success": bool(res.success),
                    "Opt_Nit": int(res.nit) if hasattr(res, "nit") else np.nan
                })

                if (best_local is None) or (aic_hat < best_local["AIC"]):
                    best_local = {"AIC": aic_hat, "kappa1": float(k1_hat), "kappa2": float(k2_hat)}

            except Exception:
                # No abortar por un arranque malo
                continue

        # Si por alguna razón no se guardó nada, fuerza al menos el centro
        if best_local is None:
            center = np.array([(kappa_low + kappa_high) / 2, (kappa_low + kappa_high) / 2], dtype=float)
            aic_center = fun_np(center)
            results_list.append({
                "p": int(p),
                "kappa1": float(center[0]),
                "kappa2": float(center[1]),
                "AIC": float(aic_center),
                "T_eff": int(T_eff),
                "Prob(H_Mean)": np.nan,
                "Prob(H_ARCH)": np.nan,
                "Prob(Norm)": np.nan,
                "Prob(M_Skew)": np.nan,
                "Prob(M_Kurt)": np.nan,
                "Prob(LM)": np.nan,
                "Tested": False,
                "Opt_Success": False,
                "Opt_Nit": np.nan
            })

    df_results = pd.DataFrame(results_list)

    # ------------------------- stage 2 (igual que tu lógica) -------------------------

    candidates_idx = df_results.sort_values(by="AIC").head(top_candidates).index
    print(f"\n🔬 ETAPA 2: Kalman completo + diagnósticos para los mejores {top_candidates} candidatos...")

    for idx in tqdm(candidates_idx, desc="Diagnosticando"):
        row = df_results.loc[idx]
        p_curr = int(row["p"])
        k1_curr = float(row["kappa1"])
        k2_curr = float(row["kappa2"])

        # Rebuild
        Y, X, _, n, k = build_var_lagged_jax_ready(series, cols, p_curr, add_const)

        trim_rows = max_p - p_curr
        if trim_rows > 0:
            Y_eff, X_eff = Y[trim_rows:], X[trim_rows:]
        else:
            Y_eff, X_eff = Y, X

        X_np, Y_np = np.array(X_eff), np.array(Y_eff)
        B_ols = np.linalg.lstsq(X_np, Y_np, rcond=None)[0]
        E_ols = Y_np - X_np @ B_ols

        Sigma0 = (E_ols.T @ E_ols) / len(Y_np)
        a0 = B_ols.T.flatten()
        P0 = np.kron(Sigma0, np.linalg.pinv(X_np.T @ X_np)) * 10.0

        try:
            history = run_tvp_kalman_jax(
                jnp.array(Y_eff),
                jnp.array(X_eff),
                jnp.array(Sigma0),
                jnp.array(a0),
                jnp.array(P0),
                float(k1_curr),
                float(k2_curr)
            )

            v_t = np.array(history["innovations"])
            S_t = np.array(history["S_t"])

            W = whiten_by_Sigma(v_t, S_t, method="zca")

            _, p_mean, _ = hosking_multivariate_test(W, max_lag=10)
            _, p_arch, _ = hosking_multivariate_test(W**2, max_lag=10)
            p_norm = joint_jarque_bera_test(W)

            _, p_mardia_skew, _, p_mardia_kurt = mardia_multivariate_test(W)
            _, p_lm = multivariate_lm_test(W, max_lag=5)

            df_results.at[idx, "Prob(H_Mean)"] = p_mean
            df_results.at[idx, "Prob(H_ARCH)"] = p_arch
            df_results.at[idx, "Prob(Norm)"] = p_norm
            df_results.at[idx, "Prob(M_Skew)"] = p_mardia_skew
            df_results.at[idx, "Prob(M_Kurt)"] = p_mardia_kurt
            df_results.at[idx, "Prob(LM)"] = p_lm
            df_results.at[idx, "Tested"] = True

        except Exception as e:
            print(f"⚠️ Fallo en modelo p={p_curr}, k1={k1_curr:.6f}, k2={k2_curr:.6f}: {e}")

    # ------------------------- final selection (igual que tu lógica) -------------------------

    valid_models = df_results[
        (df_results["Tested"] == True) &
        (df_results["Prob(H_Mean)"] > 0.05) &
        (df_results["Prob(H_ARCH)"] > 0.05)
    ]

    if not valid_models.empty:
        best_model = valid_models.sort_values(by="AIC").iloc[0]
        print("\n✅ GANADOR ROBUSTO ENCONTRADO (pasó tests).")
    else:
        print("\n⚠️ Ningún candidato pasó los tests. Retorno el mejor AIC (sin filtro).")
        best_model = df_results.sort_values(by="AIC").iloc[0]

    print(f"🏆 Modelo Seleccionado: p={int(best_model['p'])}, "
          f"k1={best_model['kappa1']:.6f}, k2={best_model['kappa2']:.6f}")
    print(f"   AIC: {best_model['AIC']:.2f}")
    print(f"   P-Values: Mean={best_model['Prob(H_Mean)']:.4f}, "
          f"ARCH={best_model['Prob(H_ARCH)']:.4f}, Norm={best_model['Prob(Norm)']:.4f}")

    return best_model, df_results

###############################################################
####### UNI CONNECTEDNESS #######
###############################################################
# ====================================================================
# MODIFICACIÓN 3: tvpvar_connectedness (Genera la máscara y guarda sub-red)
# ====================================================================
def tvpvar_connectedness(
    series_returns: pd.DataFrame,
    chosen: List[str],
    sub_nodes: List[str] = None, 
    p: int = 1,
    delta: float = 0.99,  
    delta_2: float = 0.96, 
    H: int = 20,
    add_const: bool = False,
    return_covariances: bool = False  
) -> Dict[str, object]:
    """
    Estimate a TVP-VAR model and compute Diebold–Yilmaz-type connectedness measures.

    This function:
      1. Builds JAX-ready VAR(p) lagged matrices for the selected variables.
      2. Initializes the TVP-VAR state via OLS (for a0, P0, Sigma0).
      3. Runs the JAX Kalman filter `run_tvp_kalman_jax` with discount factors
         (delta, delta_2) to obtain time-varying coefficients and covariances.
      4. Computes generalized FEVDs and connectedness indices over time using
         `compute_connectedness_jax`.
      5. Packs all results into a convenient dictionary with pandas objects
         and NumPy/JAX arrays.

    Parameters
    ----------
    series_returns : pandas.DataFrame
        Input series (typically returns) with a time index and variables in columns.
    chosen : list of str
        Names of the columns (subset of series_returns.columns) to include in the TVP-VAR.
    p : int, optional
        VAR lag order. Default is 1.
    delta : float, optional
        State discount factor (kappa1) for the TVP-VAR, 0 < delta ≤ 1. Default is 0.99.
    delta_2 : float, optional
        Observation covariance discount factor (kappa2), 0 < delta_2 ≤ 1. Default is 0.96.
    H : int, optional
        Horizon for impulse responses and FEVD. Default is 20.
    add_const : bool, optional
        If True, include a constant in the VAR regressors. Default is False.

    Returns
    -------
    out : dict
        Dictionary with the following main keys:
        - "theta"     : DataFrame (T × (n*n)) with FEVD (MultiIndex columns: to/from).
        - "TCI"       : Series (T,) with Total Connectedness Index.
        - "TO"        : DataFrame (T × n) with TO connectedness indices.
        - "FROM"      : DataFrame (T × n) with FROM connectedness indices.
        - "NET"       : DataFrame (T × n) with NET connectedness indices.
        - "pairwise"  : DataFrame (T × (n*n)) with pairwise directional connectedness.
        - "fit"       : dict with filtered states, Sigma_t, S_t, K_t, loglik, etc.
        - "meta"      : dict with model meta-information (p, delta, H, chosen, ...).
    """
    # 0. Crear máscara para la sub-red
    if sub_nodes is None:
        sub_nodes = [] # Si no se provee, no calcula la sub-red
    sub_mask_np = np.array([1.0 if c in sub_nodes else 0.0 for c in chosen], dtype=np.float64)
    sub_mask_j = jnp.array(sub_mask_np)
    # 1. Build JAX-ready VAR lagged matrices (design) for the chosen variables

    Y, X, idx, n, k = build_var_lagged_jax_ready(series_returns, chosen, p, add_const)
    X_np, Y_np = np.array(X), np.array(Y)
    
    limit = min(len(Y_np), 250)
    X_init, Y_init = X_np[:limit], Y_np[:limit]

    B_ols = np.linalg.lstsq(X_init, Y_init, rcond=None)[0]
    E_ols = Y_init - X_init @ B_ols
    Sigma0 = (E_ols.T @ E_ols) / len(Y_init)
    a0 = B_ols.T.flatten()
    P0 = np.kron(Sigma0, np.linalg.pinv(X_init.T @ X_init)) * 10.0

    Sigma0_j, a0_j, P0_j = jnp.array(Sigma0), jnp.array(a0), jnp.array(P0)
    
    print(f"Estimando TVP-VAR (p={p}, k1={delta}, k2={delta_2})...")
    history = run_tvp_kalman_jax(Y, 
                                 X, 
                                 Sigma0_j, a0_j, 
                                 P0_j, 
                                 delta, 
                                 delta_2, 
                                 return_covariances=return_covariances)
    
    sigma0_expanded = jnp.expand_dims(Sigma0_j, axis=0)
    Sigma_t_shifted = jnp.concatenate([sigma0_expanded, history['Sigma_t'][:-1]], axis=0)
    
    print(f"Calculando Conectividad (H={H})...")
    # --- Añadimos el sub_mask_j y recuperamos tci_sub, net_sub ---
    theta, theta_g, tci, to_i, from_i, net_i, pdc_i, tci_sub, net_sub = compute_connectedness_jax(
        history['beta_filt'], Sigma_t_shifted, sub_mask_j, p, H, n, k
    )
    print(f"Terminado Conectividad (H={H})...")
    
    date_idx = pd.Index(idx, name="date")
    multi_cols = pd.MultiIndex.from_product([chosen, chosen], names=["to", "from"])
    
    theta_2d = np.asarray(theta).reshape(-1, n * n)
    pdc_2d = np.asarray(pdc_i).reshape(-1, n * n)
    theta_2d_g = np.asarray(theta_g).reshape(-1, n * n)

    Theta_df = pd.DataFrame(theta_2d, index=date_idx, columns=multi_cols, copy=False)
    Theta_g_df = pd.DataFrame(theta_2d_g, index=date_idx, columns=multi_cols, copy=False)
    PDC_df = pd.DataFrame(pdc_2d, index=date_idx, columns=multi_cols, copy=False)
    
    # Extraemos el NET de la subred solo para los nodos de interés (filtra ceros)
    NET_sub_df = pd.DataFrame(np.array(net_sub), index=date_idx, columns=chosen)
    if sub_nodes:
        NET_sub_df = NET_sub_df[sub_nodes]

    fit_dict = {
        "beta_filt": np.array(history['beta_filt']),
        "Sigma_t": np.array(history['Sigma_t']),
        "y_pred": np.array(history['y_pred']),
        "innovations": np.array(history['innovations']),
        "S_t": np.array(history['S_t']),
        "K_t":np.array(history['K_t']),
        "loglik": float(history['loglik'].sum()),
        "Sigma0": Sigma0,
        "meta": {"n": n, "k": k, "m": n*k, "kappa1": float(delta), "kappa2": float(delta_2), "p": int(p), "add_const": bool(add_const), "init": "OLS"}
    }

    if return_covariances:
        fit_dict["P_filt"] = np.array(history['P_filt'])
        fit_dict["P_pred"] = np.array(history['P_pred'])
    else:
        fit_dict["P_filt"], fit_dict["P_pred"] = None, None

    out = {
        "theta": Theta_df,
        "theta_g": Theta_g_df,  
        "TCI":   pd.Series(np.array(tci), index=date_idx, name="TCI"),
        "TO":    pd.DataFrame(np.array(to_i), index=date_idx, columns=chosen),
        "FROM":  pd.DataFrame(np.array(from_i), index=date_idx, columns=chosen),
        "NET":   pd.DataFrame(np.array(net_i), index=date_idx, columns=chosen),
        "pairwise": PDC_df,
        # --- NUEVOS RESULTADOS SUB-RED ---
        "TCI_sub": pd.Series(np.array(tci_sub), index=date_idx, name="TCI_sub"),
        "NET_sub": NET_sub_df,
        
        "fit": fit_dict,
        "meta": {"chosen": list(chosen), "sub_nodes": sub_nodes, "p": int(p), "delta": float(delta), "delta_2": float(delta_2), "H": int(H)}
    }
    return out

###############################################################
####### IC  JAX  CONNECTEDNESS #######
# ###############################################################
@partial(jax.jit, static_argnames=['p', 'H'])
def _theta_ps_t_aggregate_frozen(
    B_all: jnp.ndarray,
    Sigma_t: jnp.ndarray,
    t0: int,
    p: int,
    H: int
) -> jnp.ndarray:
    """
    Compute horizon-aggregated GFEVD numerators for a VAR with frozen parameters at time t0.

    Parameters
    ----------
    B_all : jax.Array, shape (T, n, k)
        Time-varying VAR coefficient matrices for each t.
        For each t, B_all[t] is an (n, k) matrix, where
        k = n * p (+ 1 if there is a constant, which is ignored here).
        The first n * p columns must encode the lag matrices stacked row-wise.
    Sigma_t : jax.Array, shape (T, n, n)
        Time-varying innovation covariance matrices for each t.
        Sigma_t[t] is the covariance matrix Sigma_t of the observation error at time t.
    t0 : int
        Index of the time point at which coefficients and covariance are frozen.
    p : int
        VAR lag order. Must be >= 1. Treated as static for JIT.
    H : int
        FEVD horizon (number of steps) for which to accumulate contributions.
        Must be >= 1. If H = 1, only the h = 0 contribution is returned.

    Returns
    -------
    Theta_total : jax.Array, shape (n, n)
        Horizon-aggregated (from h = 0 up to h = H - 1) GFEVD numerators:
        girf2_ij = Σ_{h=0}^{H-1} [ (phi_h Sigma_t)_ij / sqrt(Sigma_t,jj) ]^2,
        where phi_0 = I_n and phi_h is the h-step impulse response matrix
        computed from the frozen VAR coefficients at time t0.

    Notes
    -----
    - This function avoids allocating an (H, n, n) array of phi_h by
      accumulating Θ in the `lax.scan` loop.
    - Only the first n * p columns of B_all[t0] are used (no constant terms).
    """

    # ----------------------- 0) Basic shapes -----------------------
    T, n, k = B_all.shape  

    # ----------------------- 1) Freeze parameters -----------------
    # Freeze VAR coefficients and covariance at time t0.
    B_fixed = B_all[t0]        # (n, k)
    St = Sigma_t[t0]           # (n, n)

    # ----------------------- 2) Build A_3d (lag matrices) ---------
    # Extract only the dynamic part (lags); ignore constant if present.
    k_dynamic = n * p
    A_dynamic = B_fixed[:, :k_dynamic]  # (n, n*p)

    # Reshape into (p, n, n), where A_3d[ell] is A_ell, the ell-th lag matrix.
    A_3d = A_dynamic.reshape(n, p, n).transpose(1, 0, 2)  # (p, n, n)

    # ----------------------- 3) GFEVD constants & h = 0 term ------
    # inv_std_j = 1 / sqrt(Sigma_t,jj), with small floor for numerical stability.
    inv_std = lax.rsqrt(jnp.maximum(jnp.diag(St), 1e-12))  # (n,)

    # Identity for Φ_0 = I_n
    I = jnp.eye(n, dtype=B_all.dtype)

    # Θ contribution at h = 0:
    # For phi_0 = I, we have (I Sigma_t)_ij = Sigma_t,ij.
    # After scaling by inv_std (per column j), we square elementwise.
    girfsq_acc = (St * inv_std) ** 2  # (n, n)

    # ----------------------- 4) Initial state for scan ------------
    # G_window[0] = phi_0 = I, other entries are previous phi's (initially zero).
    G_hist0 = jnp.zeros((p, n, n), dtype=B_all.dtype)
    G_hist0 = G_hist0.at[0].set(I)

    # If H = 1, only h = 0 contribution is needed — no scan.
    def _only_h0_case() -> jnp.ndarray:
        return girfsq_acc

    def _scan_case() -> jnp.ndarray:
        """
        Inner function used only if H >= 2.
        """

        # Carry: (G_window, Theta_sum)
        def step(carry: Tuple[jnp.ndarray, jnp.ndarray], _):
            G_window, current_sum = carry

            # a) Compute next impulse response Φ_h:
            #    G_next = sum_{ell=1}^p A_ell * Φ_{h-ell}
            G_next = jnp.einsum('lij,ljk->ik', A_3d, G_window)

            # b) Compute GFEVD increment at horizon h:
            #    girf2_inc = [ (phi_h Sigma_t)_ij * inv_std_j ]^2
            girfsq_inc = (G_next @ St * inv_std) ** 2
            new_sum = current_sum + girfsq_inc

            # c) Roll the buffer: shift history and insert Φ_h at position 0
            G_window_new = jnp.roll(G_window, 1, axis=0).at[0].set(G_next)

            return (G_window_new, new_sum), None

        # Run scan for horizons h = 1, ..., H-1 (we already included h = 0).
        (_, Theta_total), _ = lax.scan(
            step,
            (G_hist0, girfsq_acc),
            xs=None,
            length=H - 1
        )
        return Theta_total

    # Branch on H to avoid negative length and keep shapes static.
    Theta_total = lax.cond(
        H <= 1,
        _only_h0_case,
        _scan_case
    )

    return Theta_total

@partial(jax.jit, static_argnames=['p', 'H'])
def _ps_theta_t_frozen(B_all: jnp.ndarray, Sigma_t: jnp.ndarray, p: int, H: int) -> jnp.ndarray:
    """Vectorized over time version of _theta_ps_t_aggregate_frozen."""
    T = B_all.shape[0]
    # vmap over t to compute girf2(t) for each time point
    return vmap(lambda t: _theta_ps_t_aggregate_frozen(B_all, Sigma_t, t, p, H))(jnp.arange(T))

def make_measure_fn_with_pairs_frozen(n: int, sub_mask: jnp.ndarray = None):
    Iu, Ju = jnp.triu_indices(n, k=1)

    @partial(jax.jit, static_argnames=['p', 'H'])
    def measure_fn(B_all: jnp.ndarray, Sigma_t: jnp.ndarray, p: int, H: int):
        
        girfsq = _ps_theta_t_frozen(B_all, Sigma_t, p, H)

        row_sums = jnp.sum(girfsq, axis=2, keepdims=True)            
        safe_row_sums = jnp.where(row_sums <= 1e-12, 1.0, row_sums) 
        theta = girfsq / safe_row_sums                               

        own_shares = jnp.diagonal(theta, axis1=1, axis2=2)  
        tr = jnp.sum(own_shares, axis=1)                  
        TCI = 100.0 * (1.0 - tr / float(n))               

        row_sums_phi = jnp.sum(theta, axis=2)               
        FROM = 100.0 * (row_sums_phi - own_shares)  / float(n)      

        col_sums_phi = jnp.sum(theta, axis=1)               
        TO = 100.0 * (col_sums_phi - own_shares) / float(n)         
        NET = TO - FROM                                   
        PDC = theta - jnp.swapaxes(theta, 1, 2)               

        # --- CÁLCULO SUB-RED ---
        if sub_mask is not None:
            girfsq_sub = girfsq * sub_mask[None, :, None] * sub_mask[None, None, :]
            row_sums_sub = jnp.sum(girfsq_sub, axis=2, keepdims=True)
            safe_row_sums_sub = jnp.where(row_sums_sub <= 1e-12, 1.0, row_sums_sub)
            theta_sub = girfsq_sub / safe_row_sums_sub
            
            own_shares_sub = jnp.diagonal(theta_sub, axis1=1, axis2=2)
            n_sub = jnp.maximum(jnp.sum(sub_mask), 1.0)
            
            col_sums_sub = jnp.sum(theta_sub, axis=1)
            row_sums_sub_1d = jnp.sum(theta_sub, axis=2)
            
            TO_sub = 100.0 * (col_sums_sub - own_shares_sub) / n_sub
            FROM_sub = 100.0 * (row_sums_sub_1d - own_shares_sub) / n_sub
            NET_sub = TO_sub - FROM_sub
            TCI_sub = jnp.sum(TO_sub, axis=1)
        else:
            TCI_sub = jnp.zeros_like(TCI)
            NET_sub = jnp.zeros_like(NET)

        # Devolvemos 8 objetos en lugar de 6
        return theta, TCI, TO, FROM, NET, PDC, TCI_sub, NET_sub

    return measure_fn, (Iu, Ju)

@jit
def _get_zca_matrix(A: jnp.ndarray, eps: float = 1e-10) -> jnp.ndarray:
    """
    Compute the ZCA whitening matrix Σ^{-1/2} for a covariance-like matrix A.

    This matrix is used to standardize residuals as:
        u_std = Σ^{-1/2} @ u_raw

    Steps:
      1. Symmetrize A numerically.
      2. Add a small jitter proportional to trace(A)/n for stability.
      3. Eigen-decompose A = Q diag(w) Q^T.
      4. Form Σ^{-1/2} = Q diag(w^{-1/2}) Q^T, with eigenvalues clipped away
         from zero.

    Parameters
    ----------
    A : jnp.ndarray of shape (n, n)
        Symmetric (or nearly symmetric) covariance-like matrix.
    eps : float, optional
        Small factor for jitter (stability in eigen-decomposition).

    Returns
    -------
    zca_whit : jnp.ndarray of shape (n, n)
        ZCA whitening matrix Σ^{-1/2}.
    """
    # Enforce numerical symmetry: A := (A + Aᵀ) / 2
    A = 0.5 * (A + A.T)

    # Get dimension n from the last axis of A
    n = A.shape[-1]

    # Compute a small jitter scaled by trace(A)/n for numerical stability
    jitter = eps * (jnp.trace(A) / jnp.asarray(n, dtype=A.dtype))

    # Eigen-decompose the regularized matrix A + jitter * I
    # w: eigenvalues, Q: eigenvectors
    w, Q = jnp.linalg.eigh(A + jitter * jnp.eye(n, dtype=A.dtype))
    
    # Compute inverse square root of eigenvalues, clipped away from zero
    w_inv_sqrt = 1.0 / jnp.sqrt(jnp.maximum(w, 1e-12))
    
    # Reconstruct ZCA whitening matrix: Q diag(w^{-1/2}) Qᵀ
    # Implemented as Q @ (diag(w_inv_sqrt) @ Qᵀ) using broadcasting
    return Q @ (w_inv_sqrt[:, None] * Q.T)

@jit
def _get_zca_coloring_matrix(A: jnp.ndarray, eps: float = 1e-10) -> jnp.ndarray:
    """
    Compute the ZCA coloring matrix Σ^{1/2} for a covariance-like matrix A.

    This matrix is used to generate correlated shocks from i.i.d. ones:
        v_sim = Σ^{1/2} @ w_iid

    Steps:
      1. Symmetrize A numerically.
      2. Add a small jitter proportional to trace(A)/n for stability.
      3. Eigen-decompose A = Q diag(w) Q^T.
      4. Form Σ^{1/2} = Q diag(w^{1/2}) Q^T, with eigenvalues clipped away
         from zero.

    Parameters
    ----------
    A : jnp.ndarray of shape (n, n)
        Symmetric (or nearly symmetric) covariance-like matrix.
    eps : float, optional
        Small factor for jitter (stability in eigen-decomposition).

    Returns
    -------
    zca_color : jnp.ndarray of shape (n, n)
        ZCA coloring matrix Σ^{1/2}.
    """
    # Enforce numerical symmetry: A := (A + Aᵀ) / 2
    A = 0.5 * (A + A.T)

    # Get dimension n from the last axis of A
    n = A.shape[-1]

    # Compute a small jitter scaled by trace(A)/n for numerical stability
    jitter = eps * (jnp.trace(A) / jnp.asarray(n, dtype=A.dtype))

    # Eigen-decompose the regularized matrix A + jitter * I
    # w: eigenvalues, Q: eigenvectors
    w, Q = jnp.linalg.eigh(A + jitter * jnp.eye(n, dtype=A.dtype))
    
    # Compute square root of eigenvalues, clipped away from zero
    w_sqrt = jnp.sqrt(jnp.maximum(w, 1e-12))
    
    # Reconstruct ZCA coloring matrix: Q diag(w^{1/2}) Qᵀ
    # Implemented as Q @ (diag(w_sqrt) @ Qᵀ) using broadcasting
    return Q @ (w_sqrt[:, None] * Q.T)

# Vectorized versions over batch dimension T
_vzca_matrix = jit(vmap(_get_zca_matrix))
_vzca_coloring_matrix = jit(vmap(_get_zca_coloring_matrix))

@jit
def _standardize_residuals_by_Sigma_jax(u: jnp.ndarray, Sigma_t: jnp.ndarray) -> jnp.ndarray:
    """
    Whiten residuals using ZCA (Spectral Decomposition).
    W_t = Sigma_t^{-1/2} @ u_t
    """
    # 1. Obtener matrices de blanqueo para todo T
    W_mat = _vzca_matrix(Sigma_t)  # (T, n, n)
    
    # 2. Aplicar transformación: W_t @ u_t
    # einsum 'tij,tj->ti': Para cada t, multiplica matriz (n,n) por vector (n)
    W = jnp.einsum('tij,tj->ti', W_mat, u)
    
    # 3. Centrar
    return W - jnp.mean(W, axis=0, keepdims=True)

@partial(jax.jit, static_argnums=(2,))
def _mbb_resample_W_batched(
    keys: jnp.ndarray, 
    W: jnp.ndarray, 
    block_len: int
) -> jnp.ndarray:
    """
    Moving Block Bootstrap (MBB) for whitened residuals, batched over many replicas.

    This function produces B bootstrap samples W*_b(t) by:
        1. Drawing random starting points for MBB blocks,
        2. Wrapping indices modulo T to preserve continuity,
        3. Concatenating blocks until length T is reached,
        4. Returning W indexed by those resampled locations.

    The function is fully vectorized over `keys` using `vmap`, and JIT-compiled
    so that block_len is treated as a static argument (necessary for JAX shape tracing).

    Parameters
    ----------
    keys : jnp.ndarray of shape (B, 2)
        Array of PRNGKeys (one per bootstrap replicate).
    W : jnp.ndarray of shape (T, n)
        Whitened residuals from the TVP-VAR fit. T is time length, n is dimension.
    block_len : int
        Length of each bootstrap block (integer >= 1).

    Returns
    -------
    Wb : jnp.ndarray of shape (B, T, n)
        B independent MBB bootstrap samples of the whitened residuals,
        each of length T and dimension n.
    """
    # Number of time points
    T = W.shape[0]

    # Cast block length to integer
    b = int(block_len)

    # Number of blocks needed to cover T
    n_blocks = (T + b - 1) // b

    # Pre-compute [0, 1, ..., b-1]
    arange_b = jnp.arange(b)
    
    max_start_index = T - b + 1
    # ---------------------------------------------------------
    # Function that generates a single bootstrap series W*_b(t)
    # ---------------------------------------------------------
    def one(key):
        # Sample random starting indices for each block
        starts = jax.random.randint(key, shape=(n_blocks,), minval=0, maxval= max_start_index)

        # For each block, create b consecutive 
        idx = starts[:, None] + arange_b[None, :] 

        # Flatten and truncate to length T exactly
        idx = idx.reshape(-1)[:T]

        # Return the resampled residual series
        return W[idx]  # shape (T, n)

    # Vectorize "one" over all keys → shape (B, T, n)
    return vmap(one)(keys)

def probe(tensor, step_name):
    """
    Verifica si hay NaNs o Infs en el tensor.
    Si encuentra alguno, imprime un mensaje de alerta.
    """
    # Checa si hay algún NaN o Infinito en todo el array
    is_bad = jnp.isnan(tensor).any() | jnp.isinf(tensor).any()
    
    def alert():
        # Esto se imprimirá en tu terminal/notebook si JAX detecta error
        jax.debug.print("🚨 ERROR DETECTADO EN: {name} | Shape: {shp}", 
                        name=step_name, shp=tensor.shape)
        
        # Opcional: Si quieres ver el primer valor NaN (avanzado)
        # jax.debug.print("   Valores (min, max): ({mn}, {mx})", 
        #                 mn=jnp.nanmin(tensor), mx=jnp.nanmax(tensor))

    # Ejecuta la alerta solo si is_bad es True
    lax.cond(is_bad, alert, lambda: None)
    
    return tensor

@partial(jax.jit, static_argnums=(6, 7))
def _kalman_loop_core_jax(
    Y: jnp.ndarray,
    X: jnp.ndarray,
    kappa1: float,
    kappa2: float,
    Sigma0: jnp.ndarray,
    P0_scale: float,
    p: int,
    add_const: bool,
    early_Q_scale: float
):
    """
    Single-path TVP-VAR Kalman filter core (JAX).
    """
    # Extract time length T and dimension n from Y
    T, n = Y.shape
    # Number of regressors per equation from X
    k = X.shape[1]
    # Total state dimension m = n * k
    m = n * k
    # Initialize state mean a_0 at zero
    a_prev = jnp.zeros((m,), Y.dtype)
    # Initialize state covariance P_0 as scaled identity
    P_prev = P0_scale * jnp.eye(m, m, dtype=Y.dtype)

    # Build an initial “factor” G_0 from the first observation
    G0 = early_Q_scale * (Y[0][:, None] @ Y[0][None, :])
    # Initial Σ_0 via convex combination of Sigma0 and G0, symmetrized
    Sigma_prev = 0.5 * (
        kappa2 * Sigma0 + (1.0 - kappa2) * G0
        + (kappa2 * Sigma0 + (1.0 - kappa2) * G0).T
    )

    # One-step Kalman recursion for lax.scan
    def one(carry, inp):
        # Unpack previous state and covariance
        a_prev, P_prev, Sigma_prev = carry
        # Current regressors and observation
        x_t, y_t = inp

        # Reshape state vector into coefficient matrix B_{t-1}
        B_prev = a_prev.reshape(n, k)
        # One-step-ahead prediction ŷ_t = B_{t-1} x_t
        y_hat = B_prev @ x_t

        #Reshape P_{t-1} into 4D (n, k, n, k) for tensor contractions
        P_rs = P_prev.reshape(n, k, n, k)

        #Compute PHᵀ via contraction: (n,k,n,k) with x_t → (m,n)
        PHt = jnp.einsum('iakb,b->iak', P_rs, x_t).reshape(m, n)

        #Innovation covariance S_t = H P Hᵀ + Σ_{t-1}
        #Here H_t depends on x_t, so we use another contraction
        S = jnp.einsum('iajb,a,b->ij', P_rs, x_t, x_t) + Sigma_prev

        # Ganancia de Kalman: K_t = P H^T S^{-1}
        K_t = jnp.linalg.solve(S.T, PHt.T).T 
        
        # Innovation v_t = y_t − ŷ_t
        v = y_t - y_hat

        # Posterior state a_t = a_{t-1} + K_t v_t
        a_post = a_prev + K_t @ v

        # ---------------------------------------------------------------
        # FIN MODIFICACIÓN
        # ---------------------------------------------------------------

        # Posterior P_t with symmetrization
        P_post_raw = P_prev - K_t @ S @ K_t.T
        P_post = 0.5 * (P_post_raw + P_post_raw.T)

        # Reshape posterior state to B_t
        B_post = a_post.reshape(n, k)

        # Compute prediction based only on lag terms (exclude constant from residual)
        yhat_lags = B_post @ x_t
        # Residuals relative to lag-only component
        e = y_t - yhat_lags

        # Outer product G_f,t = e eᵀ
        Gf = jnp.outer(e, e)

        # Update Σ_t via EWMA mixture of Σ_{t-1} and G_f,t, then symmetrize
        Sigma_mix = kappa2 * Sigma_prev + (1.0 - kappa2) * Gf
        Sigma_post = 0.5 * (Sigma_mix + Sigma_mix.T)

        # Next carry: updated mean, discounted covariance, and Σ_t
        return (a_post, P_post / kappa1, Sigma_post), (a_post, y_hat, Sigma_post)

    # Run the filter over all time steps with lax.scan
    (_, _, _), (a_hist, y_pred, Sigma_hist) = lax.scan(
        one,
        (a_prev, P_prev, Sigma_prev),
        (X, Y)
    )

    # Return filtered states, predictions, and Σ_t history
    return a_hist, y_pred, Sigma_hist  # a_hist: (T,m), y_pred: (T,n), Sigma_hist: (T,n,n)

# VMAP Kalman over B
@partial(jax.jit, static_argnums=(6, 7))
def _kalman_batched(Yb: jnp.ndarray, Xb: jnp.ndarray,
                    kappa1: float, kappa2: float, Sigma0: jnp.ndarray, P0_scale: float,
                    p: int, add_const: bool, early_Q: float):
    return vmap(_kalman_loop_core_jax,
                in_axes=(0, 0, None, None, None, None, None, None, None),
                out_axes=(0, 0, 0))(Yb, Xb, kappa1, kappa2, Sigma0, P0_scale, p, add_const, early_Q)

@jit
def _quantile_from_sorted(xs: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """
    Compute quantiles for data ALREADY SORTED along axis 0.

    Assumes:
      - xs has shape (B, T, chunk),
      - xs is sorted increasing along axis 0,
      - q has shape (T, chunk) with values in [0,1].

    Uses linear interpolation between adjacent order statistics.
    """
    # Number of bootstrap draws (order statistics) along axis 0
    B = xs.shape[0]

    # Ensure q is in [0, 1]
    q = jnp.clip(q, 0.0, 1.0)
    
    # Desired position in the sorted sample: q * (B - 1)
    pos = q * (B - 1)

    # Lower index (floor) and upper index (ceil-like, but clipped)
    i0 = jnp.floor(pos).astype(jnp.int32)
    i1 = jnp.clip(i0 + 1, 0, B - 1)

    # Fractional distance between i0 and i1
    w = (pos - i0).astype(xs.dtype)
    
    # Expand indices to rank 3 so they match xs shape rank
    i0_exp = jnp.expand_dims(i0, axis=0)  # shape (1, T, chunk)
    i1_exp = jnp.expand_dims(i1, axis=0)  # shape (1, T, chunk)

    # Gather lower and upper order stats along axis 0, then squeeze back
    x0 = jnp.take_along_axis(xs, i0_exp, axis=0).squeeze(0)
    x1 = jnp.take_along_axis(xs, i1_exp, axis=0).squeeze(0)
    
    # Linear interpolation between x0 and x1
    return (1.0 - w) * x0 + w * x1

@jit
def _percentile_kernel_chunk(
    theta_boot_chunk: jnp.ndarray,
    alpha: float
):
    """
    Calcula intervalos de percentiles para un chunk de variables.
    """
    # 1. Ordenar los sorteos de bootstrap a lo largo del eje B
    xs = jnp.sort(theta_boot_chunk, axis=0)  # shape (B, T, chunk)
    
    # 2. Definir cuantiles nominales (ej. 0.025 y 0.975 para 95%)
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0
    
    # 3. Interpolar sobre la matriz de bootstrap ordenada
    # Creamos un array de la misma forma que el "espacio de features" para la función de cuantiles
    dummy_shape = (theta_boot_chunk.shape[1], theta_boot_chunk.shape[2])
    
    perc_lo = _quantile_from_sorted(xs, jnp.full(dummy_shape, q_lo))
    perc_hi = _quantile_from_sorted(xs, jnp.full(dummy_shape, q_hi))
    
    return perc_lo, perc_hi

def compute_percentile_intervals(
    theta_boot,
    alpha: float = 0.05,
    chunk_size: int = 500,
    desc: str = "Metric"
):
    """
    Calcula intervalos de confianza por percentiles para métricas de cualquier forma.
    """
    shape_boot = theta_boot.shape  # (B, T, *extra_dims)
    B, T = shape_boot[0], shape_boot[1]
    
    extra_dims = shape_boot[2:]
    total_features = 1
    for d in extra_dims:
        total_features *= d
        
    flat_boot = theta_boot.reshape(B, T, total_features)
    res_lo, res_hi = [], []
    
    with tqdm(total=total_features, desc=f"Percentiles {desc}", unit='feat') as pbar:
        for i in range(0, total_features, chunk_size):
            end = min(i + chunk_size, total_features)
            chunk_b = flat_boot[:, :, i:end] 
            
            # --- KERNEL JAX ---
            p_lo, p_hi = _percentile_kernel_chunk(chunk_b, alpha)
            p_hi.block_until_ready()
            
            res_lo.append(np.array(p_lo))
            res_hi.append(np.array(p_hi))
            
            pbar.update(end - i)

    def restore_shape(res_list):
        if not res_list: return np.array([])
        return np.concatenate(res_list, axis=1).reshape((T,) + extra_dims)

    return restore_shape(res_lo), restore_shape(res_hi)

@partial(jax.jit, static_argnums=(3, 4))
def _simulate_tv_ar_process_core(
    B_all: jnp.ndarray,    # (T, n, k) Trayectoria FILTRADA (t|t)
    v_colored: jnp.ndarray,# (T, n) Shocks
    y_init: jnp.ndarray,   # (p, n) Lags iniciales
    p: int,
    add_const: bool
):
    """
    Simula TVP-VAR usando coeficientes predictivos (B_{t|t-1}).
    
    Lógica de Rezagos:
    - Crea B_pred desplazando B_all un paso hacia adelante.
    - B_pred[t] será igual a B_all[t-1].
    - B_pred[0] será igual a B_all[0] (Condición inicial).
    """
    
    # -----------------------------------------------------------
    # 1. GENERACIÓN DE B_{t|t-1} (Predictive Coeffs)
    # -----------------------------------------------------------
    # Tomamos el primer elemento (t=0) y lo mantenemos
    B_t0 = B_all[:1]        # Shape (1, n, k)
    # Tomamos hasta el penúltimo elemento (t=0 ... T-2)
    B_prev = B_all[:-1]     # Shape (T-1, n, k)
    
    # Concatenamos: [B_0, B_0, B_1, ..., B_{T-2}]
    # Ahora el índice t contiene la información de t-1
    B_predictive = jnp.concatenate([B_t0, B_prev], axis=0)

    # -----------------------------------------------------------
    # 2. PREPARACIÓN DEL SCAN
    # -----------------------------------------------------------
    T, n, k = B_all.shape
    k_dynamic = n * p

    def step(y_hist, inputs):
        # inputs ahora trae B_{t|t-1} y el shock v_t
        B_t_pred, v_t = inputs
        
        # Construir regresor x_t (lags)
        x_t = jnp.zeros((k,), dtype=B_all.dtype)
        x_t = x_t.at[:k_dynamic].set(y_hist.reshape(k_dynamic))
        
        if add_const:
            x_t = x_t.at[-1].set(1.0)

        # Ecuación de estado: y_t = B_{t|t-1} * x_t + v_t
        y_sim = B_t_pred @ x_t + v_t

        # Actualizar buffer
        if p > 1:
            y_hist_new = jnp.roll(y_hist, 1, axis=0).at[0].set(y_sim)
        else:
            y_hist_new = y_hist.at[0].set(y_sim)

        return y_hist_new, (y_sim, x_t)

    # El scan itera sobre B_predictive en lugar de B_all original
    _, (Yb, Xb) = lax.scan(step, y_init, (B_predictive, v_colored))
    
    return Yb, Xb

# Vectorización sobre el eje de Bootstrap (Axis 0 de v_colored_batch)
# B_all es constante para todos los replicas (None), y_init también (None).
_simulate_tv_ar_batched = vmap(
    _simulate_tv_ar_process_core,
    in_axes=(None, 0, None, None, None)
)
    
import math
import os
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

def bootstrap_connectedness_CI_jax_fast(
    key: jax.random.PRNGKey,
    series_centered_np: np.ndarray,
    chosen_cols: list[str],
    Y: jnp.ndarray,
    X: jnp.ndarray,
    idx: pd.Index,
    base_fit: dict,
    p: int,
    H: int,
    *,
    sub_nodes: list[str] = None, 
    alpha: float = 0.05,
    B: int = 1000,
    block_len: int = 20,
    store_pairs: bool = True,
    measure_fn=None,
    batch_B: int = 128
) -> dict:
    # Extraemos el número de variables (n) y la longitud temporal (T_base) del modelo base
    n = int(Y.shape[1])
    T_base = base_fit["beta_filt"].shape[0]
    # -------------------------------------------------------------------------
    # 1. PREPARACIÓN DEL ALMACENAMIENTO FUERA DE MEMORIA (OUT-OF-CORE)
    # -------------------------------------------------------------------------
    # Creamos un directorio temporal seguro en el sistema operativo para guardar los datos
    tmp_dir = tempfile.mkdtemp()
    # Definimos las rutas físicas de los archivos para los índices globales
    path_tci = os.path.join(tmp_dir, 'tci_boot.dat')
    path_net = os.path.join(tmp_dir, 'net_boot.dat')
    # --- Archivos de Memoria para la Sub-red ---
    path_tci_sub = os.path.join(tmp_dir, 'tci_sub_boot.dat')
    path_net_sub = os.path.join(tmp_dir, 'net_sub_boot.dat')
    # Creamos mapas de memoria (memmaps). Actúan como arrays de NumPy, pero los
    # datos viven en el disco duro. Solo se carga en RAM el pedazo que se está leyendo/escribiendo.
    TCI_boot = np.memmap(path_tci, dtype='float64', mode='w+', shape=(B, T_base))
    NET_boot = np.memmap(path_net, dtype='float64', mode='w+', shape=(B, T_base, n))
    
    TCI_sub_boot = np.memmap(path_tci_sub, dtype='float64', mode='w+', shape=(B, T_base))
    NET_sub_boot = np.memmap(path_net_sub, dtype='float64', mode='w+', shape=(B, T_base, n))
    # Si requerimos las matrices direccionales par a par (PDC), creamos su memmap (es el más pesado)
    PDC_boot = None
    if store_pairs:
        path_pdc = os.path.join(tmp_dir, 'pdc_boot.dat')
        PDC_boot = np.memmap(path_pdc, dtype='float64', mode='w+', shape=(B, T_base, n, n))
    # -------------------------------------------------------------------------
    # 2. EXTRACCIÓN Y PREPARACIÓN DE PARÁMETROS DEL MODELO BASE
    # ----------------------------------------------------
    # Recuperamos los coeficientes filtrados (estado a_t) y deducimos el número de regresores (k)
    beta_filt = np.asarray(base_fit["beta_filt"])
    k_base = beta_filt.shape[1] // n
    # Reformateamos los coeficientes a tensores 3D (T, n, k) listos para JAX
    B_all_base = jnp.asarray(beta_filt.reshape(T_base, n, k_base), dtype=jnp.float64)
    
    # Recuperamos la varianza de las innovaciones S_t, la covarianza inicial Sigma0 y los errores u_hat
    S_t_base = jnp.asarray(base_fit["S_t"], dtype=jnp.float64)
    Sigma0 = jnp.asarray(base_fit["Sigma0"], dtype=jnp.float64)
    u_hat = jnp.asarray(base_fit["innovations"], dtype=jnp.float64)
    # -------------------------------------------------------------------------
    # 3. BLANQUEO Y COLORACIÓN (ZCA) PARA EL BOOTSTRAP
    # -------------------------------------------------------------
    # Blanqueamos los errores empíricos (les quitamos la correlación cruzada contemporánea)
    W = _standardize_residuals_by_Sigma_jax(u_hat, S_t_base)
    # Pre-calculamos las matrices de coloración para volver a inyectar la correlación en el futuro
    Coloring_Matrices = _vzca_coloring_matrix(S_t_base)
    # Extraemos los rezagos iniciales reales (los primeros 'p' días) para arrancar la simulación
    y_init = jnp.asarray(series_centered_np[:p, :n][::-1, :], dtype=jnp.float64)
    # Extraemos los hiperparámetros estáticos utilizados en el modelo original
    meta = base_fit["meta"]
    add_const = meta.get("add_const", False)
    kappa1 = meta.get("kappa1", 0.99)
    kappa2 = meta.get("kappa2", 0.99)
    early_Q = meta.get("early_Q_scale", 0.1)
    # -------------------------------------------------------------------------
    # 4. CONFIGURACIÓN DEL CÁLCULO DE CONECTIVIDAD
    # ------------------------------------------------
    # Construimos la máscara binaria para aislar la sub-red regional (ej. USA-MX)
    if sub_nodes is None:
        sub_nodes = []
    sub_mask_j = jnp.array([1.0 if c in sub_nodes else 0.0 for c in chosen_cols], dtype=jnp.float64)
    # Inyectamos esta máscara en la función constructora que calculará los índices DY
    measure_fn, _ = make_measure_fn_with_pairs_frozen(n, sub_mask_j)
    # -------------------------------------------------------------------------
    # 5. EJECUCIÓN DEL BOOTSTRAP POR LOTES (BATCHING)
    # --------------------------------------------------
    # Calculamos cuántos lotes necesitamos para alcanzar B iteraciones sin saturar la GPU/RAM
    n_batches = math.ceil(B / batch_B)
    # Generamos semillas aleatorias independientes para cada lote garantizando reproducibilidad
    subkeys = jax.random.split(key, n_batches)
    # Función envoltorio que extrae exactamente los 5 objetos que queremos de la función de medida
    def per_rep(Bi, Si):
        out = measure_fn(Bi, Si, int(p), int(H))
        # Extraemos: TCI_global(1), NET_global(4), PDC(5), TCI_sub(6), NET_sub(7)
        return out[1], out[4], out[5], out[6], out[7]
    # Iteramos sobre cada lote de réplicas
    for bi in tqdm(range(n_batches), desc="Batching to Disk"):
        # Calculamos los índices de inicio y fin para este lote (ej. del 0 al 128)
        b0 = bi * batch_B
        b1 = min(B, b0 + batch_B)
        Bb = b1 - b0
        # A. Remuestreamos los residuales blanqueados usando Moving Block Bootstrap (MBB) en paralelo
        Wb = _mbb_resample_W_batched(jax.random.split(subkeys[bi], Bb), W, int(block_len))
        # B. Volvemos a "colorear" los residuales aplicando la covarianza histórica
        v_boot = vmap(vmap(lambda C, w: C @ w, in_axes=(0, 0)), in_axes=(None, 0))(Coloring_Matrices, Wb)
        # C. Simulamos 'Bb' nuevas trayectorias completas del TVP-VAR usando los choques remuestreados
        Yb, Xb = _simulate_tv_ar_batched(B_all_base, v_boot, y_init, int(p), bool(add_const))
        # D. Re-entrenamos el Filtro de Kalman sobre las trayectorias simuladas para obtener nuevos parámetros
        beta_b, _, Sigma_b = _kalman_batched(Yb, Xb, kappa1, kappa2, Sigma0, 1e4, int(p), add_const, early_Q)
        
        # E. Calculamos los índices de conectividad para las 'Bb' trayectorias simuladas en paralelo usando vmap
        TCI_jb, NET_jb, PDC_jb, TCI_sub_jb, NET_sub_jb = vmap(per_rep, in_axes=(0, 0))(beta_b.reshape(Bb, T_base, n, k_base), Sigma_b)
        # F. Guardamos los resultados del lote directamente en los archivos memmap (¡en el disco duro!)
        TCI_boot[b0:b1] = np.asarray(TCI_jb)
        NET_boot[b0:b1] = np.asarray(NET_jb)
        TCI_sub_boot[b0:b1] = np.asarray(TCI_sub_jb)
        NET_sub_boot[b0:b1] = np.asarray(NET_sub_jb)
        
        if store_pairs:
            PDC_boot[b0:b1] = np.asarray(PDC_jb)
        # Forzamos la escritura física en disco para liberar memoria RAM
        TCI_boot.flush(); NET_boot.flush()
        TCI_sub_boot.flush(); NET_sub_boot.flush()
        if store_pairs: 
            PDC_boot.flush()
    # -------------------------------------------------------------------------
    # 6. CÁLCULO DE INTERVALOS DE CONFIANZA
    # -------------------------------------------
    print("\nCalculando Intervalos desde Disco...")
    # Procesamos los memmaps por "chunks" (trozos) para calcular los percentiles reales
    TCI_lo, TCI_hi = compute_percentile_intervals(TCI_boot, alpha, desc="TCI")
    NET_lo, NET_hi = compute_percentile_intervals(NET_boot, alpha, desc="NET")
    
    # --- Intervalos Sub-red ---
    TCI_sub_lo, TCI_sub_hi = compute_percentile_intervals(TCI_sub_boot, alpha, desc="TCI_sub")
    NET_sub_lo, NET_sub_hi = compute_percentile_intervals(NET_sub_boot, alpha, desc="NET_sub")
    
    PDC_ci = None
    if store_pairs:
        PDC_lo, PDC_hi = compute_percentile_intervals(PDC_boot, alpha, chunk_size=16, desc="PDC")
        PDC_ci = (PDC_lo, PDC_hi)
    # Borramos las referencias a los memmap de la sesión de Python para cerrar los archivos
    del TCI_boot, NET_boot, TCI_sub_boot, NET_sub_boot
    if store_pairs:
        del PDC_boot

    # Convertimos los numpy arrays de NET a diccionarios limpios filtrando nodos no relevantes
    NET_sub_lo_dict = {node: NET_sub_lo[:, i] for i, node in enumerate(chosen_cols) if node in sub_nodes}
    NET_sub_hi_dict = {node: NET_sub_hi[:, i] for i, node in enumerate(chosen_cols) if node in sub_nodes}

    return {
        "index": idx,
        "nodes": chosen_cols,
        "TCI": (TCI_lo, TCI_hi),
        "NET": (NET_lo, NET_hi),
        "PDC": PDC_ci,
        # --- SALIDA SUB-RED ---
        "TCI_sub": (TCI_sub_lo, TCI_sub_hi),
        "NET_sub": (NET_sub_lo_dict, NET_sub_hi_dict),
        "tmp_path": tmp_dir
    }