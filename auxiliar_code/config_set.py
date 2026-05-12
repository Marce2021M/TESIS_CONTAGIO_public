# IMPORT LIBRARIES AND SET ENVIRONMENT VARIABLES
# For future annotations (type hints)
from __future__ import annotations

# Import OS to manipulate environment variables and system-level settings

import os
###############################################
###### Environment configuration for numerical libraries
################################################
# Set default max OpenMP threads to 1 to avoid nested parallel computation storms
#os.environ.setdefault("OMP_NUM_THREADS", "1")
# Limit OpenBLAS threads to 1 for reproducible and stable performance
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
# Limit MKL threads to 1 for reproducibility and to avoid oversubscription
#os.environ.setdefault("MKL_NUM_THREADS", "1") BORRAR
# Limit NUMEXPR threads to 1 for controlled parallelism
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Limit Apple Accelerate (vecLib) threads to 1 on macOS
#os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS Accelerate
# Limit maximum nested OpenMP parallel levels
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "4")    # preferred over OMP_NESTED

# Suppress some Intel/LLVM OpenMP warnings (useful on macOS)
#os.environ.setdefault("KMP_WARNINGS", "0") BORRAR

# Suppress warnings globally in the script (optional, can be commented out)
import warnings
###############################################
###### Library imports for numerical computing and parallelism
################################################
# Import combinations generator from itertools
from itertools import combinations
from tqdm.auto import tqdm
import itertools
# Import math module for basic mathematical functions
import math
from math import erf, sqrt, ceil
# Import NumPy for numerical array operations
import numpy as np
# Import pandas for tabular data structures and manipulation
import pandas as pd
# Import cartesian product generator from itertools
from itertools import product
# Import Numba JIT compiler tools for potential performance speed-ups
from numba import njit, prange
# Import joblib for simple parallel processing (embarrassingly parallel tasks)
from joblib import Parallel, delayed
# Import partial function application utility
from functools import partial
# for type hinting
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Any
# Import function to minimize
from scipy.optimize import minimize
###############################################
###### Library for plotting and results
################################################
# Import matplotlib for plotting and visualizations
import matplotlib.pyplot as plt
# Import openpyxl to read/write Excel files (.xlsx)
import openpyxl
# Import ACF/PACF plotting utilities from statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Import seaborn for advanced statistical data visualization
import seaborn as sns
# Import matplotlib dates module for handling date formatting in plots
import matplotlib.dates as mdates
# Import FuncFormatter for custom tick formatting in plots
from matplotlib.ticker import MaxNLocator, FuncFormatter
# Import NetworkX for network analysis and graph-based computations
import networkx as nx
# For image and GIF handling
import imageio.v2 as imageio
# Import Line2D and Patch for custom legend elements in plots
from matplotlib.lines import Line2D
# Import Patch for custom legend elements in plots
from matplotlib.patches import Patch
# Import colors for custom color handling in plots
import matplotlib.colors as mcolors
# Import matplotlib.patches for advanced patch handling in plots
import matplotlib.patches as mpatches
# Import Line2D for custom line styles in legends
import matplotlib.lines as mlines
# Import patheffects for adding effects to plot elements (e.g., outlines)
import matplotlib.patheffects as pe

###############################################
###### Statistical and time-series modeling libraries
################################################
# Import Kolmogorov–Smirnov test from SciPy
from scipy.stats import kstest
from scipy import stats
# Import additional distributions from arch for univariate time-series models
from arch.univariate import SkewStudent, GeneralizedError
# Import Student's t-distribution, normal distribution and QQ-plot helper
from scipy.stats import t as student_t, norm, probplot, rankdata
# Import ARIMA model class from statsmodels for time-series modeling
from statsmodels.tsa.arima.model import ARIMA
# Import Ljung–Box test and ARCH test for residual diagnostics
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
# Import Jarque–Bera normality test for residuals
from statsmodels.stats.stattools import jarque_bera
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller, kpss, acf
# Import generic ARCH model constructor
from arch.univariate import arch_model
# Import standard Student-t and Normal distributions from arch (safe choices)
from arch.univariate import StudentsT, Normal  # dists seguras

###############################################
###### JAX configuration and imports
################################################

# Force JAX to run only on CPU (useful if GPU/TPU not desired or problematic)
os.environ["JAX_PLATFORMS"] = "cpu"
# Enable float64 precision in JAX (disabled by default for performance)
os.environ["JAX_ENABLE_X64"] = "true"
# Set highest precision for matrix multiplications in JAX
#os.environ.setdefault("XLA_FLAGS", f"--xla_cpu_multi_thread_eigen=true --xla_cpu_multi_thread_eigen_threads={3}")
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true "  # Usar multihilo en operaciones matriciales
    "intro_xla_cpu_use_run_time_thread_pool=true" # Gestión inteligente de hilos
)
# Evita que JAX intente reservar toda la RAM de golpe (causa común de crashes)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


# Import JAX core library and its NumPy-compatible API
import jax
import jax, jax.numpy as jnp
from jax import lax, vmap, jit
jax.config.update("jax_enable_x64", True)
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import ndtr, ndtri

# Print the list of JAX devices available (side-effect, useful for debugging)
#print("Devices:", jax.devices())

import gc