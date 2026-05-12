
from auxiliar_code.config_set import *

################################################
# PLOTS DAILY RETURNS FACETS
################################################

def plot_daily_returns(
    series_returns: pd.DataFrame,
    title: str = "",
    output_path: str = None,
    year_step: int = 4
):
    """
    Grafica los rendimientos diarios en un panel 4x3 adaptado 
    con calidad académica (estilo tesis).
    """

    # ------------------------------------------------------------
    # 1. Validaciones básicas
    # ------------------------------------------------------------
    if not isinstance(series_returns, pd.DataFrame):
        raise TypeError("`series_returns` debe ser un pandas DataFrame.")
    if series_returns.empty:
        raise ValueError("`series_returns` está vacío.")

    df = series_returns.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # ------------------------------------------------------------
    # 2. Configuración del lienzo (Tamaño Tesis)
    # ------------------------------------------------------------
    # Un figsize grande garantiza que la fuente 14 no se asfixie.
    # LaTeX lo escalará armónicamente en el documento.
    fig = plt.figure(figsize=(16, 18), dpi=300)

    gs = fig.add_gridspec(
        nrows=4,
        ncols=3,
        left=0.08,
        right=0.98,
        top=0.95 if not title else 0.92,
        bottom=0.10,
        wspace=0.25,
        hspace=0.45  # Espacio vertical para que quepan los años rotados
    )

    # ------------------------------------------------------------
    # 3. Posiciones y Diccionario de Nombres/Colores
    # ------------------------------------------------------------
    instruments = list(df.columns)

    positions = {
        "spx_usa":    (0, 0), "spx_canada": (0, 1), "spx_mexico": (0, 2),
        "crude_oil":  (1, 0), "gas":        (1, 1), "gold":       (1, 2),
        "usa_sov":    (2, 0), "mxn_sov":    (2, 1), "fx_canada":  (2, 2),
        "fx_usa":     (3, 1),
    }

    # Nombres formales para los títulos (puedes ajustarlos a tu gusto)
    nombres_formales = {
        "spx_usa": "S&P 500 (EE.UU.)",
        "spx_canada": "S&P/TSX (Canadá)",
        "spx_mexico": "S&P/BMV IPC (México)",
        "crude_oil": "Petróleo",
        "gas": "Gas natural",
        "gold": "Oro",
        "usa_sov": "Bonos soberanos (EE.UU.)",
        "mxn_sov": "Bonos soberanos (México)",
        "fx_canada": "Tipo de cambio (MXN/CAD)",
        "fx_usa": "Tipo de cambio (MXN/USD)",
    }

    colors = {
        "spx_usa":    "#66c2a5", "spx_canada": "#fc8d62", "spx_mexico": "#8da0cb",
        "crude_oil":  "#e78ac3", "gas":        "#a6d854", "gold":       "#ffd92f",
        "usa_sov":    "#e5c494", "mxn_sov":    "#b3b3b3", "fx_canada":  "#66c2a5",
        "fx_usa":     "#fc8d62",
    }

    # ------------------------------------------------------------
    # 4. Formateadores de Ejes
    # ------------------------------------------------------------
    def y_percent_number(x, pos):
        return f"{100 * x:.0f}"

    yfmt = FuncFormatter(y_percent_number)
    locator = mdates.YearLocator(base=year_step)
    formatter = mdates.DateFormatter("%Y")

    axes_dict = {}

    # ------------------------------------------------------------
    # 5. Creación de Paneles
    # ------------------------------------------------------------
    for name in instruments:
        row, col = positions[name]
        ax = fig.add_subplot(gs[row, col])
        axes_dict[name] = ax

        s = df[name].dropna()

        # Respiro del 5% adaptativo en el eje Y
        y_min, y_max = s.min(), s.max()
        padding_y = (y_max - y_min) * 0.05
        ax.set_ylim(y_min - padding_y, y_max + padding_y)
        
        # Ajuste del Eje X para no dejar espacios en blanco
        rango_tiempo = s.index.max() - s.index.min()
        padding_x = rango_tiempo * 0.02 
        ax.set_xlim(s.index.min(), s.index.max() + padding_x)

        # Gráfico principal
        ax.plot(s.index, s.values, color=colors.get(name, "#333333"), 
                linewidth=0.8, alpha=0.90, solid_capstyle="round", zorder=4)

        # Línea base en el cero (sólida y endurecida)
        ax.axhline(0, color='black', linewidth=1.2, linestyle='-', zorder=3)

        # Cuadrícula estándar
        ax.grid(True, axis="both", linestyle="--", linewidth=0.8, color="lightgray", alpha=0.7, zorder=0)

        # Título limpio de la variable
        titulo = nombres_formales.get(name, name)
        ax.set_title(titulo, fontsize=18, fontweight='bold', pad=12)

        # Formato de Y
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(yfmt)

        # Formato de X
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # Ticks endurecidos hacia afuera (Formato Tesis)
        ax.tick_params(axis='both', which='major', labelsize=14, direction='out', length=6, width=1.2)

        # Remoción de bordes innecesarios
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------
    # 6. Apagar etiquetas X internas y rotar las inferiores
    # ------------------------------------------------------------
    # Solo las gráficas que están hasta abajo en su respectiva columna llevan fechas
    x_label_panels = {"usa_sov", "fx_usa", "fx_canada"}

    for name, ax in axes_dict.items():
        if name in x_label_panels:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation='vertical', ha='center')
        else:
            # Apagar etiquetas de X para gráficas superiores (comparten el mismo eje temporal)
            ax.tick_params(labelbottom=False)

    # ------------------------------------------------------------
    # 7. Limpiar espacios vacíos del grid 4x3
    # ------------------------------------------------------------
    ax_empty_left = fig.add_subplot(gs[3, 0])
    ax_empty_left.axis("off")

    ax_empty_right = fig.add_subplot(gs[3, 2])
    ax_empty_right.axis("off")

    # ------------------------------------------------------------
    # 8. Etiqueta Global y Título
    # ------------------------------------------------------------
    fig.supylabel("Rendimientos diarios (%)", fontsize=18, fontweight="bold", x=0.02)

    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

    # ------------------------------------------------------------
    # 9. Guardar
    # ------------------------------------------------------------
    if output_path is not None:
        folder = os.path.dirname(output_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        fig.savefig(output_path, format='pdf', bbox_inches="tight")
        print(f"Gráfico guardado en: {output_path}")

    plt.show()
    return fig, axes_dict

################################################
# PLOTS CORRELATION HEATMAP
################################################

def plot_correlation_heatmap(
    series_returns: pd.DataFrame,
    method: str = "pearson",
    title: str = "",
    annot: bool = True,
    output_path: str = None,
    cmap_name: str = "vlag"
):
    """
    Heatmap triangular de correlaciones con formato premium para tesis.
    Escalado para que LaTeX lo reduzca sin perder nitidez.
    """

    # ------------------------------------------------------------
    # 1. Validaciones
    # ------------------------------------------------------------
    if not isinstance(series_returns, pd.DataFrame):
        raise TypeError("`series_returns` debe ser un pandas DataFrame.")

    if series_returns.empty:
        raise ValueError("`series_returns` está vacío.")

    valid_methods = {"pearson", "spearman", "kendall"}
    if method not in valid_methods:
        raise ValueError(f"`method` debe ser uno de {valid_methods}.")

    df = series_returns.select_dtypes(include=[np.number]).copy()

    if df.shape[1] < 2:
        raise ValueError("Se requieren al menos dos columnas numéricas.")

    # ------------------------------------------------------------
    # 2. Diccionario de Nombres Formales (Estilo Tesis)
    # ------------------------------------------------------------
    nombres_formales = {
        "spx_usa": "S&P 500 (EE.UU.)",
        "spx_canada": "S&P/TSX (Canadá)",
        "spx_mexico": "S&P/BMV IPC (México)",
        "crude_oil": "Petróleo",
        "gas": "Gas natural",
        "gold": "Oro",
        "usa_sov": "Bonos soberanos (EE.UU.)",
        "mxn_sov": "Bonos soberanos (México)",
        "fx_canada": "Tipo de cambio (MXN/CAD)",
        "fx_usa": "Tipo de cambio (MXN/USD)",
    }
    
    # Renombrar columnas para que la gráfica luzca profesional
    df.rename(columns=nombres_formales, inplace=True)

    # ------------------------------------------------------------
    # 3. Correlaciones y recorte del triángulo
    # ------------------------------------------------------------
    corr = df.corr(method=method)

    # Quitamos la primera fila y la última columna para el triángulo inferior.
    corr_plot = corr.iloc[1:, :-1].copy()

    # k=1 oculta solo el triángulo superior estricto.
    mask = np.triu(np.ones_like(corr_plot, dtype=bool), k=1)

    # ------------------------------------------------------------
    # 4. Configuración del Lienzo (Lienzo Grande, Fuentes Claras)
    # ------------------------------------------------------------
    sns.set_theme(style="white")
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    cmap = sns.color_palette(cmap_name, as_cmap=True)

    hm = sns.heatmap(
        corr_plot,
        mask=mask,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        linewidths=1.0,           # Separación ligeramente más gruesa entre celdas
        linecolor="white",        # Bordes blancos lucen más elegantes que grises
        cbar=True,
        cbar_kws={
            "shrink": 0.82,
            "aspect": 20,
            "pad": 0.04,
        },
        annot=annot,
        fmt=".2f",
        annot_kws={
            "size": 12,           # Números dentro de las celdas más legibles
            "color": "#111111",
            "weight": "medium"
        } if annot else None,
        ax=ax
    )

    # ------------------------------------------------------------
    # 5. Formato de Títulos y Etiquetas (El Toque Final)
    # ------------------------------------------------------------
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Aumentar y ajustar las etiquetas de los ejes
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=18
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=18
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    # Formato de la Barra de Color
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14, direction='out', length=5, width=1)
    cbar.set_label(
        f"Coeficiente de correlación ({method.capitalize()})",
        size=16,
        fontweight='bold',
        labelpad=15
    )

    # Quitar bordes para limpieza extrema
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # ------------------------------------------------------------
    # 6. Guardado
    # ------------------------------------------------------------
    if output_path is not None:
        folder = os.path.dirname(output_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        fig.savefig(
            output_path,
            format='pdf',
            bbox_inches="tight",
            facecolor="white"
        )
        print(f"Gráfico guardado en: {output_path}")

    plt.show()
    return fig, ax

################################################
# PLOT EDA DIAGNOSTICS
################################################

def plot_gold_diagnostics(
    series_returns: pd.DataFrame,
    column: str = "gold",
    rolling_window: int = 60,
    lags: int = 40,
    output_path: str | None = None
) -> None:
    """
    Genera diagnósticos gráficos para los rendimientos del oro con formato Tesis
    e incluye un Boxplot sobrelapado en el histograma.
    """

    if column not in series_returns.columns:
        raise ValueError(f"La columna '{column}' no existe en series_returns.")

    # Extraer serie, fechas y volatilidad
    r = series_returns[column].dropna()
    rolling_vol = r.rolling(rolling_window).std().dropna()

    # =====================================================================
    # CREACIÓN DEL LIENZO
    # =====================================================================
    fig = plt.figure(figsize=(16, 14), dpi=300)
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25)

    ax_hist = fig.add_subplot(gs[0, 0])
    ax_qq = fig.add_subplot(gs[0, 1])
    ax_vol = fig.add_subplot(gs[1, :])
    ax_acf_r = fig.add_subplot(gs[2, 0])
    ax_acf_r2 = fig.add_subplot(gs[2, 1])

    axes = [ax_hist, ax_qq, ax_vol, ax_acf_r, ax_acf_r2]

    # Colores institucionales
    color_main = '#003366'  # Azul marino
    color_fill = '#3399FF'  # Azul claro
    color_theo = '#A50F15'  # Rojo oscuro para líneas teóricas

    # =====================================================================
    # 1. HISTOGRAMA, DISTRIBUCIÓN NORMAL Y BOXPLOT SOBRELAPADO
    # =====================================================================
    x_grid = np.linspace(r.min(), r.max(), 500)
    normal_density = stats.norm.pdf(x_grid, loc=r.mean(), scale=r.std())

    # Histograma y curva normal
    ax_hist.hist(r.values, bins=60, density=True, color=color_fill, alpha=0.5, edgecolor='white', zorder=3)
    ax_hist.plot(x_grid, normal_density, color=color_theo, linewidth=2.5, zorder=4, label="Normal Teórica")
    ax_hist.set_title("Distribución Empírica, Normal y Boxplot", fontsize=18, fontweight='bold', pad=15)
    ax_hist.legend(loc="upper right", frameon=False, fontsize=16)
    ax_hist.set_ylabel("Densidad", fontsize=16)

    # --- BOXPLOT SOBRELAPADO ---
    # Creamos un eje gemelo para que el boxplot flote independientemente de la densidad Y del histograma
    ax_box = ax_hist.twinx()
    
    # Dibujamos el boxplot horizontal (vert=False)
    bp = ax_box.boxplot(
        r.values, 
        vert=False, 
        positions=[0.45],   # Altura relativa (flotando arriba en la gráfica)
        widths=0.15,        # Grosor de la caja
        patch_artist=True,  # Permite rellenar de color
        manage_ticks=False  # Evita que altere los ticks del eje X
    )
    
    # Estilizar el boxplot para que combine con la tesis
    for patch in bp['boxes']:
        patch.set_facecolor(color_fill)
        patch.set_alpha(0.4)
        patch.set_edgecolor(color_main)
        patch.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_color(color_theo)
        median.set_linewidth(2.5) # Mediana en rojo igual que la campana normal
    for whisker in bp['whiskers']:
        whisker.set_color(color_main)
        whisker.set_linewidth(1.5)
        whisker.set_linestyle('--')
    for cap in bp['caps']:
        cap.set_color(color_main)
        cap.set_linewidth(1.5)
    for flier in bp['fliers']:
        flier.set(marker='o', color=color_main, alpha=0.5, markersize=4)

    ax_box.set_ylim(0, 1) # Normalizamos el eje gemelo de 0 a 1
    ax_box.axis('off')    # Apagamos los bordes y números del eje secundario para que sea invisible

    # =====================================================================
    # 2. Q-Q PLOT (Dibujado a mano)
    # =====================================================================
    (osm, osr), (slope, intercept, _) = stats.probplot(r.values, dist="norm")
    ax_qq.plot(osm, osr, marker='o', linestyle='none', color=color_main, markersize=5, alpha=0.5, zorder=3)
    ax_qq.plot(osm, slope * osm + intercept, color=color_theo, linewidth=2.5, zorder=4)
    ax_qq.set_title("Q-Q Plot vs. Distribución Normal", fontsize=18, fontweight='bold', pad=15)
    ax_qq.set_xlabel("Cuantiles Teóricos", fontsize=16)
    ax_qq.set_ylabel("Cuantiles Ordenados", fontsize=16)

    # =====================================================================
    # 3. VOLATILIDAD MÓVIL
    # =====================================================================
    ax_vol.plot(rolling_vol.index, rolling_vol.values, color=color_main, linewidth=1.5, zorder=4)
    ax_vol.set_title(f"Volatilidad Móvil ({rolling_window} días)", fontsize=18, fontweight='bold', pad=15)
    ax_vol.set_ylabel("Desviación Estándar", fontsize=16)
    
    ax_vol.xaxis.set_major_locator(mdates.YearLocator())
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation='vertical', ha='center')
    
    y_min_vol, y_max_vol = rolling_vol.min(), rolling_vol.max()
    pad_vol = (y_max_vol - y_min_vol) * 0.1
    ax_vol.set_ylim(y_min_vol - pad_vol, y_max_vol + pad_vol)

    # =====================================================================
    # 4 y 5. GRÁFICAS ACF (Con rangos Y dinámicos)
    # =====================================================================
    def format_acf(ax, serie, lags, title):
        plot_acf(serie, lags=lags, zero=False, ax=ax, color=color_main, vlines_kwargs={"colors": color_main, "linewidth": 2})
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        ax.set_ylabel("Autocorrelación", fontsize=16)
        ax.set_xlabel("Rezagos (Lags)", fontsize=16)
        
        if ax.get_title() == "Autocorrelation":
            ax.set_title(title)
            
        acf_vals = acf(serie, nlags=lags)[1:]
        val_max = max(abs(acf_vals.min()), abs(acf_vals.max()))
        limit = max(val_max * 1.2, 0.1) 
        ax.set_ylim(-limit, limit)

    format_acf(ax_acf_r, r, lags, "ACF de Rendimientos")
    format_acf(ax_acf_r2, r.pow(2), lags, "ACF de Rendimientos al Cuadrado")

    # =====================================================================
    # APLICAR FORMATO MAESTRO A TODOS LOS PANELES
    # =====================================================================
    for ax in axes:
        ax.grid(True, axis='both', linestyle='--', color='lightgray', linewidth=0.8, alpha=0.7, zorder=0)
        ax.tick_params(axis='both', labelsize=16, direction='out', length=6, width=1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if ax != ax_vol:  
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=False))

    ax_hist.axhline(0, color='black', lw=1.2, zorder=2)
    ax_vol.axhline(0, color='black', lw=1.2, zorder=2)

    # =====================================================================
    # GUARDAR
    # =====================================================================
    if output_path is not None:
        fig.savefig(output_path, format='pdf', bbox_inches="tight")
        print(f"Gráfico guardado en: {output_path}")

    plt.show()

################################################
# PLOTS ROLLING CORRELATION PANELS
################################################

def plot_rolling_dynamics(x, y, y_static, **kwargs):
    """
    Plot rolling correlation vs time with a horizontal static correlation line.

    This function is designed to be used inside a Seaborn FacetGrid `map()` call.
    It:
      - Plots the rolling correlation series over time.
      - Draws a horizontal dashed line at the static (full-sample) correlation.
      - Colors areas where rolling correlation is above/below the static level.
      - Adds a zero line and formats the x-axis as years.

    Parameters
    ----------
    x : pandas.Series or array-like
        Time axis (dates) for the rolling correlation.
    y : pandas.Series or array-like
        Rolling correlation values over time.
    y_static : pandas.Series or array-like
        Static correlation values (constant in each facet); the function uses
        the first element as the horizontal reference.
    **kwargs :
        Additional keyword arguments that Seaborn passes, including:
        - 'color' : color automatically chosen for the facet.

    Returns
    -------
    None
        The function modifies the current Axes (plt.gca()) in-place.
    """
    # Get current Axes from Matplotlib
    ax = plt.gca()
    
    # Extract the color assigned by Seaborn (if any), otherwise use a default
    color = kwargs.get('color', '#2c3e50')
    
    # Plot the rolling (dynamic) correlation line over time
    ax.plot(x, y, color=color, linewidth=1.5, alpha=0.9, label='Dinámica (1a)')
    
    # Check that y_static has at least one observation
    if len(y_static) > 0:
        # Use the first value as the static correlation level
        static_val = y_static.iloc[0]
    else:
        # If no static value is available, skip drawing for this panel
        return
    
    # Draw a horizontal dashed line at the static correlation level
    ax.axhline(static_val, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.7)
    
    # Fill the area where rolling correlation is above the static value
    ax.fill_between(
        x, y, static_val,
        where=(y >= static_val),
        interpolate=True,
        color='#3498db',
        alpha=0.15
    )

    # Fill the area where rolling correlation is below the static value
    ax.fill_between(
        x, y, static_val,
        where=(y < static_val),
        interpolate=True,
        color='#e74c3c',
        alpha=0.15
    )

    # Draw a light horizontal line at zero correlation
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    
    # Set major x-ticks every 4 years
    ax.xaxis.set_major_locator(mdates.YearLocator(4))

    # Format x-axis ticks as two-digit years (e.g., '06, '10, '14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))

def plot_rolling_correlation_panels(
    series_returns,
    window_size=252,
    pairs=None,
    pairs_per_figure=9
):
    """
    Plot rolling and static correlations for all (or selected) asset pairs in paginated FacetGrids.

    For each pair of assets:
      - Computes a rolling correlation over a specified window.
      - Computes the full-sample (static) correlation.
      - Builds a tidy DataFrame with Date, Rolling Correlation, Static Correlation, Pair.
    Then:
      - Groups pairs into batches (pages) of `pairs_per_figure`.
      - For each batch, draws a Seaborn FacetGrid with a panel per pair.

    Parameters
    ----------
    series_returns : pandas.DataFrame
        DataFrame of returns with DateTime index and one column per asset.
    window_size : int, optional
        Rolling window length (in days) used for the correlation. Default is 252.
    pairs : list of (str, str), optional
        List of (asset1, asset2) tuples. If None, all unique combinations
        of columns of size 2 are used.
    pairs_per_figure : int, optional
        Number of pairs per figure (i.e., per FacetGrid). Default is 9 (3x3 grid).

    Returns
    -------
    None
        The function shows the figures and does not return anything.
    """
    # ============================================================
    # 1. CONFIGURATION AND PAIR GENERATION
    # ============================================================

    # If specific pairs are not provided, generate all 2-combinations of columns
    if pairs is None:
        # Create all unique unordered pairs of the DataFrame columns
        all_pairs = list(combinations(series_returns.columns, 2))
    else:
        # Use the user-provided list of (col1, col2) pairs
        all_pairs = list(pairs)

    # Compute how many pairs we have in total
    total_pairs = len(all_pairs)

    # Print informational message about total pairs
    print(f"Total de pares generados: {total_pairs}")

    # Compute how many separate figures will be created (pages)
    total_figures = math.ceil(total_pairs / pairs_per_figure)

    # Print informational message about total figures
    print(f"Se generarán aproximadamente {total_figures} figuras separadas.")

    # ============================================================
    # 2. PAGINATION LOOP
    # ============================================================

    # Iterate over each "page" of pairs
    for i in range(total_figures):
        # Compute the starting index of the current batch
        start_idx = i * pairs_per_figure

        # Compute the ending index (exclusive) of the current batch
        end_idx = start_idx + pairs_per_figure

        # Select the current batch of pairs for this figure
        current_pairs_batch = all_pairs[start_idx:end_idx]
        
        # Initialize a list to collect tidy DataFrames for each pair
        plot_data_list = []

        # ============================================================
        # 3. DATA PROCESSING FOR THE CURRENT BATCH
        # ============================================================

        # Loop over each pair in the current batch
        for asset1, asset2 in current_pairs_batch:
            # Compute rolling correlation between asset1 and asset2
            rolling = series_returns[asset1].rolling(window=window_size).corr(series_returns[asset2])

            # Compute static (full-sample) correlation between asset1 and asset2
            static = series_returns[asset1].corr(series_returns[asset2])
            
            # Build a small DataFrame with date, rolling and static correlations for this pair
            temp_df = pd.DataFrame({
                'Date': series_returns.index,
                'Rolling Correlation': rolling,
                'Static Correlation': static,
                'Pair': f"{asset1} vs {asset2}"
            }).dropna()

            # Append this pair's data to the list for the current batch
            plot_data_list.append(temp_df)
        
        # If no data was generated (e.g., all empty), skip to the next batch
        if not plot_data_list:
            continue

        # Concatenate all pair DataFrames into a single batch DataFrame
        df_batch = pd.concat(plot_data_list)

        # ============================================================
        # 4. PLOTTING FOR THE CURRENT BATCH
        # ============================================================

        # Set a clean, light theme for the rolling correlation panels
        sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#fdfdfd"})
        
        # Create a FacetGrid with one facet per 'Pair'
        g = sns.FacetGrid(
            df_batch,
            col='Pair',
            col_wrap=3,
            height=3.0,
            aspect=1.8,
            sharex=True,
            sharey=True
        )
        
        # Map the custom rolling dynamics plot onto each facet
        g.map(
            plot_rolling_dynamics,
            'Date',
            'Rolling Correlation',
            'Static Correlation'
        )
        
        # Set the title of each facet to the pair name
        g.set_titles(col_template="{col_name}", fontweight='bold', size=10, color='#333333')

        # Fix the y-axis limits for all panels to [-1, 1]
        g.set(ylim=(-1.0, 1.0))

        # Set axis labels (omit x-label text since dates are obvious)
        g.set_axis_labels("", "Corr")
        
        # Adjust spacing between subplots and top margin
        plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.15)

        # Add a global title for this figure (batch index)
        g.fig.suptitle(
            f"Dinámica de Conectividad - Grupo {i+1} de {total_figures}",
            fontsize=16,
            fontweight='bold',
            y=0.98,
            color='#2c3e50'
        )
        
        # Show the figure for the current batch before moving to the next one
        plt.show()

################################################################
####### plots for contribution decompositions #######
################################################################

def stacked_contributions(
    theta_df: pd.DataFrame,
    var: str,
    include_self: bool = True,
    contributors_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    
    if not isinstance(theta_df.columns, pd.MultiIndex) or theta_df.columns.nlevels != 2:
        raise ValueError("theta_df debe tener columnas MultiIndex con niveles: (to, from).")

    to_level = theta_df.columns.get_level_values(0)
    from_level = theta_df.columns.get_level_values(1)
    mask_to = (to_level == var)
    
    if not mask_to.any():
        raise KeyError(f"Variable objetivo '{var}' no encontrada.")

    sub = theta_df.loc[:, mask_to].copy()
    sub.columns = from_level[mask_to]

    # Aplicamos el ordenamiento base si se proporciona
    if contributors_order is not None:
        existing = [c for c in contributors_order if c in sub.columns]
        remaining = [c for c in sub.columns if c not in existing]
        ordered = existing + remaining
        sub = sub.reindex(columns=ordered)

    # --- LÓGICA: PROPIA CONTRIBUCIÓN HASTA ARRIBA ---
    if include_self and var in sub.columns:
        # Movemos la variable objetivo al final de la lista
        cols = list(sub.columns)
        cols.remove(var)
        cols.append(var) 
        sub = sub[cols]
    elif not include_self and var in sub.columns:
        sub = sub.drop(columns=[var])

    # Normalización al 100%
    row_sums = sub.sum(axis=1).astype(float)
    row_sums_safe = row_sums.replace(0.0, 1.0)
    sub_pct = 100.0 * sub.div(row_sums_safe, axis=0)

    return sub_pct


def graficar_contribucion_en_eje(
    ax, sub_pct: pd.DataFrame, nombre_indice: str, paleta_colores: dict
):
    x = sub_pct.index
    y = sub_pct.T.values
    labels_vars = sub_pct.columns.tolist()

    # Los colores deben coincidir con el orden actual de las columnas (con 'self' al final)
    colores_plot = [paleta_colores.get(col, '#333333') for col in labels_vars]

    # --- AJUSTE DEL EJE X ---
    rango_tiempo = x.max() - x.min()
    padding_x = rango_tiempo * 0.02 
    ax.set_xlim(x.min(), x.max() + padding_x)

    # --- EJE Y PRINCIPAL (Fijo 0-100%) ---
    ax.set_ylim(0.0, 100.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5)) 
    
    ax.tick_params(axis='y', labelsize=24)
    ax.grid(axis='y', color='lightgray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)

    # --- GRÁFICO APILADO ---
    # Usamos alpha=0.85 para colores vivos y profesionales
    ax.stackplot(x, y, labels=labels_vars, colors=colores_plot, alpha=0.85, zorder=3)
    
    ax.set_title(nombre_indice, fontsize=24, fontweight='bold', pad=15)

    # --- PERIODOS DE ESTRÉS (Líneas punteadas moradas como en NET) ---
    study_periods = [
        ('2007-08-09', '2009-06-30'), ('2008-09-15', '2008-09-15'), ('2010-05-06', '2010-05-06'),        
        ('2011-08-05', '2011-08-05'), ('2014-11-27', '2014-11-27'), ('2016-11-08', '2016-11-08'),
        ('2020-02-24', '2020-11-09'), ('2022-02-24', '2022-02-24'), ('2023-03-08', '2023-05-01'),
        ('2025-04-02', '2025-04-02'), ('2026-02-28', '2026-04-19')
    ]
    
    for start, end in study_periods:
        start_date = pd.to_datetime(start)
        if start_date <= x.max() and start_date >= x.min():
            ax.axvline(start_date, color='purple', linestyle=':', lw=2.5, alpha=0.9, zorder=4)

    # --- FORMATO DEL EJE X ---
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation='vertical', ha='center', fontsize=24)

################################################################
#######connectedness index with confidence intervals#######
################################################################
def graficar_indice_en_eje(ax, idx, datos, datos_lo=None, datos_hi=None, 
                           nombre_indice="Índice", es_net=False, es_pairwise=False, burn_in=200):
    
    serie = datos if isinstance(datos, pd.Series) else pd.Series(datos, index=idx)
    tiene_intervalos = (datos_lo is not None) and (datos_hi is not None)
    
    if tiene_intervalos:
        serie_lo = pd.Series(datos_lo, index=idx)
        serie_hi = pd.Series(datos_hi, index=idx)

    # --- AJUSTE DEL EJE X ---
    rango_tiempo = serie.index.max() - serie.index.min()
    padding_x = rango_tiempo * 0.02 
    ax.set_xlim(serie.index.min(), serie.index.max() + padding_x)

    # --- EJE Y PRINCIPAL ---
    if tiene_intervalos:
        y_min_estable = serie_lo.iloc[burn_in:].min()
        y_max_estable = serie_hi.iloc[burn_in:].max()
    else:
        y_min_estable = serie.iloc[burn_in:].min()
        y_max_estable = serie.iloc[burn_in:].max()
    
    padding_y = (y_max_estable - y_min_estable) * 0.05
    ax.set_ylim(y_min_estable - padding_y, y_max_estable + padding_y)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=False))
    
    ax.tick_params(axis='y', labelsize=24, length=8, width=1.5, direction='out')
    ax.grid(axis='y', color='lightgray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)

    # --- LÍNEA BASE PARA NETOS/PAIRWISE ---
    if es_net or es_pairwise:
        ax.axhline(0, color='black', lw=1.2, linestyle='-', zorder=3)

    legend_elements = []

    # --- GRÁFICO PRINCIPAL E INTERVALO DE CONFIANZA ---
    line_principal, = ax.plot(serie.index, serie.values, color='#003366', lw=1.5, zorder=5)
    
    if tiene_intervalos:
        # Colores y opacidad ajustados a tus especificaciones
        ax.fill_between(serie.index, serie_lo, serie_hi, color='#0883FF', alpha=0.28, zorder=4)
        legend_elements.append(Patch(facecolor='#0883FF', alpha=0.3, label='Intervalo de confianza (95%)'))
        
    ax.set_title(nombre_indice, fontsize=24, fontweight='bold', pad=15)

    # Colores ENDURECIDOS para NET/PDC
    color_emisor = '#2CA02C'   # Verde sólido
    color_receptor = '#D62728' # Rojo carmesí

    # =========================================================
    # LÓGICA CONDICIONAL: ETIQUETAS DE LEYENDA
    # =========================================================
    if es_pairwise:
        lbl_sig_pos, lbl_insig_pos = 'Significativo: Emisora Neta', 'No Significativo: Emisora Neta'
        lbl_sig_neg, lbl_insig_neg = 'Significativo: Receptora Neta', 'No Significativo: Receptora Neta'
    else:
        lbl_sig_pos, lbl_insig_pos = 'Significativo: Emisor neto', 'No Significativo: Emisor neto'
        lbl_sig_neg, lbl_insig_neg = 'Significativo: Receptor neto', 'No Significativo: Receptor neto'

    if not es_net and not es_pairwise:
        p95 = np.percentile(serie.iloc[burn_in:], 95)
        p99 = np.percentile(serie.iloc[burn_in:], 99)

        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim()) 
        ax2.set_yticks([p95, p99])
        ax2.set_yticklabels(['P95', 'P99'], fontsize=24)
        ax2.tick_params(axis='y', length=8, width=1.5, direction='out')

        for p in [p95, p99]:
            ax.axhline(p, color='red', linestyle='--', lw=0.9, alpha=0.4, zorder=2)

        ax.axhspan(p95, p99, color='#FF9999', alpha=0.3, zorder=1) 
        ax.axhspan(p99, ax.get_ylim()[1], color='#FF9999', alpha=0.6, zorder=1) 
        legend_elements.append(Patch(facecolor='#FF9999', alpha=0.4, label='Alta interconexión histórica'))

    else:
        if tiene_intervalos:
            cond_sig_pos = serie_lo > 0
            cond_sig_neg = serie_hi < 0
        else:
            cond_sig_pos = pd.Series(False, index=serie.index)
            cond_sig_neg = pd.Series(False, index=serie.index)

        cond_insig_pos = (serie > 0) & ~cond_sig_pos
        cond_insig_neg = (serie < 0) & ~cond_sig_neg

        # SIGNIFICATIVOS: Colores sólidos
        ax.fill_between(serie.index, 0, 1, where=cond_sig_pos, facecolor=color_emisor, alpha=0.70, transform=ax.get_xaxis_transform(), zorder=1)
        ax.fill_between(serie.index, 0, 1, where=cond_sig_neg, facecolor=color_receptor, alpha=0.70, transform=ax.get_xaxis_transform(), zorder=1)
        
        # NO SIGNIFICATIVOS: Trama diagonal (hatch)
        ax.fill_between(serie.index, 0, 1, where=cond_insig_pos, facecolor=color_emisor, alpha=0.25, hatch='///', edgecolor='white', linewidth=0.0, transform=ax.get_xaxis_transform(), zorder=1)
        ax.fill_between(serie.index, 0, 1, where=cond_insig_neg, facecolor=color_receptor, alpha=0.25, hatch='///', edgecolor='white', linewidth=0.0, transform=ax.get_xaxis_transform(), zorder=1)
        
        legend_elements.extend([
            Patch(facecolor=color_emisor, alpha=0.70, label=lbl_sig_pos),
            Patch(facecolor=color_emisor, alpha=0.25, hatch='///', edgecolor='white', label=lbl_insig_pos),
            Patch(facecolor=color_receptor, alpha=0.70, label=lbl_sig_neg),
            Patch(facecolor=color_receptor, alpha=0.25, hatch='///', edgecolor='white', label=lbl_insig_neg)
        ])

    # --- PERIODOS DE ESTRÉS ---
    study_periods = [
        ('2007-08-09', '2009-06-30'), ('2008-09-15', '2008-09-15'), ('2010-05-06', '2010-05-06'),        
        ('2011-08-05', '2011-08-05'), ('2014-11-27', '2014-11-27'), ('2016-11-08', '2016-11-08'),
        ('2020-02-24', '2020-11-09'), ('2022-02-24', '2022-02-24'), ('2023-03-08', '2023-05-01'),
        ('2025-04-02', '2025-04-02'), ('2026-02-28', '2026-04-19')
    ]
    
    for start, end in study_periods:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        if start_date <= serie.index.max() and end_date >= serie.index.min():
            ax.axvline(start_date, color='purple', linestyle=':', lw=2.5, alpha=0.9, zorder=3)
            # Solo la banda gris si NO es net/pairwise
            if not (es_net or es_pairwise) and start_date != end_date:
                ax.axvspan(start_date, end_date, color='#BEBEBE', alpha=0.65, zorder=0)

    legend_elements.append(Line2D([0], [0], color='purple', linestyle=':', lw=2.5, label='Inicio evento de estrés'))
    if not (es_net or es_pairwise):
        legend_elements.append(Patch(facecolor='#BEBEBE', alpha=0.65, label='Periodo de turbulencia'))

    # --- FORMATO FINAL DEL EJE X ---
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Marcas exteriores (Ticks)
    ax.tick_params(axis='x', direction='out', length=8, width=1.5, bottom=True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation='vertical', ha='center', fontsize=24)
    
    return legend_elements

def graficar_indice_publicacion(idx, datos, datos_lo=None, datos_hi=None, nombre_indice="Índice", 
                                es_net=False, guardar=False, ruta_archivo="grafica.pdf", burn_in=200,
                                leyenda_abajo=False):
    
    # 1. Conservamos EL 100% DE LOS DATOS para graficar desde el día cero
    serie = datos if isinstance(datos, pd.Series) else pd.Series(datos, index=idx)
    
    # Verificamos si se proporcionaron intervalos de confianza
    tiene_intervalos = (datos_lo is not None) and (datos_hi is not None)
    
    if tiene_intervalos:
        serie_lo = pd.Series(datos_lo, index=idx)
        serie_hi = pd.Series(datos_hi, index=idx)

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

    # --- AJUSTE DEL EJE X (Toda la muestra + respiro final) ---
    rango_tiempo = serie.index.max() - serie.index.min()
    padding_x = rango_tiempo * 0.02 
    ax1.set_xlim(serie.index.min(), serie.index.max() + padding_x)

    # --- EJE Y PRINCIPAL (Zoom inteligente adaptativo) ---
    if tiene_intervalos:
        y_min_estable = serie_lo.iloc[burn_in:].min()
        y_max_estable = serie_hi.iloc[burn_in:].max()
    else:
        y_min_estable = serie.iloc[burn_in:].min()
        y_max_estable = serie.iloc[burn_in:].max()
    
    padding_y = (y_max_estable - y_min_estable) * 0.05
    limite_inferior = y_min_estable - padding_y
    limite_superior = y_max_estable + padding_y
    
    ax1.set_ylim(limite_inferior, limite_superior)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=12, integer=False))

    # ¡AQUÍ ESTÁ EL PRIMER CAMBIO! Ticks del eje Y principal
    ax1.tick_params(axis='y', labelsize=14, direction='out', length=6, width=1.2)

    # --- CUADRÍCULA (Grid gris pálido) ---
    ax1.grid(axis='y', color='lightgray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)

    # --- LÍNEA BASE (Cero) PARA ÍNDICES NETOS ---
    if es_net:
        ax1.axhline(0, color='black', lw=1.2, linestyle='-', zorder=3)

    # --- GRÁFICO PRINCIPAL ---
    line_principal, = ax1.plot(serie.index, serie.values, color='#003366', 
                               lw=1.5, label=nombre_indice, zorder=5)
    
    if tiene_intervalos:
        ax1.fill_between(serie.index, serie_lo, serie_hi, 
                         color='#0883FF', alpha=0.28, zorder=4)
    
    # --- ETIQUETA DEL EJE Y ---
    if not es_net:
        ax1.set_ylabel("Porcentaje (%)", fontsize=14, fontweight='bold')

    # =========================================================
    # LÓGICA CONDICIONAL: TCI/DIRECCIONALES vs. ÍNDICES NETOS
    # =========================================================
    
    legend_elements = [line_principal]
    
    if tiene_intervalos:
        legend_elements.append(Patch(facecolor="#0883FF", alpha=0.3, label='Intervalo de confianza del 95%'))

    # Colores ENDURECIDOS para NET
    color_emisor = '#2CA02C'   # Verde más fuerte y sólido
    color_receptor = '#D62728' # Rojo carmesí más fuerte

    if not es_net:
        # --- LÓGICA PARA TCI, FROM, TO ---
        p95 = np.percentile(serie.iloc[burn_in:], 95)
        p99 = np.percentile(serie.iloc[burn_in:], 99)

        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim()) 
        ax2.set_yticks([p95, p99])
        ax2.set_yticklabels(['P95', 'P99'], fontsize=14)
        
        # ¡AQUÍ ESTÁ EL SEGUNDO CAMBIO! Ticks del eje Y secundario (Percentiles)
        ax2.tick_params(axis='y', direction='out', length=6, width=1.2)
        
        ax2.set_ylabel("Percentiles históricos", fontsize=14, fontweight='bold')

        for p in [p95, p99]:
            ax1.axhline(p, color='red', linestyle='--', lw=0.9, alpha=0.4, zorder=2)

        ax1.axhspan(p95, p99, color='#FF9999', alpha=0.3, zorder=1) 
        ax1.axhspan(p99, ax1.get_ylim()[1], color='#FF9999', alpha=0.6, zorder=1) 
        
        legend_elements.append(Patch(facecolor='#FF9999', alpha=0.4, label='Zonas de alta interconexión histórica'))

    else:
        # --- LÓGICA PARA NET INDEX ---
        if tiene_intervalos:
            cond_sig_pos = serie_lo > 0
            cond_sig_neg = serie_hi < 0
        else:
            cond_sig_pos = pd.Series(False, index=serie.index)
            cond_sig_neg = pd.Series(False, index=serie.index)

        cond_insig_pos = (serie > 0) & ~cond_sig_pos
        cond_insig_neg = (serie < 0) & ~cond_sig_neg

        if tiene_intervalos:
            # Áreas Significativas (Sólidas)
            ax1.fill_between(serie.index, 0, 1, where=cond_sig_pos, 
                             facecolor=color_emisor, alpha=0.70, transform=ax1.get_xaxis_transform(), zorder=1)
            ax1.fill_between(serie.index, 0, 1, where=cond_sig_neg, 
                             facecolor=color_receptor, alpha=0.70, transform=ax1.get_xaxis_transform(), zorder=1)

        # Áreas No Significativas (Con Trama / Hatch)
        ax1.fill_between(serie.index, 0, 1, where=cond_insig_pos, 
                         facecolor=color_emisor, alpha=0.25, hatch='///', edgecolor='white', linewidth=0.0, 
                         transform=ax1.get_xaxis_transform(), zorder=1)
        ax1.fill_between(serie.index, 0, 1, where=cond_insig_neg, 
                         facecolor=color_receptor, alpha=0.25, hatch='///', edgecolor='white', linewidth=0.0, 
                         transform=ax1.get_xaxis_transform(), zorder=1)
        
        if tiene_intervalos:
            legend_elements.extend([
                Patch(facecolor=color_emisor, alpha=0.70, label='Significativo: Emisor neto'),
                Patch(facecolor=color_emisor, alpha=0.25, hatch='///', edgecolor='white', label='No Significativo: Emisor neto'),
                Patch(facecolor=color_receptor, alpha=0.70, label='Significativo: Receptor neto'),
                Patch(facecolor=color_receptor, alpha=0.25, hatch='///', edgecolor='white', label='No Significativo: Receptor neto')
            ])
        else:
            legend_elements.extend([
                Patch(facecolor=color_emisor, alpha=0.25, hatch='///', edgecolor='white', label='Papel: Emisor neto'),
                Patch(facecolor=color_receptor, alpha=0.25, hatch='///', edgecolor='white', label='Papel: Receptor neto')
            ])

    # =========================================================

    # --- PERIODOS DE ESTUDIO Y CRISIS ---
    study_periods = [
        ('2007-08-09', '2009-06-30'),
        ('2008-09-15', '2008-09-15'),
        ('2010-05-06', '2010-05-06'),        
        ('2011-08-05', '2011-08-05'),
        ('2014-11-27', '2014-11-27'),
        ('2016-11-08', '2016-11-08'),
        ('2020-02-24', '2020-11-09'),
        ('2022-02-24', '2022-02-24'),
        ('2023-03-08', '2023-05-01'),
        ('2025-04-02', '2025-04-02'),
        ('2026-02-28', '2026-04-19')
    ]
    
    for start, end in study_periods:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        
        if start_date <= serie.index.max() and end_date >= serie.index.min():
            
            # La línea morada SIEMPRE se dibuja
            ax1.axvline(start_date, color='purple', linestyle=':', lw=2.8, alpha=0.9, zorder=3)
            
            # La banda gris SOLO se dibuja si NO es net
            if not es_net and start_date != end_date:
                ax1.axvspan(start_date, end_date, color='#BEBEBE', alpha=0.65, zorder=0)

    linea_morada = Line2D([0], [0], color='purple', linestyle=':', lw=2.8, label='Inicio de evento de estrés por estudiar')
    legend_elements.append(linea_morada)
    
    # La etiqueta de la banda gris solo se añade a la leyenda si no es net
    if not es_net:
        legend_elements.append(Patch(facecolor='#BEBEBE', alpha=0.65, label='Periodo de estrés por estudiar'))

    # --- FORMATO FINAL DEL EJE X ---
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Marcas (ticks) exteriores 
    ax1.tick_params(axis='x', direction='out', length=6, width=1.2, bottom=True)
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation='vertical', ha='center', fontsize=14)
    
    # --- LEYENDA ---
    if es_net or leyenda_abajo:
        # Aquí bajé el Y a -0.40 para evitar cruces con los años en las gráficas individuales
        ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.20), 
                   ncol=2, frameon=False, fontsize=14, facecolor='white', framealpha=0.9)
    else:
        ax1.legend(handles=legend_elements, loc='upper left', frameon=False, 
                   fontsize=14, facecolor='white', framealpha=0.9)
    
    ax1.set_title("", fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()

    # --- GUARDAR GRÁFICO EN PDF ---
    if guardar:
        fig.savefig(ruta_archivo, format='pdf', bbox_inches='tight')
        print(f"Gráfico guardado exitosamente en: {ruta_archivo}")

    return fig, ax1

def extraer_episodios_criticos(idx, datos, datos_hi=None, cuantil=95):
    """
    Extrae los intervalos de fechas donde el índice principal O su límite superior
    superan un cuantil específico y los cruza con eventos históricos.
    """
    # 1. Aseguramos el formato Pandas
    serie = datos if isinstance(datos, pd.Series) else pd.Series(datos, index=idx)
    
    # 2. Calcular el umbral (basado en la distribución de la serie principal)
    umbral = np.percentile(serie, cuantil)
    
    # 3. Evaluar CUALQUIERA DE LAS DOS condiciones (Índice o Límite Superior)
    if datos_hi is not None:
        serie_hi = datos_hi if isinstance(datos_hi, pd.Series) else pd.Series(datos_hi, index=idx)
        # Operador OR lógico (|): True si cualquiera de los dos supera el umbral
        es_critico = (serie >= umbral) | (serie_hi >= umbral)
        texto_evaluacion = "Índice Principal O Límite Superior (Cualquiera de los dos)"
    else:
        es_critico = serie >= umbral
        texto_evaluacion = "Índice Principal (Límite superior no proporcionado)"

    if not es_critico.any():
        print(f"No se encontraron episodios que superen el P{cuantil} ({umbral:.2f})")
        return pd.DataFrame()

    # 4. Lista de periodos de estudio actualizada
    study_periods = [
        ('2007-08-09', '2009-06-30', 'Crisis subprime y GFC'),
        ('2008-09-15', '2008-09-15', 'Quiebra de Lehman Brothers'),
        ('2010-05-06', '2010-05-06', 'Flash Crash'),
        ('2011-08-05', '2011-08-05', 'Crisis techo de deuda y rebaja S&P'),
        ('2014-11-27', '2014-11-27', 'Colapso del petróleo / shock shale'),
        ('2016-11-08', '2016-11-08', 'Elección de EE.UU. / selloff del peso'),
        ('2020-02-24', '2020-11-09', 'Crash COVID-19 y giro por vacuna'),
        ('2022-02-24', '2022-02-24', 'Invasión Rusia-Ucrania'),
        ('2023-03-08', '2023-05-01', 'Silvergate-SVB-First Republic'),
        ('2025-04-02', '2025-04-02', 'Liberation Day'),
        ('2026-02-28', '2026-04-19', 'Guerra con Irán')
    ]

    # 5. Agrupar días consecutivos en "bloques"
    id_episodios = (es_critico != es_critico.shift()).cumsum()
    
    # Extraemos basándonos en 'serie' para reportar el máximo del índice real, no del intervalo
    periodos = serie[es_critico].groupby(id_episodios)
    
    resultados = []
    for _, grupo in periodos:
        fecha_inicio = grupo.index.min()
        fecha_fin = grupo.index.max()
        duracion_dias = (fecha_fin - fecha_inicio).days + 1
        valor_maximo = grupo.max() # Ahora siempre reporta el max del índice principal
        
        # --- CRUCE CON PERIODOS DE ESTUDIO ---
        eventos_capturados = []
        for p_start, p_end, p_name in study_periods:
            ps = pd.to_datetime(p_start)
            pe = pd.to_datetime(p_end)
            
            # Condición de intersección matemática de dos intervalos
            if fecha_inicio <= pe and fecha_fin >= ps:
                eventos_capturados.append(p_name)
                
        # Formatear la salida si hay múltiples eventos empalmados
        if len(eventos_capturados) > 0:
            eventos_str = " | ".join(eventos_capturados)
        else:
            eventos_str = "-" 
            
        resultados.append({
            'Fecha Inicio': fecha_inicio.strftime('%Y-%m-%d'),
            'Fecha Fin': fecha_fin.strftime('%Y-%m-%d'),
            'Duración (Días)': duracion_dias,
            f'Valor Máx. Índice (%)': round(valor_maximo, 2),
            'Evento Identificado': eventos_str
        })

    # Convertir a DataFrame
    df_resultados = pd.DataFrame(resultados)
    
    # Resumen en consola
    print(f"--- Análisis de Episodios Críticos ---")
    print(f"Evaluando: {texto_evaluacion}")
    print(f"Umbral de corte: Percentil {cuantil} ({umbral:.2f}%)")
    print(f"Total de episodios detectados: {len(df_resultados)}")
    
    # Calcular eficiencia del modelo
    episodios_con_evento = df_resultados[df_resultados['Evento Identificado'] != '-']
    if len(df_resultados) > 0:
        porcentaje_mapeado = (len(episodios_con_evento)/len(df_resultados))*100
        print(f"Episodios mapeados a eventos macro: {len(episodios_con_evento)} ({porcentaje_mapeado:.1f}%)\n")
    
    return df_resultados

# =====================================================================
# NET INDEX PANEL DE CONTRIBUCIONES (5x2) - GRÁFICOS APILADOS
# =====================================================================
def generar_panel_contribuciones(theta_df: pd.DataFrame, orden_vars: List[str], nombres_dic: Dict[str, str], ruta_salida: str, include_self: bool = True):
    """
    Crea el panel de 5x2 con los gráficos apilados.
    """
    # 1. Definir una paleta de 10 colores distintiva (Tab10 de matplotlib)
    colores_tab10 = plt.cm.tab10.colors
    paleta_fija = {var: colores_tab10[i % 10] for i, var in enumerate(orden_vars)}

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(18, 28), dpi=300, sharex=True)
    axes_flat = axes.flatten()

    elementos_leyenda = []
    etiquetas_leyenda = []
    
    # Extraer nombres limpios para la leyenda usando el diccionario
    nombres_para_leyenda = [nombres_dic.get(v, v) for v in orden_vars]

    for j, var_objetivo in enumerate(orden_vars):
        ax = axes_flat[j]
        
        # Procesar los datos para esta variable
        sub_pct = stacked_contributions(
            theta_df=theta_df, 
            var=var_objetivo, 
            include_self=include_self, 
            contributors_order=orden_vars # Asegura que las capas siempre se apilen en el mismo orden
        )
        
        titulo = f"Receptora: {nombres_dic.get(var_objetivo, var_objetivo)}"
        
        colecciones, labels_crudos = graficar_contribucion_en_eje(
            ax=ax, 
            sub_pct=sub_pct, 
            nombre_indice=titulo, 
            paleta_colores=paleta_fija
        )
        
        # Extraer elementos de leyenda solo en el primer bucle
        if j == 0:
            elementos_leyenda = colecciones
            etiquetas_leyenda = [nombres_dic.get(lbl, lbl) for lbl in labels_crudos]

    # --- AJUSTE FINO DEL ESPACIADO ---
    fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.10, hspace=0.20) 

    # --- ETIQUETA GLOBAL EJE Y ---
    fig.supylabel("Contribución porcentual (%)", fontsize=24, fontweight='bold', x=0.02)

    # --- LEYENDA GLOBAL ---
    # Truco: Invertimos el orden de la leyenda [::-1] para que el nombre de arriba 
    # en la leyenda coincida con el color de la capa superior en la gráfica apilada.
    leyenda_crisis = Line2D([0], [0], color='black', linestyle=':', lw=2.5, label='Inicio evento de estrés')
    
    handles_final = elementos_leyenda[::-1] + [leyenda_crisis]
    labels_final = etiquetas_leyenda[::-1] + ['Inicio evento de estrés']

    fig.legend(handles=handles_final, 
               labels=labels_final,
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.08), 
               ncol=3, # 3 columnas para equilibrar los 11 elementos
               fontsize=24, 
               frameon=True, 
               facecolor='white', 
               framealpha=0.9)

    fig.savefig(ruta_salida, format='pdf', bbox_inches='tight')
    print(f"Panel 5x2 de Contribuciones guardado exitosamente en: {ruta_salida}")
    plt.close(fig)

################################################################
####### plots for conectedness GRAPHS #######
################################################################
# 0) Group definitions (5 super-nodes)
GROUPS: Dict[str, List[str]] = {
    "EE.UU.":         ["spx_usa", "usa_sov", "fx_usa"],
    "Canadá":       ["spx_canada", "canada_sov", "fx_canada"],
    "México":      ["spx_mexico", "mxn_sov"],
    "Energéticos": ["crude_oil", "gas"],
    "Oro":        ["gold"],
}
OTHERS_GROUP = "Otros"

nombres_net = {
    "spx_usa": "S&P 500 (EE.UU.)",
    "spx_canada": "S&P/TSX (Canadá)",
    "spx_mexico": "S&P/BMV IPC (México)",
    "crude_oil": "Petróleo",
    "gas": "Gas natural",
    "gold": "Oro",
    "usa_sov": "Bonos soberanos de EE.UU.",
    "mxn_sov": "Bonos soberanos de México",
    "fx_canada": "Tipo de cambio Canadá",
    "fx_usa": "Tipo de cambio EE.UU.",
}


# 1) Grouped fixed positions (layout)
# =====================================================================
# 1. LAYOUT DINÁMICO (Corrección del error cy + dy)
# =====================================================================
def make_grouped_positions(
    all_nodes: List[str], group_map: Dict[str, str], group_radius: float = 2.0,
    node_spread: float = 0.40, seed: int = 1,
    y_squash: float = 1.0  # Por defecto 1.0 (No aplasta la red normal)
) -> Dict[str, Tuple[float, float]]:
    
    groups_in_use = list(dict.fromkeys(group_map.get(v, OTHERS_GROUP) for v in all_nodes))
    grupo_mexico = group_map.get("spx_mexico", "Mexico")
    
    if grupo_mexico in groups_in_use:
        groups_in_use.remove(grupo_mexico)
        groups_in_use.insert(0, grupo_mexico)
        
    Gm = len(groups_in_use)
    angles = [np.pi/2 - i * (2.0 * np.pi / Gm) for i in range(Gm)]

    centers: Dict[str, Tuple[float, float]] = {}
    for g, ang in zip(groups_in_use, angles):
        # Multiplicamos el seno por y_squash para crear un óvalo (si es distinto a 1.0)
        centers[g] = (group_radius * np.cos(ang), group_radius * np.sin(ang) * y_squash)

    pos: Dict[str, Tuple[float, float]] = {}
    
    nodos_por_grupo = {}
    for v in all_nodes:
        g = group_map.get(v, OTHERS_GROUP)
        if g not in nodos_por_grupo:
            nodos_por_grupo[g] = []
        nodos_por_grupo[g].append(v)
        
    for g, nodes in nodos_por_grupo.items():
        cx, cy = centers[g]
        n_nodos = len(nodes)
        
        if n_nodos == 1:
            pos[nodes[0]] = (cx, cy)
        else:
            angulos_locales = np.linspace(0, 2 * np.pi, n_nodos, endpoint=False)
            offset_rotacion = np.pi / n_nodos 
            for v, ang_loc in zip(nodes, angulos_locales):
                dx = node_spread * np.cos(ang_loc + offset_rotacion)
                dy = node_spread * np.sin(ang_loc + offset_rotacion) * y_squash
                
                # ¡AQUÍ ESTABA EL ERROR! Ya está corregido a cy + dy
                pos[v] = (cx + dx, cy + dy) 
                
    return pos

# 2) Group-level aggregation of θ (G×G)
def aggregate_theta_to_groups(
    theta_matrix: pd.DataFrame,
    group_map: Dict[str, str],
    normalize_rows: bool = True
) -> pd.DataFrame:
    """
    Aggregate a node-level θ (to×from) into a group-level Θ^G (G_to × G_from).

    Parameters
    ----------
    theta_matrix : DataFrame
        Square matrix with identical index and columns (same ordered names).
        Entries are row-normalized (each row sums to 1), or arbitrary weights
        if you plan to re-normalize.
    group_map : dict
        {variable -> group}. Any variable missing is mapped to OTHERS_GROUP.
    normalize_rows : bool
        If True, re-normalize each group row to sum to 1 (robust against partial groups).

    Returns
    -------
    DataFrame
        Group-level matrix of shape (#groups_in_data, #groups_in_data).
    """
    if theta_matrix.index.tolist() != theta_matrix.columns.tolist():
        raise ValueError("theta_matrix must be square with identical index and columns (same order).")

    names = theta_matrix.index.tolist()
    # Ensure every variable has a group (fallback to Others)
    gmap = {n: group_map.get(n, OTHERS_GROUP) for n in names}

    # Preserve group order encountered in the data
    gnames = list(dict.fromkeys(gmap[n] for n in names))

    # Indices by group
    G_to = {g: [i for i in names if gmap[i] == g] for g in gnames}
    G_from = G_to  # same partitioning

    M = pd.DataFrame(0.0, index=gnames, columns=gnames)
    for gt in gnames:
        Ii = G_to[gt]
        for gf in gnames:
            Jj = G_from[gf]
            if len(Ii) == 0 or len(Jj) == 0:
                continue
            M.loc[gt, gf] = float(theta_matrix.loc[Ii, Jj].to_numpy().sum())

    if normalize_rows:
        rowsum = M.sum(axis=1)
        rowsum = rowsum.replace(0.0, 1.0)
        M = M.div(rowsum, axis=0)

    return M

# 3) Node-level NET and colors

def compute_net_by_node(theta_matrix: pd.DataFrame) -> pd.Series:
    theta = theta_matrix.values
    diag = np.diag(theta)
    rowsum = theta.sum(axis=1)
    colsum = theta.sum(axis=0)
    FROM = rowsum - diag
    TO = colsum - diag
    NET = TO - FROM
    return pd.Series(NET, index=theta_matrix.index, name="NET")


def _colors_from_net(G: nx.Graph, net_series: pd.Series, eps: float = 1e-12) -> Dict[str, str]:
    colors = {}
    for n in G.nodes():
        v = float(net_series.get(n, 0.0))
        if v > eps:
            colors[n] = "#2CA02C"  # Verde sólido endurecido (Emisor)
        elif v < -eps:
            colors[n] = "#D62728"  # Rojo carmesí endurecido (Receptor)
        else:
            colors[n] = "#bdc3c7"  # Gris Plata (Neutral)
    return colors

def build_group_map(groups: Dict[str, List[str]], universe: Iterable[str]) -> Dict[str, str]:
    uset = set(universe)
    gmap: Dict[str, str] = {}
    for g, members in groups.items():
        for v in members:
            if v in uset:
                gmap[v] = g
    for v in uset:
        if v not in gmap:
            gmap[v] = OTHERS_GROUP
    return gmap


def get_theta_matrix_at(
    cn: Dict[str, object],
    when: str | pd.Timestamp = "last",
    how: str = "pad"
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Extract the θ matrix (to×from) for a specific date from the cn bundle.

    Expects cn["theta"] to be a wide DataFrame with MultiIndex columns (to, from).
    Returns the square matrix θ_t (index=columns=variables in 'to' level, in order) and the resolved timestamp.
    """
    theta_panel: pd.DataFrame = cn["theta"]
    if not isinstance(theta_panel.columns, pd.MultiIndex) or theta_panel.columns.nlevels != 2:
        raise ValueError("cn['theta'] must have MultiIndex columns (to, from).")

    # Resolve timestamp
    if when == "last":
        t = theta_panel.index[-1]
    else:
        t_req = pd.to_datetime(when)
        # Align by label if present; otherwise use pad/backfill/nearest
        if t_req in theta_panel.index:
            t = t_req
        else:
            # Use searchsorted
            idx = theta_panel.index
            if how == "pad":
                pos = idx.searchsorted(t_req, side="right") - 1
            elif how == "backfill":
                pos = idx.searchsorted(t_req, side="left")
            else:  # "nearest"
                pos = idx.searchsorted(t_req)
                if 0 < pos < len(idx):
                    prev = idx[pos - 1]
                    nxt = idx[pos]
                    pos = pos - 1 if abs(t_req - prev) <= abs(nxt - t_req) else pos
            pos = min(max(pos, 0), len(idx) - 1)
            t = idx[pos]

    row = theta_panel.loc[t]
    # Wide → square matrix (to × from); preserve the "to" names order
    to_names = list(dict.fromkeys(theta_panel.columns.get_level_values(0)))
    mat = row.unstack(level=1).reindex(index=to_names, columns=to_names)
    return mat, pd.to_datetime(t)

# 2. PLOT CENTRAL (Preparado para Subfiguras en LaTeX)
def plot_directed_network_with_pos(
    edges_df: pd.DataFrame,
    theta_matrix: pd.DataFrame,
    pos: Dict[str, Tuple[float, float]],
    group_map: Dict[str, str],
    title: str = "Network",
    eps: float = 1e-12,
    net_sig_series: Optional[pd.Series] = None,
    edges_sig_dict: Optional[Dict[Tuple[str, str], bool]] = None,
    show_significance_legend: bool = True,
    ruta_salida: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 8.5),
    node_base_size: float = 2500.0,  # <--- VALOR AUMENTADO POR DEFECTO
    node_mult_size: float = 3000.0,  # <--- VALOR AUMENTADO POR DEFECTO
    font_size: int = 10,
    zoom_margin: float = 0.1,
    draw_legend: bool = True,
    node_labels: Optional[Dict[str, str]] = None,
    label_color: str = "black"  # <--- NUEVO: Control de color de la letra
) -> None:
    
    names = theta_matrix.index.tolist()
    pos_full = {n: pos.get(n, (0.0, 0.0)) for n in names}

    if net_sig_series is None:
        net_sig_series = pd.Series(True, index=names)
    if edges_sig_dict is None:
        edges_sig_dict = {}

    G = nx.DiGraph()
    G.add_nodes_from(names)
    for _, r in edges_df.iterrows():
        src, tgt, w = r["source"], r["target"], float(r["weight"])
        if src in G and tgt in G and w > 0.0:
            G.add_edge(src, tgt, weight=w)

    out_strength = {n: sum(d["weight"] for _, _, d in G.out_edges(n, data=True)) for n in G.nodes()}
    max_out = max(out_strength.values()) if out_strength else 1.0
    
    # APLICACIÓN DE LOS TAMAÑOS
    node_sizes_dict = {n: node_base_size + node_mult_size * (out_strength[n] / max_out if max_out > 0 else 0.0) for n in G.nodes()}
    node_colors_dict = _colors_from_net(G, compute_net_by_node(theta_matrix), eps=eps)

    sig_nodes = [n for n in G.nodes() if net_sig_series.get(n, True)]
    insig_nodes = [n for n in G.nodes() if not net_sig_series.get(n, True)]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    if sig_nodes:
        nx.draw_networkx_nodes(
            G, pos_full, nodelist=sig_nodes,
            node_color=[node_colors_dict[n] for n in sig_nodes],
            node_size=[node_sizes_dict[n] for n in sig_nodes],
            linewidths=0, alpha=0.95, ax=ax
        )
        
    if insig_nodes:
        nodes_collection = nx.draw_networkx_nodes(
            G, pos_full, nodelist=insig_nodes,
            node_color=[node_colors_dict[n] for n in insig_nodes],
            node_size=[node_sizes_dict[n] for n in insig_nodes],
            linewidths=0, edgecolors='white', alpha=0.85, ax=ax
        )
        nodes_collection.set_hatch('////')

    # --- DIBUJO DE ETIQUETAS CON COLOR PERSONALIZADO ---
    labels_to_draw = node_labels if node_labels else {n: n for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos_full, 
        labels=labels_to_draw, 
        font_size=font_size, 
        font_weight='bold', 
        font_color=label_color, # <--- APLICA EL COLOR AQUÍ
        ax=ax
    )

    GROUP_COLORS = {
        "EE.UU.": "#2980b9",         
        "México": "#5fff24",      
        "Canadá": "#f9611bff",      
        "Energéticos": "#fc3bff", 
        "Oro": "#fdc600",        
        OTHERS_GROUP: "#bdc3c7"
    }

    weights = nx.get_edge_attributes(G, 'weight')
    if weights:
        max_w = max(weights.values())
        
        for u, v in G.edges():
            w = weights[(u, v)]
            width = 1.5 + 6.0 * (w / max_w)
            
            src_group = group_map.get(u, OTHERS_GROUP)
            base_hex = GROUP_COLORS.get(src_group, "#bdc3c7")
            
            is_sig = edges_sig_dict.get((u, v), True)
            
            if is_sig:
                rgba_sig = list(mcolors.to_rgba(base_hex))
                rgba_sig[3] = 0.90 
                nx.draw_networkx_edges(
                    G, pos_full, edgelist=[(u, v)], width=width, 
                    edge_color=[tuple(rgba_sig)], style="solid",
                    arrowsize=20, connectionstyle="arc3,rad=0.25", 
                    node_size=node_sizes_dict[v], ax=ax
                )
            else:
                rgba_line = list(mcolors.to_rgba(base_hex))
                rgba_line[3] = 0.30  
                rgba_head = list(mcolors.to_rgba(base_hex))
                rgba_head[3] = 0.90  
                
                nx.draw_networkx_edges(
                    G, pos_full, edgelist=[(u, v)], width=width, 
                    edge_color=[tuple(rgba_line)], style="dashed",
                    arrowstyle="-", connectionstyle="arc3,rad=0.25", 
                    node_size=node_sizes_dict[v], ax=ax
                )
                nx.draw_networkx_edges(
                    G, pos_full, edgelist=[(u, v)], width=0.0,       
                    edge_color=[tuple(rgba_head)], style="solid",
                    arrowsize=20, connectionstyle="arc3,rad=0.25", 
                    node_size=node_sizes_dict[v], ax=ax
                )

    if draw_legend:
        if show_significance_legend:
            legend_handles = [
                mpatches.Patch(facecolor="#2CA02C", label="Emisor Neto (Sig.)"),
                mpatches.Patch(facecolor="#D62728", label="Receptor Neto (Sig.)"),
                mlines.Line2D([0], [0], color="black", lw=2, label="Transmisión Sig."),
                mpatches.Patch(facecolor="#2CA02C", hatch='////', edgecolor='white', linewidth=0, label="Emisor Neto (No Sig.)"),
                mpatches.Patch(facecolor="#D62728", hatch='////', edgecolor='white', linewidth=0, label="Receptor Neto (No Sig.)"),
                mlines.Line2D([0], [0], color="black", lw=2, linestyle='dashed', alpha=0.4, label="Transmisión No Sig."),
            ]
        else:
            legend_handles = [
                mpatches.Patch(facecolor="#2CA02C", label="Emisor Neto"),
                mpatches.Patch(facecolor="#D62728", label="Receptor Neto"),
                mlines.Line2D([0], [0], color="black", lw=2, label="Transmisión Neta"),
            ]
        
        grupos_presentes = set(group_map.values())
        for g, color in GROUP_COLORS.items():
            if g in grupos_presentes:
                legend_handles.append(mlines.Line2D([0], [0], color=color, lw=3, label=f"Origen: {g}"))
        
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.06), 
                  ncol=3, frameon=False, fontsize=11)
    
    ax.margins(zoom_margin)
    ax.axis("off")
    plt.tight_layout()

    if ruta_salida:
        fig.savefig(ruta_salida, format='pdf', bbox_inches='tight')
        print(f"Gráfica guardada en: {ruta_salida}")

    plt.show()

def make_pairwise_net_edges(
    theta_matrix: pd.DataFrame,
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Build a directed edge list from a square θ by using pairwise NET(j→i) = θ_{i←j} − θ_{j←i}.

    - For each unordered pair {i, j}, compute w_ij = θ_{i←j} − θ_{j←i}.
    - If w_ij > threshold: add edge j → i with weight w_ij.
    - If w_ij < −threshold: add edge i → j with weight −w_ij.
    - Otherwise: no edge (ties or small magnitudes are suppressed).

    Returns a DataFrame with columns ["source", "target", "weight"].
    """
    if theta_matrix.index.tolist() != theta_matrix.columns.tolist():
        raise ValueError("theta_matrix must be square with identical index and columns (same order).")
    names = theta_matrix.index.tolist()
    edges: List[Tuple[str, str, float]] = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            i, j = names[a], names[b]
            w_ij = float(theta_matrix.loc[i, j] - theta_matrix.loc[j, i])
            if w_ij > threshold:
                edges.append((j, i, w_ij))  # j → i
            elif w_ij < -threshold:
                edges.append((i, j, -w_ij))  # i → j
            # else: skip
    if not edges:
        return pd.DataFrame(columns=["source", "target", "weight"])
    df = pd.DataFrame(edges, columns=["source", "target", "weight"])
    # Optional: sort by weight descending for diagnostics
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    return df

# =====================================================================
# ENVOLTORIO 1: RED DESAGREGADA (10 Nodos)
# =====================================================================

# 1. Define tu nueva lista (diccionario) con los nombres que prefieras
# Diccionario modificado para que los nombres salgan en dos renglones
nombres_dos_lineas = {
    "spx_usa": "S&P\n(EE.UU.)",
    "spx_canada": "S&P\n(Canadá)",
    "spx_mexico": "S&P\n(México)",
    "crude_oil": "Petróleo",
    "gas": "Gas",
    "gold": "Oro",
    "usa_sov": "Bonos\n(EE.UU.)",
    "mxn_sov": "Bonos\n(México)",
    "fx_canada": "CAD/\nMXN",
    "fx_usa": "USD/\nMXN",
}

def plot_vars_grouped_layout(
    cn: Dict[str, object], 
    ci: Dict[str, object], 
    idx: pd.DatetimeIndex, 
    orden_net: List[str],
    when: str | pd.Timestamp = "last",
    how: str = "pad", 
    threshold: float = 0.0,
    ruta_salida: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 7.5),
    node_spread: float = 0.7,
    node_labels: Optional[Dict[str, str]] = nombres_dos_lineas,
    # --- NUEVOS CONTROLES DE TAMAÑO Y COLOR ---
    node_base_size: float = 3000.0, # Bolas mucho más grandes
    node_mult_size: float = 3000.0, 
    font_size: int = 13,            # Letra un poco más grande
    label_color: str = "#090E5BFF"    # Código Hex para el color de la letra (ej. #333333 para gris oscuro)
) -> None:
    
    theta_matrix, t_idx_real = get_theta_matrix_at(cn, when=when, how=how)
    names = theta_matrix.index.tolist()
    
    try:
        t_pos = idx.get_loc(pd.to_datetime(t_idx_real))
    except KeyError:
        print(f"La fecha {t_idx_real} no se encontró. Usando la más cercana.")
        t_pos = idx.get_indexer([pd.to_datetime(t_idx_real)], method='nearest')[0]

    lo_net = np.asarray(ci["NET"][0])[t_pos, :]
    hi_net = np.asarray(ci["NET"][1])[t_pos, :]
    net_sig_series = pd.Series(False, index=orden_net)
    
    for i, var in enumerate(orden_net):
        if lo_net[i] > 0 or hi_net[i] < 0:
            net_sig_series[var] = True

    matriz_lo_pdc = np.asarray(ci["PDC"][0])[t_pos, :, :]
    matriz_hi_pdc = np.asarray(ci["PDC"][1])[t_pos, :, :]
    edges_sig_dict = {}

    edges_df = make_pairwise_net_edges(theta_matrix, threshold=threshold)

    for _, row in edges_df.iterrows():
        src = row['source']
        tgt = row['target']
        if src in orden_net and tgt in orden_net:
            idx1 = orden_net.index(src)
            idx2 = orden_net.index(tgt)
            if matriz_lo_pdc[idx1, idx2] > 0 or matriz_hi_pdc[idx1, idx2] < 0:
                edges_sig_dict[(src, tgt)] = True
            else:
                edges_sig_dict[(src, tgt)] = False

    gmap = build_group_map(GROUPS, names)
    pos = make_grouped_positions(names, gmap, group_radius=2.0, node_spread=node_spread, seed=1)
    
    plot_directed_network_with_pos(
        edges_df=edges_df, 
        theta_matrix=theta_matrix, 
        pos=pos,
        group_map=gmap, 
        edges_sig_dict=edges_sig_dict,
        net_sig_series=net_sig_series,
        title=f"Red de Derrames Direccionales Netos — {pd.to_datetime(t_idx_real).date()}",
        eps=1e-12,
        show_significance_legend=True,
        ruta_salida=ruta_salida,
        figsize=figsize,
        node_labels=node_labels,
        node_base_size=node_base_size, # Pasa el tamaño base
        node_mult_size=node_mult_size, # Pasa el multiplicador
        font_size=font_size,           # Pasa el tamaño de fuente
        label_color=label_color        # Pasa el color
    )
# =====================================================================
# ENVOLTORIO 2: RED AGREGADA (Ajustada para LaTeX - Proporción 1/3 de hoja)
# =====================================================================
def plot_5node_aggregated(
    cn: Dict[str, object],
    when: str | pd.Timestamp = "last",  
    how: str = "pad",
    threshold: float = 0.0,
    normalize_rows: bool = True,
    ruta_salida: Optional[str] = None
) -> None:
    
    theta_matrix, t_idx_real = get_theta_matrix_at(cn, when=when, how=how)
    names = theta_matrix.index.tolist()
    gmap_vars = build_group_map(GROUPS, names)
    
    ThetaG = aggregate_theta_to_groups(theta_matrix, gmap_vars, normalize_rows=normalize_rows)
    edgesG = make_pairwise_net_edges(ThetaG, threshold=threshold)

    group_names = ThetaG.index.tolist()
    gmap_identity = {g: g for g in group_names}
    
    posG = make_grouped_positions(
        group_names, 
        gmap_identity,
        group_radius=2.5, 
        node_spread=0.0, 
        seed=1,
        y_squash=0.70  
    )

    plot_directed_network_with_pos(
        edges_df=edgesG, 
        theta_matrix=ThetaG, 
        pos=posG,
        group_map=gmap_identity, 
        eps=1e-12,
        figsize=(8, 4.5),          
        draw_legend=True,         
        show_significance_legend=False, 
        title="",                  
        node_base_size=4000.0,     
        node_mult_size=4500.0,
        font_size=10,               
        zoom_margin=0.20,  
        ruta_salida=ruta_salida,
        label_color="#FFFFFF"
        # NO PASAMOS node_labels AQUÍ para que mantenga los nombres de grupo ("EE.UU.", "México", etc.)
    )

# 8) GIF with fixed positions over time


def _fig_to_rgb_array(fig: plt.Figure, dpi: int = 120) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(h, w, 4)[..., :3].copy()  
    except Exception:
        from PIL import Image  
        bio = io.BytesIO()
        fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight")
        bio.seek(0)
        img = np.array(Image.open(bio).convert("RGB"))
    plt.close(fig)
    return img

def _draw_frame(
    theta_matrix: pd.DataFrame,
    pos: Dict[str, Tuple[float, float]],
    group_map: Dict[str, str],
    threshold: float,
    max_w_global: float,
    title: str,
    net_sig_series: Optional[pd.Series] = None,          
    edges_sig_dict: Optional[Dict[Tuple[str, str], bool]] = None, 
    show_legend: bool = True,
    dpi: int = 120,
    node_labels: Optional[Dict[str, str]] = None,
    label_color: str = "black",
    node_base_size: float = 2500.0,
    node_mult_size: float = 3000.0,
    font_size: int = 10,
    # --- NUEVOS PARÁMETROS PARA CONGELAR LA CÁMARA ---
    fixed_xlim: Optional[Tuple[float, float]] = None,
    fixed_ylim: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    
    names = theta_matrix.index.tolist()
    if net_sig_series is None:
        net_sig_series = pd.Series(True, index=names)
    if edges_sig_dict is None:
        edges_sig_dict = {}

    edges = make_pairwise_net_edges(theta_matrix, threshold=threshold)

    G = nx.DiGraph()
    G.add_nodes_from(names)
    for _, r in edges.iterrows():
        G.add_edge(r["source"], r["target"], weight=float(r["weight"]))

    out_strength = {n: sum(d["weight"] for _, _, d in G.out_edges(n, data=True)) for n in G.nodes()}
    max_out = max(out_strength.values()) if out_strength else 1.0
    
    node_sizes_dict = {n: node_base_size + node_mult_size * (out_strength[n] / max_out if max_out > 0 else 0.0) for n in G.nodes()}
    node_colors_dict = _colors_from_net(G, compute_net_by_node(theta_matrix), eps=1e-12)

    sig_nodes = [n for n in G.nodes() if net_sig_series.get(n, True)]
    insig_nodes = [n for n in G.nodes() if not net_sig_series.get(n, True)]

    fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)

    if sig_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=sig_nodes,
            node_color=[node_colors_dict[n] for n in sig_nodes],
            node_size=[node_sizes_dict[n] for n in sig_nodes],
            linewidths=0, alpha=0.95, ax=ax
        )
        
    if insig_nodes:
        nodes_collection = nx.draw_networkx_nodes(
            G, pos, nodelist=insig_nodes,
            node_color=[node_colors_dict[n] for n in insig_nodes],
            node_size=[node_sizes_dict[n] for n in insig_nodes],
            linewidths=0, edgecolors='white', alpha=0.85, ax=ax
        )
        nodes_collection.set_hatch('////')

    labels_to_draw = node_labels if node_labels else {n: n for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, 
        labels=labels_to_draw, 
        font_size=font_size, 
        font_weight='bold', 
        font_color=label_color, 
        ax=ax
    )

    GROUP_COLORS = {
        "EE.UU.": "#2980b9",         
        "México": "#5fff24",      
        "Canadá": "#f9611bff",      
        "Energéticos": "#fc3bff", 
        "Oro": "#fdc600",        
        OTHERS_GROUP: "#bdc3c7"
    }

    weights = nx.get_edge_attributes(G, 'weight')
    if weights:
        max_divisor = max(1e-12, max_w_global)
        
        for u, v in G.edges():
            w = weights[(u, v)]
            width = 1.5 + 6.0 * (w / max_divisor)
            
            src_group = group_map.get(u, OTHERS_GROUP)
            base_hex = GROUP_COLORS.get(src_group, "#bdc3c7")
            rgba = list(mcolors.to_rgba(base_hex))
            
            is_sig = edges_sig_dict.get((u, v), True)
            edge_style = "solid" if is_sig else "dashed"
            rgba[3] = 0.85 if is_sig else 0.20 
            
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=[(u, v)],
                width=width, 
                edge_color=[tuple(rgba)],
                style=edge_style,
                arrowsize=20, 
                connectionstyle="arc3,rad=0.25", 
                node_size=node_sizes_dict[v], 
                ax=ax
            )

    if show_legend:
        legend_handles = [
            mpatches.Patch(facecolor="#2CA02C", label="Emisor Neto (Sig.)"),
            mpatches.Patch(facecolor="#D62728", label="Receptor Neto (Sig.)"),
            mlines.Line2D([0], [0], color="black", lw=2, label="Transmisión Sig."),
            mpatches.Patch(facecolor="#2CA02C", hatch='////', edgecolor='white', linewidth=0, label="Emisor Neto (No Sig.)"),
            mpatches.Patch(facecolor="#D62728", hatch='////', edgecolor='white', linewidth=0, label="Receptor Neto (No Sig.)"),
            mlines.Line2D([0], [0], color="black", lw=2, linestyle='dashed', alpha=0.4, label="Transmisión No Sig."),
        ]
        grupos_presentes = set(group_map.values())
        for g, color in GROUP_COLORS.items():
            if g in grupos_presentes:
                legend_handles.append(mlines.Line2D([0], [0], color=color, lw=3, label=f"Origen: {g}"))
        
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.05), 
                  ncol=3, frameon=False, fontsize=10)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    # =======================================================
    # LA MAGIA ESTÁ AQUÍ: Congelar los ejes independientemente
    # del grosor de las flechas que intente dibujar networkx
    # =======================================================
    if fixed_xlim:
        ax.set_xlim(fixed_xlim)
    if fixed_ylim:
        ax.set_ylim(fixed_ylim)
        
    ax.axis("off")
    # Quitamos tight_layout que es el enemigo número 1 de las animaciones estables
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.15)

    return _fig_to_rgb_array(fig, dpi=dpi)

def animate_full_network(
    cn: Dict[str, object],
    ci: Dict[str, object],                   
    idx_ref: pd.DatetimeIndex,               
    orden_net: List[str],                    
    when_list: Optional[Iterable[pd.Timestamp | str]] = None,
    threshold: float = 0.0,
    step: int = 1,
    outfile: str = "connectedness.gif",
    dpi: int = 100,  
    duration: float = 0.3,
    node_labels: Optional[Dict[str, str]] = None,
    label_color: str = "#090E5BFF",
    node_base_size: float = 3000.0,
    node_mult_size: float = 3000.0,
    font_size: int = 13
) -> None:
    
    import imageio.v2 as imageio

    theta_panel: pd.DataFrame = cn["theta"]
    idx_all = theta_panel.index if when_list is None else pd.DatetimeIndex(when_list)
    idx_all = idx_all[::max(1, step)]

    names = list(dict.fromkeys(theta_panel.columns.get_level_values(0)))
    gmap = build_group_map(GROUPS, names)
    
    pos = make_grouped_positions(names, gmap, group_radius=2.8, node_spread=0.85, seed=1) 

    # --- CÁLCULO DE LA CÁMARA (Bounding Box Estático) ---
    xs = [coord[0] for coord in pos.values()]
    ys = [coord[1] for coord in pos.values()]
    
    # 25% de margen extra alrededor de los nodos más extremos
    x_margin = (max(xs) - min(xs)) * 0.25
    y_margin = (max(ys) - min(ys)) * 0.25
    
    global_xlim = (min(xs) - x_margin, max(xs) + x_margin)
    global_ylim = (min(ys) - y_margin, max(ys) + y_margin)

    max_w = 0.0
    for t in idx_all:
        row = theta_panel.loc[t]
        theta_t = row.unstack(level=1).reindex(index=names, columns=names)
        edges = make_pairwise_net_edges(theta_t, threshold=threshold)
        if not edges.empty:
            max_w = max(max_w, float(edges["weight"].max()))
    if max_w <= 0.0:
        max_w = 1.0

    matriz_lo_net = np.asarray(ci["NET"][0])
    matriz_hi_net = np.asarray(ci["NET"][1])
    matriz_lo_pdc = np.asarray(ci["PDC"][0])
    matriz_hi_pdc = np.asarray(ci["PDC"][1])

    frames: List[np.ndarray] = []
    total = len(idx_all)
    
    print(f"Iniciando renderizado con cruce de significancia para {total} fotogramas...")
    
    for i, t in enumerate(idx_all):
        row = theta_panel.loc[t]
        theta_t = row.unstack(level=1).reindex(index=names, columns=names)
        title = f"Red Direccional (Desagregada) — {pd.to_datetime(t).date()}"
        
        try:
            t_pos = idx_ref.get_loc(pd.to_datetime(t))
        except KeyError:
            t_pos = idx_ref.get_indexer([pd.to_datetime(t)], method='nearest')[0]

        lo_net_t = matriz_lo_net[t_pos, :]
        hi_net_t = matriz_hi_net[t_pos, :]
        net_sig_series = pd.Series(False, index=orden_net)
        for j, var in enumerate(orden_net):
            if lo_net_t[j] > 0 or hi_net_t[j] < 0:
                net_sig_series[var] = True

        lo_pdc_t = matriz_lo_pdc[t_pos, :, :]
        hi_pdc_t = matriz_hi_pdc[t_pos, :, :]
        edges_sig_dict = {}
        edges_t = make_pairwise_net_edges(theta_t, threshold=threshold)
        
        for _, row_e in edges_t.iterrows():
            src = row_e['source']
            tgt = row_e['target']
            if src in orden_net and tgt in orden_net:
                idx1 = orden_net.index(src)
                idx2 = orden_net.index(tgt)
                if lo_pdc_t[idx1, idx2] > 0 or hi_pdc_t[idx1, idx2] < 0:
                    edges_sig_dict[(src, tgt)] = True
                else:
                    edges_sig_dict[(src, tgt)] = False

        frame = _draw_frame(
            theta_matrix=theta_t, 
            pos=pos, 
            group_map=gmap, 
            threshold=threshold, 
            max_w_global=max_w, 
            title=title, 
            net_sig_series=net_sig_series,       
            edges_sig_dict=edges_sig_dict,       
            show_legend=True, 
            dpi=dpi,
            node_labels=node_labels,
            label_color=label_color,
            node_base_size=node_base_size,
            node_mult_size=node_mult_size,
            font_size=font_size,
            fixed_xlim=global_xlim, # <-- Congelando Eje X
            fixed_ylim=global_ylim  # <-- Congelando Eje Y
        )
        frames.append(frame)
        
        if (i+1) % 50 == 0:
            print(f"Renderizando frame {i+1}/{total}...")

    print("Construyendo archivo GIF...")
    imageio.mimsave(outfile, frames, duration=duration, loop=0)
    print(f"¡Éxito! GIF con significancia dinámica guardado en: {outfile}")

def animate_grouped5_network(
    cn: dict,
    when_list: Optional[Iterable[pd.Timestamp | str]] = None,
    threshold: float = 0.0,
    step: int = 1,
    outfile: str = "connectedness_groups.gif",
    dpi: int = 100,
    duration: float = 0.3,
    normalize_rows: bool = True,
) -> None:
    
    import imageio.v2 as imageio

    theta_panel: pd.DataFrame = cn["theta"]
    idx_all = theta_panel.index if when_list is None else pd.DatetimeIndex(when_list)
    idx_all = idx_all[::max(1, step)]

    var_names = list(dict.fromkeys(theta_panel.columns.get_level_values(0)))
    gmap_vars = build_group_map(GROUPS, var_names)

    def _grouped_theta_at(tstamp) -> pd.DataFrame:
        row = theta_panel.loc[tstamp]
        theta_t = row.unstack(level=1).reindex(index=var_names, columns=var_names)
        return aggregate_theta_to_groups(theta_t, gmap_vars, normalize_rows=normalize_rows)

    ThetaG0 = None
    for t in idx_all:
        ThetaG0 = _grouped_theta_at(t)
        if ThetaG0 is not None and ThetaG0.size > 0:
            break
    if ThetaG0 is None or ThetaG0.size == 0:
        raise RuntimeError("No frames disponibles.")

    group_names = ThetaG0.index.tolist()
    gmap_identity = {g: g for g in group_names}
    
    posG = make_grouped_positions(group_names, gmap_identity, group_radius=3.0, node_spread=0.0, seed=1, y_squash=0.70)

    # --- CÁLCULO DE LA CÁMARA (Bounding Box Estático para la red agregada) ---
    xs = [coord[0] for coord in posG.values()]
    ys = [coord[1] for coord in posG.values()]
    
    x_margin = (max(xs) - min(xs)) * 0.25 if max(xs) > min(xs) else 2.0
    y_margin = (max(ys) - min(ys)) * 0.25 if max(ys) > min(ys) else 2.0
    
    global_xlim = (min(xs) - x_margin, max(xs) + x_margin)
    global_ylim = (min(ys) - y_margin, max(ys) + y_margin)

    max_w = 0.0
    for t in idx_all:
        ThetaGt = _grouped_theta_at(t).reindex(index=group_names, columns=group_names)
        edgesG = make_pairwise_net_edges(ThetaGt, threshold=threshold)
        if not edgesG.empty:
            max_w = max(max_w, float(edgesG["weight"].max()))
    if max_w <= 0.0:
        max_w = 1.0

    frames: List[np.ndarray] = []
    total = len(idx_all)
    for i, t in enumerate(idx_all):
        ThetaGt = _grouped_theta_at(t).reindex(index=group_names, columns=group_names)
        title = f"Red Direccional (Bloques Agregados) — {pd.to_datetime(t).date()}"
        
        frame = _draw_frame(
            theta_matrix=ThetaGt, 
            pos=posG, 
            group_map=gmap_identity, 
            threshold=threshold, 
            max_w_global=max_w, 
            title=title, 
            show_legend=True, 
            dpi=dpi,
            node_base_size=4000.0,
            node_mult_size=4500.0,
            font_size=10,
            label_color="#FFFFFF",
            fixed_xlim=global_xlim, # <-- Congelando Eje X
            fixed_ylim=global_ylim  # <-- Congelando Eje Y
        )
        frames.append(frame)
        
        if (i+1) % 50 == 0:
            print(f"Renderizando frame {i+1}/{total}...")

    print("Construyendo archivo GIF...")
    imageio.mimsave(outfile, frames, duration=duration, loop=0)
    print(f"¡Éxito! GIF guardado en: {outfile}")