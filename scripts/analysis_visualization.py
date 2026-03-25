"""
Statistical analysis and visualization - Pico Humboldt Glacier

Exponential model on dry years (n=6) with sub-pixel unmixing data.
Theoretical basis: Huss & Fischer (2016) - retreat proportional to area.

Thresholds:
  - Reclassification (0.005 km²): glacier → ice patch
  - Disappearance (0.0001 km²): below Sentinel-2 detection limit

Outputs:
  - 4 PNG plots (300 DPI)
  - Comparative RGB image 2020 vs 2025
  - 3 CSV files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.lines import Line2D
import rasterio
import warnings
import geopandas as gpd

warnings.filterwarnings('ignore', message='overflow encountered in exp')
warnings.filterwarnings('ignore', message='invalid value encountered')
warnings.filterwarnings('ignore', message='This figure includes Axes')

# ============================================================================
# MODEL DEFINITION
# ============================================================================

def exp_model(t, a, k, t0):
    """Exponential decay: A(t) = a·exp(k·(t-t0))"""
    return a * np.exp(k * (t - t0))

def project_threshold(func, popt, threshold, t0, max_year=2080):
    """Calculate year when model reaches a threshold area."""
    t_range = np.arange(t0, max_year, 0.1)
    areas = func(t_range, *popt)
    crossings = np.where(areas <= threshold)[0]
    return t_range[crossings[0]] if len(crossings) > 0 else None

def plot_polygon(ax, gdf, color, linewidth, alpha=0.9):
    """Draw polygons (Polygon or MultiPolygon) on an axes."""
    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, linewidth=linewidth, linestyle='-', alpha=alpha)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=color, linewidth=linewidth, linestyle='-', alpha=alpha)

# ============================================================================
# VISUAL CONFIGURATION
# ============================================================================

sns.set_theme(style='ticks', context='paper', font_scale=1.1)
sns.set_palette('deep')

COLORS = {
    'wet': '#3498db', 'dry': '#e74c3c', 'trend': '#f39c12',
    'reference': '#27ae60', 'neutral': "#a6aaab", 'black': "#2C2A2A"
}

plt.rcParams.update({
    'figure.dpi': 300, 'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.alpha': 0.3, 'grid.linestyle': '--', 'grid.linewidth': 0.5
})

Path('results/plots').mkdir(parents=True, exist_ok=True)

print('='*70)
print('STATISTICAL ANALYSIS - PICO HUMBOLDT GLACIER')
print('='*70)

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv('data/snow_stats_2015_2026.csv')
df['ndsi_cv'] = (df['ndsi_std'] / df['ndsi_mean']) * 100

try:
    with open('data/processing_metadata.json', 'r') as f:
        metadata = json.load(f)
    omitted_years = metadata.get('years_excluded', [])
except FileNotFoundError:
    omitted_years = []
    metadata = {}

print(f'\nData loaded: {len(df)} years ({df["year"].min()}-{df["year"].max()})')
if omitted_years:
    print(f'Omitted years: {omitted_years}')

has_binary = 'binary_area_km2' in df.columns
has_ndsi = 'ndsi_area_km2' in df.columns

# ============================================================================
# YEAR CLASSIFICATION AND KEY METRICS
# ============================================================================

initial_year = int(df['year'].min())
final_year = int(df['year'].max())
minimum_year = int(df.loc[df['snow_area_km2'].idxmin(), 'year'])
minimum_area = df['snow_area_km2'].min()

dry_threshold = 0.060
df['year_type'] = df['snow_area_km2'].apply(lambda x: 'Wet' if x > dry_threshold else 'Dry')
dry_years = df[df['year_type'] == 'Dry']
wet_years = df[df['year_type'] == 'Wet']

print(f'\nClassification (threshold: {dry_threshold} km²):')
print(f'  Wet ({len(wet_years)}): {wet_years["year"].tolist()}')
print(f'  Dry ({len(dry_years)}): {dry_years["year"].tolist()}')

# ============================================================================
# VALIDATION AGAINST RAMÍREZ
# ============================================================================

ramirez_data = {
    2018: {'area': 0.079, 'error': 0.005},
    2019: {'area': 0.046, 'error': 0.004}
}

validation = []
for year, data in ramirez_data.items():
    entry = {'year': year, 'ramirez_area_km2': data['area'],
             'fractional_area_km2': None, 'diff_pct': None, 'status': 'NO_DATA'}
    if year in df['year'].values:
        frac = df.loc[df['year'] == year, 'snow_area_km2'].values[0]
        entry['fractional_area_km2'] = round(frac, 4)
        entry['diff_pct'] = round(((frac - data['area']) / data['area']) * 100, 1)
        entry['status'] = 'Includes seasonal snow' if entry['diff_pct'] > 50 else 'Comparable'
    validation.append(entry)

pd.DataFrame(validation).to_csv('results/validation_ramirez2020.csv', index=False)

# ============================================================================
# EXPONENTIAL MODEL (DRY YEARS ONLY)
# ============================================================================

print(f'\nFitting exponential model on dry years (n={len(dry_years)})...')

df_dry = dry_years.copy()
initial_year_dry = int(df_dry['year'].min())

popt, pcov = curve_fit(exp_model, df_dry['year'].values, df_dry['snow_area_km2'].values,
                       p0=[df_dry['snow_area_km2'].iloc[0], -0.2, initial_year_dry], maxfev=10000)

y_pred = exp_model(df_dry['year'].values, *popt)
r2 = r2_score(df_dry['snow_area_km2'].values, y_pred)
rmse = np.sqrt(mean_squared_error(df_dry['snow_area_km2'].values, y_pred))
rate_pct = popt[1] * 100
confidence_interval = 1.96 * rmse

print(f'  R²={r2:.4f}, RMSE={rmse:.4f}, Rate={rate_pct:.2f}%/year')
print(f'  95% CI: ±{confidence_interval:.4f} km²')

# ============================================================================
# PROJECTIONS
# ============================================================================

RECLASSIFICATION_THRESHOLD = 0.005
DISAPPEARANCE_THRESHOLD = 0.0001

reclass_year = project_threshold(exp_model, popt, RECLASSIFICATION_THRESHOLD, initial_year_dry)
disappear_year = project_threshold(exp_model, popt, DISAPPEARANCE_THRESHOLD, initial_year_dry)

if reclass_year:
    print(f'  Reclassification (glacier → ice patch): ~{int(reclass_year)}')
if disappear_year:
    print(f'  Disappearance (below detection): ~{int(disappear_year)}')

plot_year_limit = max(int(disappear_year) + 5, 2035) if disappear_year and disappear_year <= 2060 else 2045

pd.DataFrame([{
    'model': 'Exponential', 'params': 3, 'r2': round(r2, 4),
    'rmse': round(rmse, 6), 'rate_pct_year': round(rate_pct, 2),
    'reclassification_year': int(reclass_year) if reclass_year else None,
    'disappearance_year': int(disappear_year) if disappear_year else None,
    'physical_basis': 'Retreat rate proportional to current area (Huss & Fischer, 2016)'
}]).to_csv('results/model_comparison.csv', index=False)

# ============================================================================
# COMPARATIVE IMAGE 2020 vs 2025
# ============================================================================

area_2020 = area_2025 = loss_pct_img = None
try:
    if 2020 in df['year'].values and 2025 in df['year'].values:
        rgb_path = 'data/2025_rgb.tif'
        poly_2020_path = 'data/glacier_polygons/2020.geojson'
        poly_2025_path = 'data/glacier_polygons/2025.geojson'
        if Path(rgb_path).exists() and Path(poly_2020_path).exists() and Path(poly_2025_path).exists():
            with rasterio.open(rgb_path) as src:
                rgb = src.read([1, 2, 3]); bounds = src.bounds
                rgb = np.transpose(rgb, (1, 2, 0))
                if rgb.max() > 255: rgb = ((rgb / rgb.max()) * 255).astype(np.uint8)
            gdf_2020 = gpd.read_file(poly_2020_path)
            gdf_2025 = gpd.read_file(poly_2025_path)
            area_2020 = df.loc[df['year'] == 2020, 'snow_area_km2'].values[0]
            area_2025 = df.loc[df['year'] == 2025, 'snow_area_km2'].values[0]
            loss_pct_img = ((area_2020 - area_2025) / area_2020) * 100

            fig, ax = plt.subplots(figsize=(16, 10))
            ax.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
            plot_polygon(ax, gdf_2020, '#e74c3c', 1)
            plot_polygon(ax, gdf_2025, '#3498db', 0.9)
            text_x, text_y_start = 0.98, 0.7
            ax.text(text_x, text_y_start, 'RETROCESO GLACIAR\n2020-2025',
                   transform=ax.transAxes, fontsize=14, fontweight='bold', color='white', ha='right', va='top')
            ax.text(text_x, text_y_start - 0.05, f'{loss_pct_img:.1f}%',
                   transform=ax.transAxes, fontsize=28, fontweight='bold', color='white', ha='right', va='top')
            stats_text = (f'\nÁrea 2020: {area_2020:.4f} km²\n'
                         f'Área 2025: {area_2025:.4f} km²\n'
                         f'Pérdida: {area_2020 - area_2025:.4f} km²\n\n'
                         f'Tasa anual: {loss_pct_img/5:.1f}%/año')
            ax.text(text_x, text_y_start - 0.20, stats_text, transform=ax.transAxes, fontsize=12,
                   color='white', ha='right', va='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.75))
            ax.legend(handles=[Line2D([0],[0],color='#e74c3c',lw=2,label='Glaciar 2020'),
                               Line2D([0],[0],color='#3498db',lw=2,label='Glaciar 2025')],
                     loc='lower right', fontsize=11, frameon=True, facecolor='white', edgecolor='black', framealpha=0.9)
            ax.set_xlabel(''); ax.set_ylabel('')
            ax.set_title('Pico Humboldt - Venezuela', fontsize=18, fontweight='bold', pad=15)
            plt.tight_layout()
            logo_path = 'assets/logo_leo.png'
            if Path(logo_path).exists():
                try:
                    from PIL import Image; from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                    logo = Image.open(logo_path)
                    if logo.mode != 'RGBA': logo = logo.convert('RGBA')
                    logo_ax = inset_axes(ax, width="5%", height="5%", loc='lower left', borderpad=1)
                    logo_ax.imshow(np.array(logo)); logo_ax.axis('off')
                except Exception: pass
            plt.savefig('results/plots/comparison_2020_2025.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f'\nComparative image: {loss_pct_img:.1f}% loss ({area_2020:.4f} → {area_2025:.4f} km²)')
except Exception as e:
    print(f'\nError generating comparative image: {e}')

# ============================================================================
# PLOTS
# ============================================================================

print('\nGenerating plots...')

# ── PLOT 1: Temporal evolution + model + Ramírez ─────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for _, row in df.iterrows():
    color = COLORS['wet'] if row['year_type'] == 'Wet' else COLORS['dry']
    marker = 'o' if row['year_type'] == 'Wet' else 's'
    ax.scatter(row['year'], row['snow_area_km2'], s=60, color=color,
              marker=marker, linewidth=0, zorder=3, alpha=0.8)

ax.plot(df['year'], df['snow_area_km2'],
       color=COLORS['neutral'], linewidth=1, alpha=0.4, zorder=1)

years_fit = np.linspace(initial_year_dry, final_year, 100)
ax.plot(years_fit, exp_model(years_fit, *popt), '-', color=COLORS['trend'],
       linewidth=2, label=f'Modelo Exponencial (R²={r2:.2f}, {rate_pct:.1f}%/año)', zorder=2)

ramirez_years = [2018, 2019]
ramirez_areas = [0.079, 0.046]
ax.scatter(ramirez_years, ramirez_areas, s=70, marker='D', color=COLORS['reference'],
          label='Ramírez et al. 2020', zorder=4, alpha=0.9, edgecolors='white', linewidth=0.5)

ax.scatter([], [], s=50, color=COLORS['wet'], marker='o', label='Años húmedos (excluídos)')
ax.scatter([], [], s=50, color=COLORS['dry'], marker='s', label='Años secos')

ax.set_xlabel('Año')
ax.set_ylabel('Cobertura de nieve (km²)')
ax.set_title(f'Evolución Temporal - Glaciar Pico Humboldt ({initial_year}-{final_year})', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

sns.despine()
plt.tight_layout()
plt.savefig('results/plots/01_time_series.png', dpi=300, bbox_inches='tight')
print('  01_time_series.png')
plt.close()

# ── PLOT 2: Variability + Anomalies ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

df_sorted = df.sort_values('year')
changes = df_sorted['snow_area_km2'].diff()
changes_pct = (changes / df_sorted['snow_area_km2'].shift(1)) * 100
change_years = df_sorted['year'][1:]
change_colors = [COLORS['dry'] if x < 0 else COLORS['reference'] for x in changes[1:]]

bars = ax1.bar(change_years, changes[1:], color=change_colors,
               edgecolor='white', linewidth=0.5, alpha=0.8)
for bar, pct in zip(bars, changes_pct[1:]):
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h, f'{pct:.0f}%',
            ha='center', va='bottom' if h > 0 else 'top', fontsize=8, alpha=0.7)

ax1.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax1.set_ylabel('Cambio anual (km²)')
ax1.set_title('Variabilidad Interanual', fontweight='bold')
ax1.grid(True, alpha=0.3)

residuals_all = df['snow_area_km2'] - exp_model(df['year'], *popt)
residual_colors = [COLORS['wet'] if r > 0 else COLORS['dry'] for r in residuals_all]
ax2.bar(df['year'], residuals_all, color=residual_colors,
       edgecolor='white', linewidth=0.5, alpha=0.8)
ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax2.set_xlabel('Año')
ax2.set_ylabel('Desviación (km²)')
ax2.set_title('Anomalías Respecto al Modelo Exponencial', fontweight='bold')
ax2.grid(True, alpha=0.3)

for year, res in zip(df['year'], residuals_all):
    if abs(res) > 0.005:
        ax2.text(year, res, f'{res:+.4f}', ha='center',
               va='bottom' if res > 0 else 'top', fontsize=7, alpha=0.7)

sns.despine()
plt.tight_layout()
plt.savefig('results/plots/02_annual_variability.png', dpi=300, bbox_inches='tight')
print('  02_annual_variability.png')
plt.close()

# ── PLOT 3: Projection ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))

all_years_range = list(range(initial_year, plot_year_limit + 1))
ax.set_xticks(all_years_range[::2])
ax.set_xticklabels([str(a) for a in all_years_range[::2]], rotation=45, ha='right')

ax.scatter(df_dry['year'], df_dry['snow_area_km2'], s=60, color=COLORS['dry'],
          marker='s', label='Años normales', zorder=6, alpha=0.9)
ax.scatter(wet_years['year'], wet_years['snow_area_km2'], s=40, color=COLORS['wet'],
          marker='o', label='Años húmedos (excluídos)', zorder=5, alpha=0.5)

years_extended = np.arange(initial_year_dry, plot_year_limit + 1, 0.1)
areas_proj = exp_model(years_extended, *popt)

mask_hist = years_extended <= final_year
mask_fut = years_extended >= final_year

ax.plot(years_extended[mask_hist], areas_proj[mask_hist], '-', color=COLORS['trend'],
       linewidth=2.5, label=f'Modelos Exponencial (R²={r2:.2f})', zorder=3)
ax.plot(years_extended[mask_fut], areas_proj[mask_fut], '--', color=COLORS['trend'],
       linewidth=2.5, alpha=0.7, label='Proyección', zorder=3)

ax.axhline(RECLASSIFICATION_THRESHOLD, color='gray', linestyle=':', linewidth=1, alpha=0.6, zorder=1)
if reclass_year:
    ax.annotate(f'~{int(reclass_year)} (±2)\nReclasificación\n(glaciar → parche de hielo)',
               xy=(reclass_year, RECLASSIFICATION_THRESHOLD),
               xytext=(reclass_year + 2, RECLASSIFICATION_THRESHOLD + 0.008),
               fontsize=7, color=COLORS['black'], fontweight='bold', ha='center', va='bottom',
               arrowprops=dict(arrowstyle='-', color=COLORS['neutral'], lw=1.5, linestyle=':', alpha=0.6))

ax.axhline(DISAPPEARANCE_THRESHOLD, color='darkred', linestyle=':', linewidth=1, alpha=0.3, zorder=1)
if disappear_year:
    ax.annotate(f'~{int(disappear_year)} (±3)\nDesaparición\n(no detectable)',
               xy=(disappear_year, DISAPPEARANCE_THRESHOLD),
               xytext=(disappear_year, 0.006),
               fontsize=7, color='darkred', fontweight='bold', ha='center', va='bottom',
               arrowprops=dict(arrowstyle='-', color='darkred', lw=1.5, linestyle=':', alpha=0.4))

ax.set_xlabel('Año')
ax.set_ylabel('Área (km²)')
ax.set_title('Proyección sobre la Desaparición del Glaciar', fontweight='bold')
ax.legend(
    loc='upper right',
    fontsize=7.5,
    frameon=True,
    facecolor='white',
    edgecolor='gray'
)
y_max = max(df_dry['snow_area_km2'].max(), 0.06)
ax.set_xlim(initial_year - 1, plot_year_limit)
ax.set_ylim(-0.002, y_max * 1.15)
ax.grid(True, alpha=0.3)

sns.despine()
plt.tight_layout()
plt.savefig('results/plots/03_proyection.png', dpi=300, bbox_inches='tight')
print('  03_proyection.png')
plt.close()

# ── PLOT 4: Method comparison (NDSI binary vs FCLSU unmixing) ─────────
# CHANGED: Now compares the traditional NDSI method against the unmixing
# method, showing the real improvement. Bottom panel shows overestimation
# of each method relative to Ramírez et al. (2020) reference (0.046 km²).
if has_ndsi:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    
    x = np.arange(len(df))
    width = 0.25
    
   # Top: area by method
    ax1.bar(x - width, df['ndsi_area_km2'], width,
            label='NDSI binario (≥ 0.4)', color=COLORS['wet'], alpha=0.85)
    ax1.bar(x, df['binary_area_km2'], width,
            label='Unmixing binario', color=COLORS['reference'], alpha=0.85)
    ax1.bar(x + width, df['snow_area_km2'], width,
            label='FCLSU unmixing', color=COLORS['trend'], alpha=0.85)

    ax1.set_ylabel('Area (km²)')
    ax1.set_title('Comparación de Métodos: NDSI vs FCLSU: unmixing and binario', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['year'].astype(int))
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: % difference of each method vs Ramírez reference (2020 only has ref)
    # For all years: show unmixing reduction relative to NDSI
    ndsi_vs_unmix = ((df['snow_area_km2'] - df['ndsi_area_km2']) / df['ndsi_area_km2']) * 100
    
    bar_colors = [COLORS['dry'] if d < 0 else COLORS['reference'] for d in ndsi_vs_unmix]
    ax2.bar(x, ndsi_vs_unmix, color=bar_colors, alpha=0.75,
            edgecolor='white', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    
    mean_reduction = ndsi_vs_unmix.mean()
    ax2.axhline(mean_reduction, color=COLORS['neutral'], linewidth=1, linestyle='--',
               alpha=1, label=f'Diferencia promedio FCLSU/NDSI: {mean_reduction:.1f}%')
    
    for i, (yr, diff) in enumerate(zip(df['year'], ndsi_vs_unmix)):
        ax2.text(i, diff, f'{diff:.1f}%', ha='center',
                va='bottom' if diff > 0 else 'top', fontsize=8, alpha=0.8)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Unmixing vs NDSI (%)')
    ax2.set_title('Corrección del Unmixing FCLSU respecto al NDSI binario',
                 fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['year'].astype(int))
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('results/plots/04_method_comparision.png', dpi=300, bbox_inches='tight')
    print('  04_method_comparision.png')
    plt.close()

# ============================================================================
# UPDATE METADATA
# ============================================================================

metadata['model'] = {
    'type': 'exponential',
    'equation': f'A(t) = {popt[0]:.6f} × exp({popt[1]:.6f} × (t - {int(popt[2])}))',
    'r2': round(float(r2), 4),
    'rmse': round(float(rmse), 6),
    'rate_pct_year': round(float(rate_pct), 2),
    'n_dry_years': int(len(df_dry)),
    'years_used': df_dry['year'].tolist(),
    'years_excluded_wet': wet_years['year'].tolist(),
    'parameters': {'a': float(popt[0]), 'k': float(popt[1]), 't0': int(popt[2])},
    'physical_basis': 'Retreat rate proportional to current area (Huss & Fischer, 2016)',
    'uncertainty': {
        'model_rmse_km2': round(float(rmse), 4),
        'confidence_interval_95pct_km2': round(float(confidence_interval), 4)
    },
    'projections': {
        'reclassification': {
            'year': round(float(reclass_year), 1) if reclass_year else None,
            'threshold_km2': float(RECLASSIFICATION_THRESHOLD),
            'definition': 'Glacier → ice patch (Huss & Fischer, 2016)',
            'uncertainty_years': 2
        },
        'disappearance': {
            'year': round(float(disappear_year), 1) if disappear_year else None,
            'threshold_km2': float(DISAPPEARANCE_THRESHOLD),
            'definition': 'Below Sentinel-2 detection limit (~1 pixel at 20m)',
            'uncertainty_years': 3
        }
    }
}

if area_2020 is not None and area_2025 is not None:
    metadata['comparative_image'] = {
        'file': 'results/plots/comparison_2020_2025.png',
        'area_2020_km2': round(float(area_2020), 6),
        'area_2025_km2': round(float(area_2025), 6),
        'loss_km2': round(float(area_2020 - area_2025), 6),
        'loss_percent': round(float(loss_pct_img), 2),
        'annual_rate_percent': round(float(loss_pct_img / 5), 2),
        'time_span_years': 5
    }

with open('data/processing_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# ============================================================================
# EXPORT METRICS
# ============================================================================

# CHANGED: include ndsi_area_km2 in export
export_cols = ['year', 'snow_area_km2', 'snow_pct', 'year_type',
               'ndsi_mean', 'ndsi_cv', 'images_count', 'valid_pct']
export_names = ['Year', 'Area unmixing (km²)', '% AOI', 'Type',
                'Mean NDSI', 'NDSI CV (%)', 'Images', '% Valid']
if has_ndsi:
    export_cols.insert(2, 'ndsi_area_km2')
    export_names.insert(2, 'Area NDSI (km²)')

metrics = df[export_cols].copy()
metrics.columns = export_names
metrics.to_csv('results/final_metrics.csv', index=False)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print('\n' + '='*70)
print('ANALYSIS COMPLETED')
print('='*70)
print(f'\nPeriod: {initial_year}-{final_year} ({len(df)} years, {len(df_dry)} dry)')
print(f'Model: Exponential (R²={r2:.4f}, {rate_pct:.2f}%/year)')
print(f'  Reclassification (→ ice patch): ~{int(reclass_year)} (±2 years)' if reclass_year else '')
print(f'  Disappearance (below detection): ~{int(disappear_year)} (±3 years)' if disappear_year else '')
print(f'\nOutputs: 4 plots, 3 CSVs, metadata updated')
if omitted_years: print(f'Note: Years {omitted_years} omitted (no Sentinel-2 data)')
print('='*70)