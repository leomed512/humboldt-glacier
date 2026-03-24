"""
Statistical analysis and visualization - Pico Humboldt Glacier

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
# HELPER FUNCTIONS
# ============================================================================

def plot_polygon(ax, gdf, color, linewidth, alpha=0.9):
    """
    Draw polygons (Polygon or MultiPolygon) on an axes
    
    Args:
        ax: Matplotlib axes
        gdf: GeoDataFrame with geometries
        color: Line color
        linewidth: Line width
        alpha: Transparency
    """
    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, linewidth=linewidth, 
                   linestyle='-', alpha=alpha)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=color, linewidth=linewidth, 
                       linestyle='-', alpha=alpha)

def compare_models(years, areas):
    """
    Compare multiple regression models (exponential, linear, polynomial)
    
    Args:
        years: Array of years
        areas: Array of areas (km²)
    
    Returns:
        DataFrame sorted by AIC (lower = better fit)
    """
    results = []
    
    # MODEL 1: EXPONENTIAL
    def exp_model(t, a, k, t0):
        return a * np.exp(k * (t - t0))
    
    try:
        popt, _ = curve_fit(exp_model, years, areas, 
                           p0=[areas[0], -0.2, years[0]],
                           maxfev=5000)
        pred = exp_model(years, *popt)
        r2 = r2_score(areas, pred)
        rmse = np.sqrt(mean_squared_error(areas, pred))
        n = len(years)
        aic = n * np.log(rmse**2) + 2 * 3
        
        results.append({
            'model': 'Exponential',
            'params': 3,
            'r2': r2,
            'rmse': rmse,
            'aic': aic,
            'rate_of_change': popt[1] * 100,
            'equation': f'A(t) = {popt[0]:.4f} × exp({popt[1]:.4f} × (t - {popt[2]:.0f}))'
        })
    except Exception as e:
        print(f'    Exponential failed: {str(e)[:50]}')
    
    # MODEL 2: LINEAR
    def lin_model(t, a, b):
        return a + b * t
    
    try:
        popt, _ = curve_fit(lin_model, years, areas)
        pred = lin_model(years, *popt)
        r2 = r2_score(areas, pred)
        rmse = np.sqrt(mean_squared_error(areas, pred))
        n = len(years)
        aic = n * np.log(rmse**2) + 2 * 2
        
        rate_rel = (popt[1] / areas.mean()) * 100
        
        results.append({
            'model': 'Linear',
            'params': 2,
            'r2': r2,
            'rmse': rmse,
            'aic': aic,
            'rate_of_change': rate_rel,
            'equation': f'A(t) = {popt[0]:.4f} + {popt[1]:.6f} × t'
        })
    except Exception as e:
        print(f'    Linear failed: {str(e)[:50]}')
    
    # MODEL 3: POLYNOMIAL DEGREE 2
    def pol2_model(t, a, b, c):
        return a + b*t + c*t**2
    
    try:
        popt, _ = curve_fit(pol2_model, years, areas)
        pred = pol2_model(years, *popt)
        r2 = r2_score(areas, pred)
        rmse = np.sqrt(mean_squared_error(areas, pred))
        n = len(years)
        aic = n * np.log(rmse**2) + 2 * 3
        
        results.append({
            'model': 'Polynomial 2',
            'params': 3,
            'r2': r2,
            'rmse': rmse,
            'aic': aic,
            'rate_of_change': None,
            'equation': f'A(t) = {popt[0]:.4f} + {popt[1]:.6f}×t + {popt[2]:.8f}×t²'
        })
    except Exception as e:
        print(f'    Polynomial failed: {str(e)[:50]}')
    
    if len(results) > 0:
        return pd.DataFrame(results).sort_values('aic')
    else:
        return pd.DataFrame()

# ============================================================================
# VISUAL CONFIGURATION
# ============================================================================

sns.set_theme(style='ticks', context='paper', font_scale=1.1)
sns.set_palette('deep')

COLORS = {
    'wet': '#3498db',      
    'dry': '#e74c3c',      
    'trend': '#f39c12',    
    'reference': '#27ae60',
    'neutral': "#a6aaab",
    'black': "#2C2A2A"
}

plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'legend.frameon': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5
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

print(f'\nData loaded: {len(df)} years')
print(f'Period: {df["year"].min()}-{df["year"].max()}')

try:
    with open('data/processing_metadata.json', 'r') as f:
        metadata = json.load(f)
    omitted_years = metadata.get('years_excluded', [])
    if omitted_years:
        print(f'Omitted years: {omitted_years} (no Sentinel-2 data)')
except FileNotFoundError:
    omitted_years = []
    metadata = {}

# ============================================================================
# KEY METRICS
# ============================================================================

initial_year = int(df['year'].min())
final_year = int(df['year'].max())
initial_area = df.loc[df['year'] == initial_year, 'snow_area_km2'].values[0]
final_area = df.loc[df['year'] == final_year, 'snow_area_km2'].values[0]
elapsed_years = final_year - initial_year

exp_rate = np.log(final_area / initial_area) / elapsed_years
rate_pct = exp_rate * 100
total_loss = initial_area - final_area
loss_pct = (total_loss / initial_area) * 100

minimum_year = int(df.loc[df['snow_area_km2'].idxmin(), 'year'])
minimum_area = df['snow_area_km2'].min()

print(f'\nRetreat rate: {rate_pct:.2f}% per year')
print(f'Total loss: {loss_pct:.1f}% ({total_loss:.4f} km²)')
print(f'Historical minimum: {minimum_year} ({minimum_area:.4f} km²)')

# ============================================================================
# VALIDATION AGAINST RAMÍREZ
# ============================================================================

ramirez_data = {
    2015: {'area': 0.110, 'error': 0.005},
    2016: {'area': 0.079, 'error': 0.005},
    2019: {'area': 0.046, 'error': 0.004}
}

validation = []
for year, data in ramirez_data.items():
    if year not in df['year'].values:
        validation.append({
            'year': year,
            'ramirez_area_km2': data['area'],
            'this_study_km2': None,
            'status': 'NO_DATA'
        })
    else:
        study_area = df.loc[df['year'] == year, 'snow_area_km2'].values[0]
        diff_pct = ((study_area - data['area']) / data['area']) * 100
        validation.append({
            'year': year,
            'ramirez_area_km2': data['area'],
            'this_study_km2': round(study_area, 4),
            'diff_pct': round(diff_pct, 1),
            'status': 'Includes seasonal snow' if diff_pct > 50 else 'Acceptable'
        })

pd.DataFrame(validation).to_csv('results/validation_ramirez2020.csv', index=False)
print('\nValidation saved: results/validation_ramirez2020.csv')

# ============================================================================
# YEAR CLASSIFICATION
# ============================================================================

dry_threshold = 0.060

df['year_type'] = df['snow_area_km2'].apply(
    lambda x: 'Wet' if x > dry_threshold else 'Dry'
)

dry_years = df[df['year_type'] == 'Dry']
wet_years = df[df['year_type'] == 'Wet']

print(f'\nClassification (threshold: {dry_threshold:.4f} km²):')
print(f'  Wet: {len(wet_years)} years → {wet_years["year"].tolist()}')
print(f'  Dry: {len(dry_years)} years → {dry_years["year"].tolist()}')

# ============================================================================
# MODEL A: ALL YEARS
# ============================================================================

def exp_func(x, a, k, x0):
    return a * np.exp(k * (x - x0))

popt_all, _ = curve_fit(exp_func, df['year'], df['snow_area_km2'], 
                        p0=[initial_area, exp_rate, initial_year])
y_pred_all = exp_func(df['year'], *popt_all)
residuals = df['snow_area_km2'] - y_pred_all

r2_all = 1 - (np.sum(residuals**2) / np.sum((df['snow_area_km2'] - df['snow_area_km2'].mean())**2))
rate_all = popt_all[1] * 100

# ============================================================================
# MODEL B: DRY YEARS ONLY
# ============================================================================

df_dry_model = df[df['year_type'] == 'Dry'].copy()

if len(df_dry_model) < 3:
    print(f'\n WARNING: Only {len(df_dry_model)} dry years available')
    print('   Minimum 3 years required for exponential fit')

initial_year_dry = int(df_dry_model['year'].min())

popt_dry, _ = curve_fit(exp_func, df_dry_model['year'], 
                          df_dry_model['snow_area_km2'],
                          p0=[df_dry_model['snow_area_km2'].iloc[0], -0.2, initial_year_dry])

y_pred_dry = exp_func(df_dry_model['year'], *popt_dry)
r2_dry = 1 - (np.sum((df_dry_model['snow_area_km2'] - y_pred_dry)**2) / 
                np.sum((df_dry_model['snow_area_km2'] - df_dry_model['snow_area_km2'].mean())**2))
rate_dry = popt_dry[1] * 100

# Model RMSE
residuals_dry = df_dry_model['snow_area_km2'] - y_pred_dry
rmse_dry = np.sqrt(np.mean(residuals_dry**2))
confidence_interval = 1.96 * rmse_dry

print(f'\nModel A (all, n={len(df)}): {rate_all:.2f}%/year (R²={r2_all:.4f})')
print(f'Model B (dry, n={len(df_dry_model)}): {rate_dry:.2f}%/year (R²={r2_dry:.4f})')
print(f'95% confidence interval: ±{confidence_interval:.4f} km²')

# ============================================================================
# CALCULATE PROJECTIONS 
# ============================================================================

glacier_threshold = 0.005
total_disappearance_threshold = 0.0001

glacier_year_calc = None
total_disappearance_year_calc = None

if popt_dry[1] < 0:
    t_glacier = np.log(glacier_threshold / popt_dry[0]) / popt_dry[1]
    glacier_year_calc = initial_year_dry + t_glacier
    
    t_total_disappearance = np.log(total_disappearance_threshold / popt_dry[0]) / popt_dry[1]
    total_disappearance_year_calc = initial_year_dry + t_total_disappearance

# Dynamic limit for plots
if total_disappearance_year_calc and total_disappearance_year_calc <= 2060:
    plot_year_limit = max(int(total_disappearance_year_calc) + 5, 2030)
else:
    plot_year_limit = 2035

# ============================================================================
# ALTERNATIVE MODEL COMPARISON
# ============================================================================

print('\n' + '='*70)
print('ALTERNATIVE MODEL COMPARISON (DRY YEARS)')
print('='*70)

model_results = compare_models(
    df_dry_model['year'].values, 
    df_dry_model['snow_area_km2'].values
)

if len(model_results) > 0:
    print('\nFitted models:')
    print(model_results[['model', 'params', 'r2', 'rmse', 'aic', 'rate_of_change']].to_string(index=False))
    print(f'\nSelected model (minimum AIC): {model_results.iloc[0]["model"]}')
    
    model_results.to_csv('results/model_comparison.csv', index=False)
    print('\n   Table saved: results/model_comparison.csv')
else:
    print('\n   Could not fit models')

print('='*70 + '\n')

# ============================================================================
# GENERATE COMPARATIVE IMAGE 2020 vs 2025
# ============================================================================

print('\n' + '='*70)
print('GENERATING COMPARATIVE RGB IMAGE (2020 vs 2025)')
print('='*70 + '\n')

# Variables for metadata (declare with None as default)
area_2020 = None
area_2025 = None
loss_pct_img = None

try:
    if 2020 not in df['year'].values or 2025 not in df['year'].values:
        print('   Years 2020 or 2025 not available')
    else:
        rgb_path = 'data/2025_rgb.tif'
        poly_2020_path = 'data/glacier_polygons/2020.geojson'
        poly_2025_path = 'data/glacier_polygons/2025.geojson'
        
        if not (Path(rgb_path).exists() and Path(poly_2020_path).exists() and Path(poly_2025_path).exists()):
            print('   Required files not available')
        else:
            print('Generating image...', end=' ')
            
            # Load RGB
            with rasterio.open(rgb_path) as src:
                rgb = src.read([1, 2, 3])
                bounds = src.bounds
                rgb = np.transpose(rgb, (1, 2, 0))
                if rgb.max() > 255:
                    rgb = ((rgb / rgb.max()) * 255).astype(np.uint8)
            
            # Load polygons
            gdf_2020 = gpd.read_file(poly_2020_path)
            gdf_2025 = gpd.read_file(poly_2025_path)
            
            # Statistics
            area_2020 = df.loc[df['year'] == 2020, 'snow_area_km2'].values[0]
            area_2025 = df.loc[df['year'] == 2025, 'snow_area_km2'].values[0]
            loss_pct_img = ((area_2020 - area_2025) / area_2020) * 100
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Show RGB
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            ax.imshow(rgb, extent=extent)
            
            # Polygons using helper function
            plot_polygon(ax, gdf_2020, '#e74c3c', 1)
            plot_polygon(ax, gdf_2025, '#3498db', 0.9)
            
            # Text labels
            text_x = 0.98
            text_y_start = 0.7
            
            ax.text(text_x, text_y_start, 'GLACIER RETREAT\n2020-2025',
                   transform=ax.transAxes, fontsize=14, fontweight='bold',
                   color='white', ha='right', va='top')
            
            ax.text(text_x, text_y_start - 0.05, f'{loss_pct_img:.1f}%',
                   transform=ax.transAxes, fontsize=28, fontweight='bold',
                   color='white', ha='right', va='top')
            
            stats_text = f'''
Area 2020: {area_2020:.4f} km²
Area 2025: {area_2025:.4f} km²
Loss: {area_2020 - area_2025:.4f} km²

Annual rate: {loss_pct_img/5:.1f}%/year'''
            
            ax.text(text_x, text_y_start - 0.20, stats_text,
                   transform=ax.transAxes, fontsize=12, color='white',
                   ha='right', va='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.75))
            
            # Legend
            legend_elements = [
                Line2D([0], [0], color='#e74c3c', linewidth=2, label='Glacier 2020'),
                Line2D([0], [0], color='#3498db', linewidth=2, label='Glacier 2025')
            ]
            ax.legend(handles=legend_elements, loc='lower right', 
                     fontsize=11, frameon=True, facecolor='white', 
                     edgecolor='black', framealpha=0.9)
            
            # Configuration
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('Pico Humboldt - Venezuela', 
                        fontsize=18, fontweight='bold', pad=15)
            
            # ADJUST LAYOUT BEFORE LOGO
            plt.tight_layout()
            
            # Logo
            logo_path = 'assets/logo_leo.png'
            if Path(logo_path).exists():
                try:
                    from PIL import Image
                    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                    
                    logo = Image.open(logo_path)
                    if logo.mode != 'RGBA':
                        logo = logo.convert('RGBA')
                    logo_array = np.array(logo)
                    
                    logo_ax = inset_axes(ax, width="5%", height="5%", 
                                        loc='lower left', borderpad=1)
                    logo_ax.imshow(logo_array)
                    logo_ax.axis('off')
                    
                except ImportError:
                    print('\n   Pillow not installed')
                except Exception as e:
                    print(f'\n   Logo error: {e}')
            
            # Save
            output_path = 'results/plots/comparison_2020_2025.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print('')
            print(f'\n   Image saved: {output_path}')
            print(f'    Loss: {loss_pct_img:.1f}% ({area_2020:.4f} → {area_2025:.4f} km²)')
            print(f'    Annual rate: {loss_pct_img/5:.1f}%/year')

except Exception as e:
    print(f'\n   Error generating image: {e}')
    import traceback
    traceback.print_exc()

# ============================================================================
# PLOTS
# ============================================================================

print('\nGenerating plots...')

# PLOT 1: Time series + Anomalies
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for idx, row in df.iterrows():
    color = COLORS['wet'] if row['year_type'] == 'Wet' else COLORS['dry']
    marker = 'o' if row['year_type'] == 'Wet' else 's'
    ax1.scatter(row['year'], row['snow_area_km2'], 
              s=60, color=color, marker=marker, linewidth=0, 
              zorder=3, alpha=0.8)

ax1.plot(df['year'], df['snow_area_km2'], 
        color=COLORS['neutral'], linewidth=1, alpha=0.4, zorder=1)

years_fit = np.linspace(initial_year, final_year, 100)
area_fit = exp_func(years_fit, *popt_all)
ax1.plot(years_fit, area_fit, '--', color=COLORS['trend'], 
        linewidth=2, label=f'Trend ({rate_all:.1f}%/year)', zorder=2)

ax1.scatter([], [], s=50, color=COLORS['wet'], marker='o', 
          label='Wet years')
ax1.scatter([], [], s=50, color=COLORS['dry'], marker='s',
          label='Dry years')

ax1.set_ylabel('Snow area (km²)')
ax1.set_title(f'Temporal evolution - Pico Humboldt Glacier ({initial_year}-{final_year})', 
             fontweight='bold')
ax1.legend(loc='upper right', fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# Anomalies
residual_colors = [COLORS['wet'] if r > 0 else COLORS['dry'] for r in residuals]
ax2.bar(df['year'], residuals, color=residual_colors, 
        edgecolor='white', linewidth=0.5, alpha=0.8)
ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax2.set_xlabel('Year')
ax2.set_ylabel('Deviation (km²)')
ax2.set_title('Anomalies relative to exponential model', fontweight='bold')
ax2.grid(True, alpha=0.3)

for year, res in zip(df['year'], residuals):
    if abs(res) > 0.05:
        ax2.text(year, res, f'{res:+.4f}', ha='center', 
               va='bottom' if res > 0 else 'top',
               fontsize=7, alpha=0.7)

sns.despine()
plt.tight_layout()
plt.savefig('results/plots/01_time_series_anomalies.png', dpi=300, bbox_inches='tight')
print('   01_time_series_anomalies.png')
plt.close()

# PLOT 2: Data availability
fig, ax = plt.subplots(figsize=(10, 4))

colors_quality = [COLORS['dry'] if x < 5 else COLORS['trend'] if x < 7 else COLORS['reference'] 
                  for x in df['images_count']]

ax.bar(df['year'], df['images_count'], color=colors_quality, 
       edgecolor='white', linewidth=0.5, alpha=0.8)
ax.axhline(7, color=COLORS['black'], linestyle='--', linewidth=1, 
           alpha=0.5, label='High quality (≥7 images)')
ax.axhline(5, color=COLORS['trend'], linestyle='--', linewidth=1, 
           alpha=0.5, label='Medium quality (5-6)')
ax.set_xlabel('Year')
ax.set_ylabel('N° Sentinel-2 images')
ax.set_title('Data availability per year')
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(0, max(df['images_count']) * 1.2)

sns.despine()
plt.tight_layout()
plt.savefig('results/plots/02_data_availability.png', dpi=300, bbox_inches='tight')
print('   02_data_availability.png')
plt.close()

# PLOT 3: Variability + Classification
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

df_sorted = df.sort_values('year')
changes = df_sorted['snow_area_km2'].diff()
changes_pct = (changes / df_sorted['snow_area_km2'].shift(1)) * 100
change_years = df_sorted['year'][1:]
change_colors = [COLORS['dry'] if x < 0 else COLORS['reference'] for x in changes[1:]]

bars = ax1.bar(change_years, changes[1:], color=change_colors, 
               edgecolor='white', linewidth=0.5, alpha=0.8)

for bar, pct in zip(bars, changes_pct[1:]):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{pct:.0f}%', ha='center', 
             va='bottom' if height > 0 else 'top',
             fontsize=8, alpha=0.7)

ax1.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax1.set_ylabel('Annual change (km²)')
ax1.set_title('Interannual variability')
ax1.grid(True, alpha=0.3)

ax2.scatter(wet_years['year'], wet_years['snow_area_km2'],
            s=80, color=COLORS['wet'], marker='o',
            label='Wet years', zorder=3, alpha=0.8)
ax2.scatter(dry_years['year'], dry_years['snow_area_km2'],
            s=80, color=COLORS['dry'], marker='s',
            label='Dry years', zorder=3, alpha=0.8)
ax2.scatter([2019], [0.046], s=50, marker='D', 
            color=COLORS['reference'],
            label='Ramírez 2019', zorder=4, alpha=0.9)

if minimum_year in df['year'].values:
    ax2.scatter([minimum_year], [minimum_area], s=30, marker='^',
                color='black', label=f'{minimum_year}: Minimum', zorder=5)

ax2.set_xlabel('Year')
ax2.set_ylabel('Area (km²)')
ax2.set_title('Classification: Seasonal snow vs Permanent glacier')
ax2.grid(True, alpha=0.3, which='both')
legend = ax2.legend(loc='upper right', 
                   fontsize=8,
                   frameon=True,
                   fancybox=True,
                   shadow=False,
                   framealpha=0.5,
                   edgecolor='black',
                   borderpad=0.4,
                   labelspacing=0.4)
legend.get_frame().set_linewidth(0.8)

sns.despine()
plt.tight_layout()
plt.savefig('results/plots/03_variability_classification.png', dpi=300, bbox_inches='tight')
print('   03_variability_classification.png')
plt.close()

# PLOT 4: Projection
fig, ax = plt.subplots(figsize=(10, 5))

all_years = list(range(initial_year, plot_year_limit + 1))
ax.set_xticks(all_years[::2])
ax.set_xticklabels([str(a) for a in all_years[::2]], rotation=45, ha='right')

years_extended = np.arange(initial_year_dry, plot_year_limit + 1, 0.1)
area_extended_dry = exp_func(years_extended, *popt_dry)

years_hist = years_extended[years_extended <= final_year]
area_hist = area_extended_dry[years_extended <= final_year]
years_fut = years_extended[years_extended >= final_year]
area_fut = area_extended_dry[years_extended >= final_year]

ax.scatter(df['year'], df['snow_area_km2'], s=50, 
           color=COLORS['neutral'], label='Other years', zorder=4, alpha=0.9)
ax.scatter(df_dry_model['year'], df_dry_model['snow_area_km2'], 
           s=50, color=COLORS['dry'], label='Model years', zorder=5, alpha=0.9)
ax.plot(years_hist, area_hist, '-', color=COLORS['trend'], 
        linewidth=2, label='Fitted model', zorder=2)
ax.plot(years_fut, area_fut, '--', color=COLORS['dry'], 
        linewidth=2, label='Projection', zorder=3, alpha=0.8)

# Thresholds
if glacier_year_calc:
    ax.annotate(f'{int(glacier_year_calc)}(±2)\nGlacier threshold\n(< 0.005 km²)',
                xy=(glacier_year_calc, 0), xytext=(glacier_year_calc, glacier_threshold * 8),
                fontsize=7, color=COLORS['black'], fontweight='bold',
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-', color=COLORS['neutral'], 
                              lw=1.5, linestyle=':', alpha=0.6))

ax.axhline(total_disappearance_threshold, color='darkred', linestyle='--', 
          linewidth=1.5, alpha=0.5, zorder=2)

if total_disappearance_year_calc:
    ax.annotate(f'{int(total_disappearance_year_calc)} (±2)\nTotal disappearance',
                xy=(total_disappearance_year_calc, 0), 
                xytext=(total_disappearance_year_calc, initial_area * 0.15),
                fontsize=7, color='black', fontweight='bold',
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-', color='black', 
                              lw=1.5, linestyle=':', alpha=0.6))

ax.set_xlabel('Year')
ax.set_ylabel('Area (km²)')
ax.set_title('Glacier disappearance projection')

note_text = (f'Model: dry years ({df_dry_model["year"].min()}-{df_dry_model["year"].max()})\n'
              f'Excludes: {", ".join(map(str, wet_years["year"].values))}\n'
              f'Rate: {rate_dry:.1f}%/year (R²={r2_dry:.2f})')
ax.text(0.02, 0.98, note_text, transform=ax.transAxes, fontsize=7.5, 
        va='top', ha='left', alpha=0.6,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                  edgecolor='gray', alpha=0.3, linewidth=0.5))

ax.legend(loc='upper right', fontsize=7)
ax.set_xlim(initial_year - 1, plot_year_limit)
ax.set_ylim(0, initial_area * 1.15)
ax.grid(True, alpha=0.3)

sns.despine()
plt.tight_layout()
plt.savefig('results/plots/04_adjusted_projection.png', dpi=300, bbox_inches='tight')
print('   04_adjusted_projection.png')
plt.close()

# ============================================================================
# UPDATE METADATA
# ============================================================================

print('\n Updating metadata...')

metadata['model_b'] = {
    'type': 'exponential',
    'description': 'Dry years only (excludes wet years with seasonal snow)',
    'equation': f'A(t) = {popt_dry[0]:.6f} × exp({popt_dry[1]:.6f} × (t - {initial_year_dry}))',
    'rate_percent_per_year': round(float(rate_dry), 2),
    'r_squared': round(float(r2_dry), 3),
    'years_used': df_dry_model['year'].tolist(),
    'years_excluded_wet': wet_years['year'].tolist(),
    'n_years': int(len(df_dry_model)),
    'parameters': {
        'a': float(popt_dry[0]),
        'k': float(popt_dry[1]),
        't0': int(initial_year_dry)
    },
    'thresholds': {
        'glacier_definition_km2': float(glacier_threshold),
        'glacier_definition_source': 'Huss & Fischer 2016',
        'practical_disappearance_km2': float(total_disappearance_threshold),
        'practical_disappearance_note': 'Area below Sentinel-2 detection'
    },
    'uncertainty': {
        'model_rmse_km2': round(float(rmse_dry), 4),
        'confidence_interval_95pct_km2': round(float(confidence_interval), 4)
    }
}

# Projections
projections = {}
if glacier_year_calc and 2020 <= glacier_year_calc <= 2050:
    projections['glacier_threshold'] = {
        'year': round(float(glacier_year_calc), 1),
        'threshold_km2': float(glacier_threshold),
        'uncertainty_years': 2,
        'valid': True
    }

if total_disappearance_year_calc and 2020 <= total_disappearance_year_calc <= 2060:
    projections['total_disappearance'] = {
        'year': round(float(total_disappearance_year_calc), 1),
        'threshold_km2': float(total_disappearance_threshold),
        'uncertainty_years': 3,
        'valid': True
    }

metadata['model_b']['projections'] = projections

# Comparative image (only if generated)
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

print('   Metadata updated')

# ============================================================================
# EXPORT METRICS
# ============================================================================

metrics = df[['year', 'snow_area_km2', 'snow_pct', 'year_type',
               'ndsi_mean', 'ndsi_cv', 'images_count', 'valid_pct']].copy()
metrics.columns = ['Year', 'Area (km²)', '% AOI', 'Type',
                    'Mean NDSI', 'NDSI CV (%)', 'Images', '% Valid']
metrics = metrics.round({
    'Area (km²)': 4,
    '% AOI': 2,
    'Mean NDSI': 3,
    'NDSI CV (%)': 2,
    '% Valid': 1
})
metrics.to_csv('results/final_metrics.csv', index=False)
print('   Metrics exported')

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print('\n' + '='*70)
print('ANALYSIS COMPLETED')
print('='*70)
print(f'\nPeriod: {initial_year}-{final_year} ({len(df)} years)')
print(f'\nModel A (all): {rate_all:.2f}%/year (R²={r2_all:.4f})')
print(f'Model B (dry): {rate_dry:.2f}%/year (R²={r2_dry:.4f})')
print(f'\nClassification (threshold: {dry_threshold:.4f} km²):')
print(f'  Wet: {wet_years["year"].tolist()}')
print(f'  Dry: {dry_years["year"].tolist()}')

if popt_dry[1] < 0 and glacier_year_calc and glacier_year_calc <= 2050:
    print(f'\nModel B Projection:')
    print(f'  • Glacier threshold: ~{int(glacier_year_calc)} (±2 years)')
    if total_disappearance_year_calc and total_disappearance_year_calc <= 2060:
        print(f'  • Total disappearance: ~{int(total_disappearance_year_calc)} (±3 years)')

print('\nPlots generated:')
print('   01_time_series_anomalies.png')
print('   02_data_availability.png')
print('   03_variability_classification.png')
print('   04_adjusted_projection.png')
if area_2020 is not None:
    print('   comparison_2020_2025.png')

print('\nCSV files:')
print('   final_metrics.csv')
print('   model_comparison.csv')
print('   validation_ramirez2020.csv')

print('\nMetadata:')
print('   processing_metadata.json')

if omitted_years:
    print(f'\nNote: Years {omitted_years} omitted (no data)')

print('\n' + '='*70 + '\n')