"""
Process Sentinel-2 imagery in Google Earth Engine

- Sub-pixel spectral unmixing for fractional snow cover estimation
- Binary NDSI preserved as quality control metric
- Years without data are OMITTED (no fake values are generated)
- Complete metadata documents excluded years

Output:
  - data/snow_stats_2015_2026.csv (only years with valid data)
  - data/humboldt_dem_30m.tif (extended AOI)
  - data/2025_rgb.tif (extended AOI)
  - data/glacier_polygons/{year}.geojson 
  - data/aoi_humboldt.geojson (small AOI)
  - data/aoi_visualization.geojson (extended AOI)
  - data/processing_metadata.json (complete documentation)

"""

import ee
import geemap
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import geojson  
import numpy as np


# ============================================================================
# 1. AUTHENTICATION AND CONFIGURATION
# ============================================================================

print('='*70)
print('SENTINEL-2 PROCESSING - PICO HUMBOLDT GLACIER')
print('='*70)

# Initialize Earth Engine
try:
    ee.Initialize(project='pico-umboldt-feb-2019-2026')  # Adjust your project ID
    print('  Google Earth Engine initialized successfully')
except Exception as e:
    print(f' Error initializing GEE: {e}')
    print('\nSolution: Run first:')
    print('  import ee')
    print('  ee.Authenticate()')
    exit(1)

# ============================================================================
# 2. PARAMETERS
# ============================================================================

START_YEAR = 2015
END_YEAR = 2026
NDSI_THRESHOLD = 0.4
CLOUD_PCT_MAX = 60
SCALE = 20

# ---------------------------------------------------------------------------
# Spectral unmixing configuration
# ---------------------------------------------------------------------------
# Endmember spectra for Sentinel-2 bands [B2, B3, B4, B8, B11, B12]
# Units: surface reflectance (Sentinel-2 SR scale)
#
# Calibrated from pure pixels within AOI using 2020 composite (14 images).
# See data/endmember_calibration.json for sampling coordinates and validation.
#
# Snow:       NDSI=0.924 | High visible, near-zero SWIR (clean glacier ice)
# Rock:       NDSI=-0.142 | Moderate visible, high SWIR (exposed moraine)
# Vegetation: NDSI=-0.568 | Low visible, moderate NIR/SWIR (páramo)
#
# Spectral separability (all GOOD):
#   snow vs rock: 43.0° | snow vs vegetation: 63.6° | rock vs vegetation: 21.1°
# ---------------------------------------------------------------------------
ENDMEMBERS = {
    'snow':       [8242, 7734, 6799, 4998, 304, 314],
    'rock':       [2050, 2367, 2445, 2397, 3151, 3022],
    'vegetation': [636,  723,  808,  1261, 2623, 2172]
}
UNMIX_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

# Fraction threshold for binary polygon generation
# Pixels with snow_fraction >= 0.5 are classified as glacier for vectorization
# (continuous fractions cannot be vectorized; polygons require discrete boundaries)
SNOW_FRACTION_THRESHOLD = 0.5

# NDSI pre-mask for unmixing candidate zone
# Only pixels with NDSI >= this value enter the unmixing.
# All others are forced to snow_fraction = 0.
# Rationale: the AOI is 4.09 km² but the glacier is ~0.05 km² (1.2%).
# Without pre-masking, thousands of non-snow pixels receive small residual
# snow fractions that accumulate into a large false positive area.
# Threshold of 0.2 is deliberately relaxed (vs 0.4 for binary classification)
# to include mixed pixels at glacier margins while excluding pure rock/vegetation.
NDSI_PREMASK_THRESHOLD = 0.2

print(f'\nProcessing parameters:')
print(f'  Period: {START_YEAR}-{END_YEAR}')
print(f'  Temporal window: Dec 15 (year-1) - Mar 15 (year)')
print(f'  NDSI threshold: {NDSI_THRESHOLD} (quality control)')
print(f'  NDSI pre-mask: {NDSI_PREMASK_THRESHOLD} (unmixing candidate zone)')
print(f'  Snow fraction threshold: {SNOW_FRACTION_THRESHOLD} (polygon generation)')
print(f'  Maximum cloud cover: {CLOUD_PCT_MAX}%')
print(f'  Resolution: {SCALE}m')
print(f'  Method: Linear spectral unmixing ({len(ENDMEMBERS)} endmembers, NDSI-masked)')

# Create data folders
Path('data').mkdir(exist_ok=True)
Path('data/glacier_polygons').mkdir(exist_ok=True)


# ============================================================================
# 3. DEFINE TWO AOIs (CRITICAL)
# ============================================================================

# SMALL AOI - For processing (metrics)
aoi_processing = ee.Geometry.Polygon([[
    [-71.0079,   8.55679],
    [-70.988288, 8.55679],
    [-70.988288, 8.53973],
    [-71.0079,   8.53973],
    [-71.0079,   8.55679]
]])

# EXTENDED AOI - For visualization (context ~28 km²)
aoi_visualization = ee.Geometry.Polygon([[
    [-71.02021478567892, 8.576517837930535],
          [-71.02021478567892, 8.529666220919108],
          [-70.97163461600118, 8.529666220919108],
          [-70.97163461600118, 8.576517837930535]
          ]])

print(f'\nAreas of interest defined:')
print(f'    Processing AOI: ~4 km² (metrics calculation)')
print(f'    Visualization AOI: ~28 km² (geographic context)')

# ============================================================================
# 4. PROCESSING FUNCTIONS
# ============================================================================

def mask_s2_scl(img):
    """
    Mask invalid pixels using Scene Classification Layer (SCL)
    
    Removes: NO_DATA, SATURATED, CLOUD_SHADOWS, CLOUD_MEDIUM, 
             CLOUD_HIGH, THIN_CIRRUS
    """
    scl = img.select('SCL')
    # Keep: 2,4,5,6,7,11
    mask = (scl.eq(2).Or(scl.eq(4)).Or(scl.eq(5))
           .Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11)))
    return img.updateMask(mask)


def add_snow_bands(img):
    """
    Compute fractional snow cover via NDSI-masked linear spectral unmixing.
    
    Two-stage approach:
      1. NDSI pre-mask (>= 0.2): identifies candidate snow zone.
         Excludes pure rock/vegetation pixels that would accumulate
         false positive snow fractions across the 4 km² AOI.
      2. Spectral unmixing: decomposes only candidate pixels into
         fractional abundances of snow, rock, and vegetation.
         Non-candidate pixels are assigned snow_fraction = 0.
    
    This hybrid approach combines NDSI's robustness for gross discrimination
    with unmixing's sub-pixel precision at glacier margins.
    (Painter et al., 2009; Sirguey et al., 2009)
    
    Output bands:
        snow_fraction: Continuous 0.0-1.0 (primary area metric)
        snow:          Binary mask from fraction >= 0.5 (for polygons)
        NDSI:          Normalized Difference Snow Index (quality control)
    """
    # ── Stage 1: NDSI pre-mask ─────────────────────────────────────────
    # Relaxed threshold (0.2) captures mixed pixels at glacier margins
    # while eliminating pure rock/vegetation (NDSI typically < 0)
    ndsi = img.normalizedDifference(['B3', 'B11']).rename('NDSI')
    candidate_mask = ndsi.gte(NDSI_PREMASK_THRESHOLD)

    # ── Stage 2: Spectral unmixing (candidate zone only) ───────────────
    endmember_list = [
        ENDMEMBERS['snow'],
        ENDMEMBERS['rock'],
        ENDMEMBERS['vegetation']
    ]

    fractions = img.select(UNMIX_BANDS).unmix(
        endmembers=endmember_list,
        sumToOne=True,
        nonNegative=True
    ).rename(['snow_fraction', 'rock_fraction', 'vegetation_fraction'])

    # Apply pre-mask: non-candidate pixels → snow_fraction = 0
    snow_fraction = fractions.select('snow_fraction').updateMask(candidate_mask).unmask(0)

    # ── Binary mask for polygon generation ─────────────────────────────
    snow_binary = snow_fraction.gte(SNOW_FRACTION_THRESHOLD).rename('snow')

    # Morphological filter: close small gaps, remove isolated pixels
    snow_binary = snow_binary.focal_max(1).focal_min(1).rename('snow')

    return img.addBands([snow_fraction, snow_binary, ndsi])


def generate_composite(year):
    """
    Generate dry season composite ENDING in the given year
    
    Temporal window: Dec 15 (year-1) → Mar 15 (year)
    e.g.: year=2019 → Dec 15, 2018 to Mar 15, 2019
           
    Returns:
        ee.Image: Median composite if data available
        None: If window is in the future, insufficient, or no images
    """
    from datetime import datetime
    
    year_int = int(year)
    
    # Nominal window: starts in December of the previous year
    start = ee.Date.fromYMD(year_int - 1, 12, 15)
    end = ee.Date.fromYMD(year_int, 3, 15)
    
    # Python dates for validation
    today = datetime.now()
    today_ee = ee.Date(today)
    start_date_py = datetime(year_int - 1, 12, 15)
    end_date_py = datetime(year_int, 3, 15)
    
    # ════════════════════════════════════════════════════════════
    # TEMPORAL VALIDATION
    # ════════════════════════════════════════════════════════════
    
    # Window entirely in the future
    if start_date_py > today:
        print(f'  {year_int}: Window in future (starts {start_date_py.strftime("%Y-%m-%d")}) - YEAR OMITTED')
        return None
    
    # Window partially in the future
    window_adjusted = False
    if end_date_py > today:
        end = today_ee
        end_date_py = today
        window_adjusted = True
        
        # Calculate available days
        days_available = (today - start_date_py).days
        
        # Check minimum duration
        if days_available < 30:
            print(f'  {year_int}: Insufficient window ({days_available} days available) - YEAR OMITTED')
            return None
        
        print(f'  {year_int}: Window adjusted to present ({days_available} days available)')
    
    # ════════════════════════════════════════════════════════════
    # SEARCH AND PROCESS IMAGES
    # ════════════════════════════════════════════════════════════
    
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(aoi_processing)
           .filterDate(start, end)
           .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_PCT_MAX))
           .map(mask_s2_scl))
    
    # Evaluate availability
    n = col.size().getInfo()
    
    if n == 0:
        print(f'  {year_int}: No images available - YEAR OMITTED')
        return None
    
    # Process composite
    composite = col.median()
    
    # Informational message
    if not window_adjusted:
        print(f'  {year_int}: {n} images processed')
    
    # Return with metadata
    return (composite
            .set('year', year_int)
            .set('images_count', n)
            .set('period_start', start.format('YYYY-MM-dd').getInfo())
            .set('period_end', end.format('YYYY-MM-dd').getInfo())
            .set('window_adjusted', window_adjusted))

def calculate_stats(composite):
    """
    Calculate zonal statistics using sub-pixel fractional snow cover.
    
    Primary metric (snow_area_km2):
        Each pixel contributes its snow_fraction × pixel_area.
        A 20m pixel that is 60% snow contributes 240 m² (not 0 or 400 m²).
    
    Secondary metric (binary_area_km2):
        Traditional binary count for comparison with previous results
        and external studies that use threshold-based methods.
    
    Quality metrics:
        NDSI mean/std computed over pixels where snow_fraction >= 0.5,
        providing spectral quality assessment of the detected snow.
    """
    # Add unmixing + NDSI bands
    composite_snow = add_snow_bands(composite)
    snow_fraction = composite_snow.select('snow_fraction')
    snow_binary = composite_snow.select('snow')
    ndsi = composite_snow.select('NDSI')
    
    # Composite with all bands for polygon export
    composite_processed = composite_snow

    # ── Primary: fractional snow area (sub-pixel) ──────────────────────
    fractional_area = snow_fraction.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi_processing,
        scale=SCALE,
        maxPixels=1e9
    ).get('snow_fraction')
    
    # ── Secondary: unmixing binary area (fraction >= 0.5) ────────────────
    binary_area = snow_binary.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi_processing,
        scale=SCALE,
        maxPixels=1e9
    ).get('snow')
    
    # ── Tertiary: NDSI binary area (traditional method, >= 0.4) ────────
    # Independent from unmixing — pure NDSI threshold for comparison
    ndsi_binary = ndsi.gte(NDSI_THRESHOLD).focal_max(1).focal_min(1)
    ndsi_binary_area = ndsi_binary.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi_processing,
        scale=SCALE,
        maxPixels=1e9
    ).get('NDSI')
    
    # ── NDSI statistics (over snow_fraction >= 0.5 zone) ───────────────
    ndsi_stats = ndsi.updateMask(snow_binary).reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        ),
        geometry=aoi_processing,
        scale=SCALE,
        maxPixels=1e9
    )
    
    # ── Mean snow fraction (quality indicator) ─────────────────────────
    # Average fraction across all valid pixels in AOI
    # High values (>0.7) indicate well-resolved glacier core
    # Low values (<0.3) indicate mostly mixed/marginal pixels
    mean_fraction = snow_fraction.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi_processing,
        scale=SCALE,
        maxPixels=1e9
    ).get('snow_fraction')
    
    # ── Valid pixel percentage ─────────────────────────────────────────
    valid_pixels = composite.select('B3').mask().reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi_processing,
        scale=SCALE,
        maxPixels=1e9
    ).get('B3')
    
    total_pixels = ee.Number(aoi_processing.area()).divide(SCALE * SCALE)
    valid_pct = ee.Number(valid_pixels).divide(total_pixels).multiply(100)
    
    # ── Extract values from server ─────────────────────────────────────
    year = composite.get('year').getInfo()
    images_count = composite.get('images_count').getInfo()
    period_start = composite.get('period_start').getInfo()
    period_end = composite.get('period_end').getInfo()
    
    fractional_area_m2 = ee.Number(fractional_area).getInfo()
    fractional_area_km2 = fractional_area_m2 / 1e6
    
    binary_area_m2 = ee.Number(binary_area).getInfo()
    binary_area_km2 = binary_area_m2 / 1e6
    
    ndsi_binary_area_m2 = ee.Number(ndsi_binary_area).getInfo()
    ndsi_binary_area_km2 = ndsi_binary_area_m2 / 1e6
    
    ndsi_mean = ee.Number(ndsi_stats.get('NDSI_mean')).getInfo()
    ndsi_std = ee.Number(ndsi_stats.get('NDSI_stdDev')).getInfo()
    
    mean_fraction_val = ee.Number(mean_fraction).getInfo()
    
    valid_pct_val = valid_pct.getInfo()
    
    # Clip valid_pct to [0, 100]
    # Values >100% are artifacts from multi-resolution resampling
    valid_pct_val = max(0.0, min(100.0, valid_pct_val))
    
    # Calculate AOI percentage (using fractional area as primary)
    aoi_area_km2 = 4.09
    snow_pct = (fractional_area_km2 / aoi_area_km2) * 100
    
    # ── Diagnostic: fractional vs binary difference ────────────────────
    if binary_area_km2 > 0:
        method_diff_pct = ((fractional_area_km2 - binary_area_km2) / binary_area_km2) * 100
    else:
        method_diff_pct = 0.0
    
    stats = {
        'year': year,
        'period_start': period_start,
        'period_end': period_end,
        'images_count': images_count,
        'snow_area_km2': round(fractional_area_km2, 6),      # PRIMARY: sub-pixel unmixing
        'binary_area_km2': round(binary_area_km2, 6),         # Unmixing discretized (fraction >= 0.5)
        'ndsi_area_km2': round(ndsi_binary_area_km2, 6),      # Traditional NDSI >= 0.4
        'method_diff_pct': round(method_diff_pct, 2),         # DIAGNOSTIC
        'snow_pct': round(snow_pct, 2),
        'valid_pct': round(valid_pct_val, 2), 
        'ndsi_mean': round(ndsi_mean, 3) if ndsi_mean else None,
        'ndsi_std': round(ndsi_std, 3) if ndsi_std else None,
        'mean_snow_fraction': round(mean_fraction_val, 4) if mean_fraction_val else None
    }

    return stats, composite_processed

# ============================================================================
# Calculate polygons
# ============================================================================

def export_glacier_polygon(composite_processed, year, aoi):
    """
    Convert binary snow mask to GeoJSON polygon.
    
    Uses the snow band (fraction >= 0.5 threshold) for vectorization.
    Polygons represent the discrete glacier boundary; area metrics
    use the continuous fraction separately.
    
    Args:
        composite_processed: ee.Image with 'snow' binary band
        year: Year
        aoi: Area of interest
    
    Returns:
        Path to GeoJSON file or None if it fails
    """
    try:
        snow_mask = composite_processed.select('snow')
        
        # Convert mask to vectors
        vectors = snow_mask.reduceToVectors(
            geometry=aoi,
            scale=10, # increased resolution
            geometryType='polygon',
            eightConnected=False,
            labelProperty='snow',
            maxPixels=1e10
        )
        
        # Filter only snow polygons (label=1)
        glacier_polygons = vectors.filter(ee.Filter.eq('snow', 1))
        
        # Simplify geometry (2m tolerance)
        glacier_polygons = glacier_polygons.map(
            lambda feat: feat.simplify(maxError=2)
        )
        
        # Download as GeoJSON
        fc_info = glacier_polygons.getInfo()
        
        if not fc_info or 'features' not in fc_info or len(fc_info['features']) == 0:
            print(f'      Year {year}: No glacier polygons')
            return None
        
        # Save file
        output_path = f'data/glacier_polygons/{year}.geojson'
        with open(output_path, 'w') as f:
            geojson.dump(fc_info, f, indent=2)
        
        num_polygons = len(fc_info['features'])
        print(f'    ✓ Polygon: {output_path} ({num_polygons} geom)')
        
        return output_path
        
    except Exception as e:
        print(f'     Error polygon {year}: {e}')
        return None
# ============================================================================
# 5. PROCESS ALL YEARS
# ============================================================================

print('\n' + '='*70)
print(f'PROCESSING YEARS {START_YEAR}-{END_YEAR}')
print('='*70 + '\n')

results = []
omitted_years = []
polygon_files = {} # Polygons
composites = {}  

for year in range(START_YEAR, END_YEAR + 1):
    try:
        composite = generate_composite(year)
        
        if composite is None:
            omitted_years.append(year)
            continue
        
        # Store composite for reuse
        composites[year] = composite  
        
        # Calculate statistics
        stats, composite_processed = calculate_stats(composite)
        results.append(stats)
        
        # Polygons
        polygon_path = export_glacier_polygon(composite_processed, year, aoi_processing)
        if polygon_path:
            polygon_files[year] = polygon_path
        
        print(f'  → Area: {stats["snow_area_km2"]:.4f} km² (unmix) | '
              f'{stats["ndsi_area_km2"]:.4f} km² (NDSI) | '
              f'NDSI: {stats["ndsi_mean"]:.2f} | '
              f'Valid: {stats["valid_pct"]:.1f}%')
    
    except Exception as e:
        print(f' {year}: Error - {e}')
        omitted_years.append(year)

# ============================================================================
# 6. EXPORT CSV
# ============================================================================

print('\n' + '='*70)
print('GENERATING METRICS CSV')
print('='*70 + '\n')

df = pd.DataFrame(results)

# Add calculated columns
df['ndsi_cv'] = (df['ndsi_std'] / df['ndsi_mean']) * 100

# Save
csv_path = 'data/snow_stats_2015_2026.csv'
df.to_csv(csv_path, index=False)

print(f'  CSV saved: {csv_path}')
print(f'    Rows (years with data): {len(df)}')
print(f'    Columns: {len(df.columns)}')

# Show summary
print('\nDATA SUMMARY:')
print(df[['year', 'snow_area_km2', 'ndsi_area_km2', 'binary_area_km2',
          'ndsi_mean', 'images_count', 'valid_pct']].to_string(index=False))

# ── Unmixing diagnostic summary ───────────────────────────────────────
print('\nUNMIXING DIAGNOSTIC:')
if len(df) > 0:
    avg_diff = df['method_diff_pct'].mean()
    print(f'  Avg fractional vs binary difference: {avg_diff:+.2f}%')
    print(f'  Range: {df["method_diff_pct"].min():+.2f}% to {df["method_diff_pct"].max():+.2f}%')
    print(f'  Avg snow fraction (AOI): {df["mean_snow_fraction"].mean():.4f}')
    
    # Sanity check: fractional should generally be <= binary
    # (unmixing assigns partial coverage where binary assigns full pixel)
    years_frac_higher = df[df['method_diff_pct'] > 5]
    if len(years_frac_higher) > 0:
        print(f'\n  ⚠ WARNING: {len(years_frac_higher)} years where fractional > binary by >5%')
        print(f'    Years: {years_frac_higher["year"].tolist()}')
        print(f'    This may indicate endmember calibration issues.')
        print(f'    Review endmember spectra against pure pixels in AOI.')

# ============================================================================
# 7. DOWNLOAD DEM (EXTENDED AOI)
# ============================================================================

print('\n' + '='*70)
print('DOWNLOADING DEM (EXTENDED AOI)')
print('='*70 + '\n')

dem = ee.Image('COPERNICUS/DEM/GLO30').clip(aoi_visualization)

try:
    dem_collection = ee.ImageCollection('COPERNICUS/DEM/GLO30')
    dem = dem_collection.select('DEM').mosaic().clip(aoi_visualization)
    
    # Export
    geemap.ee_export_image(
        dem,
        filename='data/humboldt_dem_30m.tif',
        scale=30,
        region=aoi_visualization.getInfo()['coordinates'],
        file_per_band=False
    )
    print('  DEM downloaded: data/humboldt_dem_30m.tif')
    print('    Area: ~28 km² (visualization context)')
    
except Exception as e:
    print(f' Error downloading DEM: {e}')



# ============================================================================
# 9. EXPORT GEOMETRIES (GeoJSON)
# ============================================================================

print('\n' + '='*70)
print('EXPORTING GEOMETRIES (GeoJSON)')
print('='*70 + '\n')

# Small AOI
geojson_aoi = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {
            "name": "Processing AOI",
            "area_km2": 4.09,
            "purpose": "Metrics calculation",
            "description": "Reduced area focused on glacier zone"
        },
        "geometry": aoi_processing.getInfo()
    }]
}

with open('data/aoi_humboldt.geojson', 'w') as f:
    json.dump(geojson_aoi, f, indent=2)
print('  data/aoi_humboldt.geojson')

# EXTENDED AOI
geojson_extended = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {
            "name": "Visualization AOI",
            "area_km2": 25.0,
            "purpose": "Geographic context",
            "description": "Extended area for maps and 3D visualizations"
        },
        "geometry": aoi_visualization.getInfo()
    }]
}

with open('data/aoi_visualization.geojson', 'w') as f:
    json.dump(geojson_extended, f, indent=2)
print(' data/aoi_visualization.geojson')

# ============================================================================
# 8. EXPORT RGB 2025
# ============================================================================

print('\n' + '='*70)
print('EXPORTING RGB 2025')
print('='*70 + '\n')

try:
    if 2025 in composites:
        
        # Reuse already processed composite
        composite_2025 = composites[2025]
        
        # Generate RGB visualization
        rgb_2025 = composite_2025.select(['B4', 'B3', 'B2']).clip(aoi_visualization)
        rgb_2025_vis = rgb_2025.visualize(min=0, max=3000, gamma=1.4)
        
        # Export
        geemap.ee_export_image(
            rgb_2025_vis,
            filename='data/2025_rgb.tif',
            scale=10,
            region=aoi_visualization.getInfo()['coordinates'],
            file_per_band=False
        )
        print('  RGB exported: data/2025_rgb.tif')
        
    elif 2025 in df['year'].values:
        # Fallback: if not in composites but present in data
        print('  Regenerating 2025 composite (was not in cache)...', end=' ')
        composite_2025 = generate_composite(2025)
        
        if composite_2025 is not None:
            rgb_2025 = composite_2025.select(['B4', 'B3', 'B2']).clip(aoi_visualization)
            rgb_2025_vis = rgb_2025.visualize(min=0, max=3000, gamma=1.4)
            
            geemap.ee_export_image(
                rgb_2025_vis,
                filename='data/2025_rgb.tif',
                scale=10,
                region=aoi_visualization.getInfo()['coordinates'],
                file_per_band=False
            )
        else:
            print(' Could not generate composite')
    else:
        print('  Year 2025 not available in data')
        
except Exception as e:
    print(f' Error: {e}')
    import traceback
    traceback.print_exc()


# ============================================================================
# 10. GENERATE COMPLETE METADATA
# ============================================================================

print('\n' + '='*70)
print('GENERATING METADATA')
print('='*70 + '\n')

# Detect adjusted years
adjusted_years = []
if len(df) > 0:
    for _, row in df.iterrows():
        year = int(row['year'])
        end_date = datetime(year, 3, 15)
        if end_date > datetime.now():
            adjusted_years.append(year)

# Calculate statistics
total_years_requested = END_YEAR - START_YEAR + 1
years_processed = len(df)
completeness_pct = (years_processed / total_years_requested) * 100

metadata = {
    # Temporal information
    'period_requested': f'{START_YEAR}-{END_YEAR}',
    'period_effective': f'{int(df["year"].min())}-{int(df["year"].max())}' if len(df) > 0 else 'N/A',
    'total_years_requested': total_years_requested,
    'years_processed': years_processed,
    'years_excluded': omitted_years,
    'years_adjusted': adjusted_years,
    'completeness_percent': round(completeness_pct, 1),
    
    # Temporal interpretation
    'temporal_interpretation': {
        'year_meaning': 'End of dry season window',
        'window_definition': 'December 15 (year-1) to March 15 (year)',
        'rationale': 'Year assigned represents when dry season ends, capturing minimum seasonal snow conditions',
        'examples': {
            2019: 'Dec 15, 2018 to Mar 15, 2019',
            2020: 'Dec 15, 2019 to Mar 15, 2020',
            2025: 'Dec 15, 2024 to Mar 15, 2025',
            2026: 'Dec 15, 2025 to Mar 15, 2026'
        }
    },
    
    # Processing parameters
    'processing_parameters': {
        'temporal_window': {
            'start_formula': 'December 15 (year - 1)',
            'end_formula': 'March 15 (year)',
            'nominal_duration_days': 90,
            'minimum_duration_days': 30,
            'adjustment_rule': 'If end date exceeds present, adjust to current date'
        },
        'ndsi_threshold': NDSI_THRESHOLD,
        'ndsi_role': 'Pre-mask for unmixing candidate zone (>= 0.2) + quality control',
        'ndsi_premask_threshold': NDSI_PREMASK_THRESHOLD,
        'ndsi_premask_rationale': 'Excludes pure rock/vegetation from unmixing to prevent residual fraction accumulation over large AOI',
        'cloud_coverage_max': CLOUD_PCT_MAX,
        'spatial_resolution': f'{SCALE}m',
        'sensor': 'Sentinel-2 SR Harmonized',
        'collection': 'COPERNICUS/S2_SR_HARMONIZED',
        
        # Unmixing documentation
        'area_estimation_method': {
            'primary': 'NDSI-masked linear spectral unmixing (sub-pixel fractional cover)',
            'secondary': 'Binary NDSI threshold (for comparison only)',
            'unmixing_bands': UNMIX_BANDS,
            'endmembers': ENDMEMBERS,
            'endmember_source': 'Calibrated from pure pixels within AOI (2020 composite, n=14 images)',
            'endmember_calibration': 'See data/endmember_calibration.json',
            'spectral_separability': {
                'snow_vs_rock': '43.0°',
                'snow_vs_vegetation': '63.6°',
                'rock_vs_vegetation': '21.1°',
                'assessment': 'All GOOD (>10°)'
            },
            'constraints': ['sumToOne=True', 'nonNegative=True'],
            'polygon_threshold': SNOW_FRACTION_THRESHOLD,
            'references': [
                'Painter et al. (2009) - Retrieval of subpixel snow covered area',
                'Sirguey et al. (2009) - Subpixel monitoring of snow cover'
            ]
        }
    },
    # RGB image
    'rgb_export': {
        'year': 2025,
        'file': 'data/2025_rgb.tif',
        'aoi': 'visualization',
        'bands': ['B4_Red', 'B3_Green', 'B2_Blue'],
        'scale_m': 10,
        'visualization': {
            'min': 0,
            'max': 3000,
            'gamma': 1.4
        },
        'note': 'Used as base for comparative image (Script 02)'
    },
    
    # Areas of interest
    'aoi_processing': {
        'area_km2': 4.09,
        'purpose': 'Metrics calculation',
        'bounds': 'See aoi_humboldt.geojson'
    },
    'aoi_visualization': {
        'area_km2': 25.0,
        'purpose': 'Geographic context',
        'bounds': 'See aoi_visualization.geojson'
    },
    
    # Exclusion reasons
    'exclusion_reasons': {
        'no_images': 'No Sentinel-2 images available in temporal window',
        'window_in_future': 'Entire window in future relative to processing date',
        'window_insufficient': 'Less than 30 days of data available',
        'high_clouds': f'All images exceeded {CLOUD_PCT_MAX}% cloud coverage',
        'note': 'Years excluded are documented - NO fake data generated'
    },
    
    # Data quality
    'data_quality': {
        'avg_images_per_year': round(df['images_count'].mean(), 1) if len(df) > 0 else 0,
        'avg_valid_pixels_pct': round(df['valid_pct'].mean(), 1) if len(df) > 0 else 0,
        'avg_method_diff_pct': round(df['method_diff_pct'].mean(), 2) if len(df) > 0 else 0,
        'avg_snow_fraction': round(df['mean_snow_fraction'].mean(), 4) if len(df) > 0 else 0,
        'years_high_quality': len(df[df['images_count'] >= 7]) if len(df) > 0 else 0,
        'years_medium_quality': len(df[(df['images_count'] >= 5) & (df['images_count'] < 7)]) if len(df) > 0 else 0,
        'years_low_quality': len(df[df['images_count'] < 5]) if len(df) > 0 else 0
    },
    
    # Year classification threshold
    'year_classification': {
        'method': 'Threshold based on permanent glacier area',
        'reference': 'Ramírez et al. (2020)',
        'ramirez_glacier_km2': 0.046,
        'firn_factor': 1.2,
        'threshold_km2': 0.055,
        'justification': 'Ramirez permanent glacier area plus 20% margin for residual firn and methodological differences',
        'interpretation': {
            'wet_years': 'Area > 0.055 km² (seasonal snow + glacier)',
            'dry_years': 'Area ≤ 0.055 km² (glacier + residual firn)'
        }
    },
    # Polygons
    'glacier_polygons': {
        'generated': len(polygon_files),
        'years_available': sorted(list(polygon_files.keys())),
        'files': {str(year): path for year, path in polygon_files.items()},
        'method': f'Binary mask from snow_fraction >= {SNOW_FRACTION_THRESHOLD}, vectorized at 10m'
    },
    
    # Processing metadata
    'processing_info': {
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'gee_project': ' ',
        'python_packages': {
            'earthengine-api': ee.__version__,
            'pandas': pd.__version__
        }
    },
    
    # References
    'references': {
        'ndsi_method': 'Hall et al. (1995) - Remote Sensing of Environment',
        'unmixing_method': 'Painter et al. (2009) - Remote Sensing of Environment',
        'unmixing_validation': 'Sirguey et al. (2009) - Remote Sensing of Environment',
        'validation': 'Ramírez et al. (2020) - Arctic, Antarctic, and Alpine Research',
        'best_practices': 'WGMS, USGS Landsat, ESA Copernicus guidelines'
    }
}

# Save metadata
metadata_path = 'data/processing_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'  Metadata saved: {metadata_path}')

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

print('\n' + '='*70)
print('PROCESSING COMPLETED')
print('='*70 + '\n')

print('PROCESSING STATISTICS:')
print(f'  Years requested: {total_years_requested} ({START_YEAR}-{END_YEAR})')
print(f'  Years processed: {years_processed}')
print(f'  Years omitted: {len(omitted_years)}')
if omitted_years:
    print(f'    → {omitted_years}')
print(f'  Completeness: {completeness_pct:.1f}%')
print(f'    Effective period: {int(df["year"].min())}-{int(df["year"].max())}' if len(df) > 0 else '    Effective period: N/A')

print('\n FILES GENERATED:')
print('    data/snow_stats_2015_2026.csv')
print('    data/humboldt_dem_30m.tif')
print('   data/2025_rgb.tif')
print('    data/aoi_humboldt.geojson')
print('    data/aoi_visualization.geojson')
print('    data/processing_metadata.json')
print(f'   data/glacier_polygons/*.geojson ({len(polygon_files)} files)')

print('\n METHOD:')
print(f'    Primary: Linear spectral unmixing ({len(ENDMEMBERS)} endmembers)')
print(f'    Secondary: Binary NDSI >= {NDSI_THRESHOLD} (quality control)')
print(f'    Polygon threshold: snow_fraction >= {SNOW_FRACTION_THRESHOLD}')

print('\n DATA QUALITY:')
if len(df) > 0:
    print(f'    Average images/year: {df["images_count"].mean():.1f}')
    print(f'    Average valid pixels: {df["valid_pct"].mean():.1f}%')
    print(f'    Avg fractional vs binary diff: {df["method_diff_pct"].mean():+.2f}%')
    print(f'    High quality years (≥7 img): {len(df[df["images_count"] >= 7])}')
    print(f'    Medium quality years (5-6 img): {len(df[(df["images_count"] >= 5) & (df["images_count"] < 7)])}')
    print(f'    Low quality years (<5 img): {len(df[df["images_count"] < 5])}')
    print(f'\nPolygons generated: {len(polygon_files)} years')