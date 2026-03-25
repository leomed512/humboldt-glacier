"""
Endmember calibration for spectral unmixing - Pico Humboldt Glacier

Extracts real spectral signatures from pure pixels within the AOI
to replace library-based endmember estimates.

Output:
  - Prints calibrated endmember values ready to copy into script 01
  - data/endmember_calibration.json (documentation)
"""

import ee
import json
import numpy as np
from datetime import datetime

# ============================================================================
# INITIALIZATION
# ============================================================================

try:
    ee.Initialize(project='pico-umboldt-feb-2019-2026')
    print('GEE initialized')
except Exception as e:
    print(f'Error: {e}')
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Bands used for unmixing (must match script 01)
UNMIX_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

# Reference year: 2020 (dry year, 14 images, known glacier extent)
REF_YEAR = 2020

# ============================================================================
# SAMPLING POINTS
# ============================================================================
# 
# Points identified from visual inspection of the study area.


SAMPLE_POINTS = {
    'snow': [

        ee.Geometry.Point([-70.998033, 8.550038]),
        ee.Geometry.Point([-70.998314, 8.549986]),
        ee.Geometry.Point([-70.997716, 8.550018]),
    ],
    'rock': [
   
        ee.Geometry.Point([-70.99696, 8.54831]),
        ee.Geometry.Point([-71.001101, 8.548985]),
        ee.Geometry.Point([-70.9988, 8.5433]),
    ],
    'vegetation': [
        ee.Geometry.Point([-70.9901, 8.54051]),
    ]
}

# Processing AOI 
aoi_processing = ee.Geometry.Polygon([[
    [-71.0079,   8.55679],
    [-70.988288, 8.55679],
    [-70.988288, 8.53973],
    [-71.0079,   8.53973],
    [-71.0079,   8.55679]
]])

# ============================================================================
# GENERATE REFERENCE COMPOSITE
# ============================================================================


def mask_s2_scl(img):
    scl = img.select('SCL')
    mask = (scl.eq(2).Or(scl.eq(4)).Or(scl.eq(5))
           .Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11)))
    return img.updateMask(mask)

start = ee.Date.fromYMD(REF_YEAR - 1, 12, 15)
end = ee.Date.fromYMD(REF_YEAR, 3, 15)

col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
       .filterBounds(aoi_processing)
       .filterDate(start, end)
       .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 60))
       .map(mask_s2_scl))

n_images = col.size().getInfo()
print(f'  Images in composite: {n_images}')

composite = col.median()

# Add NDSI for validation
ndsi = composite.normalizedDifference(['B3', 'B11']).rename('NDSI')
composite = composite.addBands(ndsi)

# ============================================================================
# EXTRACT ENDMEMBER SPECTRA
# ============================================================================

print('\nExtracting endmember spectra...\n')

bands_to_sample = UNMIX_BANDS + ['NDSI']
calibrated_endmembers = {}
calibration_details = {}

for material, points in SAMPLE_POINTS.items():
    print(f'  {material.upper()}:')
    
    all_values = {band: [] for band in bands_to_sample}
    
    for i, point in enumerate(points):
        # Sample single pixel at each point
        sample = composite.select(bands_to_sample).reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=20
        ).getInfo()
        
        if sample and sample.get(UNMIX_BANDS[0]) is not None:
            for band in bands_to_sample:
                val = sample.get(band)
                if val is not None:
                    all_values[band].append(val)
            
            band_str = ', '.join([f'{sample.get(b, 0):.0f}' for b in UNMIX_BANDS])
            ndsi_val = sample.get('NDSI', 0)
            print(f'    Point {i+1}: [{band_str}] | NDSI: {ndsi_val:.3f}')
        else:
            print(f'    Point {i+1}: NO DATA (masked pixel or outside image)')
    
    # Compute mean across valid samples
    if all(len(all_values[b]) > 0 for b in UNMIX_BANDS):
        mean_spectrum = [round(np.mean(all_values[b])) for b in UNMIX_BANDS]
        std_spectrum = [round(np.std(all_values[b]), 1) for b in UNMIX_BANDS]
        mean_ndsi = np.mean(all_values['NDSI']) if all_values['NDSI'] else None
        
        calibrated_endmembers[material] = mean_spectrum
        calibration_details[material] = {
            'mean_spectrum': mean_spectrum,
            'std_spectrum': std_spectrum,
            'n_samples': len(all_values[UNMIX_BANDS[0]]),
            'mean_ndsi': round(mean_ndsi, 3) if mean_ndsi else None,
            'bands': UNMIX_BANDS
        }
        
        print(f'    → Mean: {mean_spectrum}')
        print(f'    → Std:  {std_spectrum}')
        if mean_ndsi is not None:
            print(f'    → NDSI: {mean_ndsi:.3f}')
    else:
        print(f'    → FAILED: Not enough valid samples')
    
    print()

# ============================================================================
# VALIDATE SPECTRAL SEPARABILITY
# ============================================================================

print('='*60)
print('SPECTRAL SEPARABILITY ANALYSIS')
print('='*60 + '\n')

if len(calibrated_endmembers) == 3:
    materials = list(calibrated_endmembers.keys())
    
    for i in range(len(materials)):
        for j in range(i+1, len(materials)):
            m1, m2 = materials[i], materials[j]
            s1 = np.array(calibrated_endmembers[m1])
            s2 = np.array(calibrated_endmembers[m2])
            
            # Spectral Angle (radians) - measures shape similarity
            cos_angle = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle_deg = np.degrees(np.arccos(cos_angle))
            
            # Euclidean distance - measures magnitude difference
            euclidean = np.linalg.norm(s1 - s2)
            
            # Assessment
            if angle_deg > 10:
                quality = 'GOOD'
            elif angle_deg > 5:
                quality = 'ACCEPTABLE'
            else:
                quality = 'POOR - endmembers too similar'
            
            print(f'  {m1} vs {m2}:')
            print(f'    Spectral angle: {angle_deg:.1f}° ({quality})')
            print(f'    Euclidean dist: {euclidean:.0f}')
            print()
    
    # NDSI validation
    print('NDSI VALIDATION:')
    snow_ndsi = calibration_details.get('snow', {}).get('mean_ndsi')
    rock_ndsi = calibration_details.get('rock', {}).get('mean_ndsi')
    veg_ndsi = calibration_details.get('vegetation', {}).get('mean_ndsi')
    
    if snow_ndsi is not None:
        print(f'  Snow NDSI:  {snow_ndsi:.3f}  {"✓ Expected > 0.4" if snow_ndsi > 0.4 else "⚠ LOW - verify snow point"}')
    if rock_ndsi is not None:
        print(f'  Rock NDSI:  {rock_ndsi:.3f}  {"✓ Expected < 0.2" if rock_ndsi < 0.2 else "⚠ HIGH - may contain snow"}')
    if veg_ndsi is not None:
        print(f'  Veg NDSI:   {veg_ndsi:.3f}  {"✓ Expected < 0.1" if veg_ndsi < 0.1 else "⚠ HIGH - check point location"}')

else:
    print('  Cannot compute separability: missing endmembers')

# ============================================================================
# OUTPUT
# ============================================================================

print('\n' + '='*60)
print('CALIBRATED ENDMEMBERS')
print('='*60 + '\n')

if len(calibrated_endmembers) == 3:
    print('ENDMEMBERS = {')
    for material, spectrum in calibrated_endmembers.items():
        padding = ' ' * (15 - len(material))
        print(f"    '{material}':{padding}{spectrum},")
    print('}')
    
    # Save calibration report
    report = {
        'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'reference_year': REF_YEAR,
        'reference_images': n_images,
        'bands': UNMIX_BANDS,
        'endmembers': calibrated_endmembers,
        'details': calibration_details,
        'sample_points': {
            material: [p.getInfo()['coordinates'] for p in points]
            for material, points in SAMPLE_POINTS.items()
        },
        'notes': [
            'Values are mean surface reflectance from Sentinel-2 SR',
            'Extracted from median composite of dry season window',
            'Review if processing a different AOI or sensor'
        ]
    }
    
    with open('data/endmember_calibration.json', 'w') as f:
        json.dump(report, f, indent=2)
    print('\nCalibration saved: data/endmember_calibration.json')
    
else:
    print('CALIBRATION INCOMPLETE - check sampling points')
