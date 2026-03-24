"""
Interactive dashboard with Streamlit

Run:
  streamlit run scripts/03_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import rasterio
import json
from pathlib import Path
import geopandas as gpd
from streamlit_js_eval import streamlit_js_eval

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Glaciar Pico Humboldt",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
""", unsafe_allow_html=True)

# ============================================================================
# CSS STYLING
# ============================================================================

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>

    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }

    html, body, [class*="css"], 
    [data-testid="stAppViewContainer"],
    [data-testid="stMarkdownContainer"],
    p, div, span, label {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        letter-spacing: -0.1px;
    }

    .block-container {
        max-width: 1000px;
        padding: 2rem;
    }

  
    h1 {
        font-size: 2.5rem !important;
        font-weight: bolder !important;
        line-height: 0.5 !important;
        letter-spacing: -5px;
        text-align: center;
    }

    h2 {
        font-size: 1.6rem !important;
        font-weight: 800;
        line-height: 1.3;
        letter-spacing: -0.3px;
        margin-top: 1.5rem;
        text-align: center;
    }

    h3 {
        font-size: 1.2rem;
        font-weight: 500;
    }

    p {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #c9d1d9;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 600;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        color: #9aa0a6;
    }

    hr {
        margin: 2rem 0;
        border-color: rgba(255, 255, 255, 0.1);
    }

    @media (max-width: 1200px) {
        .block-container {
            max-width: 90%;
            padding: 1.5rem;
        }
    }


    @media (max-width: 768px) {

        .block-container {
            max-width: 95%;
            padding: 0.7rem;
        }

        h1 {
            font-size: 1.6rem !important;
            line-height: 1.2 !important;
        }

        h2 {
            font-size: 1.5rem !important;
            line-height: 1.25 !important;
        }

        h3 {
            font-size: 1.05rem !important;
        }

        p {
            font-size: 0.8rem !important;
            line-height: 1.5 !important;
        }
        li{
             font-size: 0.8rem !important;
            line-height: 1.5 !important;
            }

        /* Metrics adjustment */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
         h1 .material-icons {
            font-size: 2rem !important;
            margin-right: 5px !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }

        /* Icons inside headings */
        h2 .material-icons {
            font-size: 1.1rem !important;
            vertical-align: middle;
        }
    }


</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS (WITH CACHE)
# ============================================================================

@st.cache_data
def load_data():
    """Load annual metrics CSV"""
    df = pd.read_csv('data/snow_stats_2015_2026.csv')
    df['ndsi_cv'] = (df['ndsi_std'] / df['ndsi_mean']) * 100
    return df

@st.cache_data
def load_metadata():
    """Load metadata with Model B info"""
    try:
        with open('data/processing_metadata.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_data
def load_dem():
    """Load DEM for 3D visualization"""
    try:
        with rasterio.open('data/humboldt_dem_30m.tif') as src:
            dem = src.read(1)
            bounds = src.bounds
        return dem, bounds
    except:
        return None, None

@st.cache_data
def load_polygon(year):
    """Load glacier GeoJSON polygon"""
    try:
        polygon_path = f'data/glacier_polygons/{year}.geojson'
        if Path(polygon_path).exists():
            return gpd.read_file(polygon_path)
        return None
    except:
        return None

# ============================================================================
# COMPONENT: INDEPENDENT YEAR SELECTOR
# ============================================================================

def year_selector(key_prefix, label="Seleccionar año"):
    """
    Independent year selector for each section.
    Each slider controls only its own section.
    
    Args:
        key_prefix: Unique prefix ('kpi', 'map', '3d')
        label: Selector label (user-facing, kept in Spanish)
    
    Returns:
        Selected year for this section
    """
    st.markdown(f"**{label}:**")
    
    # Section-specific session state variable
    state_key = f'selected_year_{key_prefix}'
    
    # Create slider
    selected = st.select_slider(
        label=f"slider_{key_prefix}",
        options=available_years,
        value=st.session_state[state_key],
        key=f'{key_prefix}_slider',
        label_visibility="collapsed"
    )
    
    # Update section-specific state
    st.session_state[state_key] = selected
    
    return selected

# ============================================================================
# INITIALIZATION
# ============================================================================

# Load data
df = load_data()
metadata = load_metadata()

# Available years
available_years = sorted(df['year'].tolist())
year_min = int(df['year'].min())
year_max = int(df['year'].max())

# Wet/dry classification
dry_threshold = 0.060
df['year_type'] = df['snow_area_km2'].apply(
    lambda x: 'Húmedo' if x > dry_threshold else 'Seco'
)

# Load Model B info
model_b = metadata.get('model_b', {})
model_b_rate = model_b.get('rate_percent_per_year', -26.9)
model_b_r2 = model_b.get('r_squared', 0.86)
model_b_years = model_b.get('years_used', [2020, 2021, 2022, 2023, 2024, 2025])

# ============================================================================
# SESSION STATE - INDEPENDENT SELECTORS
# ============================================================================

# KPI selector
if 'selected_year_kpi' not in st.session_state:
    st.session_state.selected_year_kpi = year_max

# Map selector
if 'selected_year_map' not in st.session_state:
    st.session_state.selected_year_map = year_max

# 3D selector
if 'selected_year_3d' not in st.session_state:
    st.session_state.selected_year_3d = year_max

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<h1>
<span class="material-icons" style="vertical-align: middle; margin-right:8px; font-size:4rem;">terrain</span>
Glaciar La Corona
</h1>
""", unsafe_allow_html=True)

st.markdown(f"### Monitoreo del Último glaciar de Venezuela · {year_min}-{year_max}")


st.markdown("""
Durante el último siglo el glaciar en el Pico Humboldt ha enfrentado una reducción acelerada de su superficie 
debido al aumento de las temperaturas en los Andes Tropicales, uniéndose a las extintas *Nieves Eternas*
de la cordillera de Mérida como los existentes en el Pico Bolívar, La Concha, El Toro y León.

Con este escenario Venezuela se convierte en el primer país tropical en despedirse de sus glaciares, 
abriendo interrogantes sobre la dinámica ecológica en estos ecosistemas de alta montaña y cómo se relaciona
con el cambio climático.
            
Este estudio combina el análisis de imágenes satelitales Sentinel-2, Modelo Digital de Terreno de Copernicus GLO-30 y validación por observaciones de campo y fotografías proporcionadas por guías de la asociación [UGAM](https://www.instagram.com/ugamvenezuela/)
""")

st.markdown("---")

# ============================================================================
# DYNAMIC KPIs WITH SELECTOR
# ============================================================================

st.markdown("""
<h2><span class="material-icons">analytics</span> Estadísticas</h2>
""", unsafe_allow_html=True)

# Year selector
selected_year = year_selector('kpi', 'Seleccionar año para análisis')

# Selected year data
area_selected = df.loc[df['year'] == selected_year, 'snow_area_km2'].values[0]
type_selected = df.loc[df['year'] == selected_year, 'year_type'].values[0]
ndsi_selected = df.loc[df['year'] == selected_year, 'ndsi_mean'].values[0]
images_selected = df.loc[df['year'] == selected_year, 'images_count'].values[0]

# Loss relative to 2020
if 2020 in df['year'].values:
    area_2020 = df.loc[df['year'] == 2020, 'snow_area_km2'].values[0]
    loss_vs_2020 = area_selected - area_2020
else:
    loss_vs_2020 = 0

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    f"Área {selected_year}", 
    f"{area_selected:.4f} km²", 
    f"{loss_vs_2020:+.4f} km²",
    help=f"Área del glaciar en {selected_year}. Delta respecto a 2020."
)

col2.metric(
    "Tipo de Año", 
    type_selected,
    None,
    help="Clasificación según umbral de 0.060 km². Húmedo: nieve estacional presente."
)

col3.metric(
    "NDSI Promedio", 
    f"{ndsi_selected:.3f}",
    f"{images_selected} imágenes",
    help="Índice de Diferencia Normalizada de Nieve. Basado en imágenes Sentinel-2."
)

col4.metric(
    "Modelo (Secos)",
    f"{model_b_rate:.2f}%/año",
    f"R² = {model_b_r2:.2f}",
    help="Tasa anual de retroceso según modelo exponencial ajustado a años secos."
)

with st.expander("Metodología del estudio", icon=":material/info:"):
    st.markdown("""
Este estudio muestra la dinámica reciente del área de hielo y nieve utilizando 
imágenes satelitales Sentinel-2 y análisis espectral mediante el índice de diferencia 
normalizada de nieve (NDSI). 

De manera automatizada se consultaron decenas de imágenes Sentinel-2 desde la base de datos en la nube de Google Earth Engine (GEE) utilizando Python
y se estimó y validó la superficie de nieve/hielo presente. Se evaluaron datos desde 2019 hasta 2026, (8 años) durante la época de sequía (diciembre-marzo), cuando 
la cobertura de hielo es más representativa, al ser menos influenciada por nieve temporal, mostrando un aproximado realista de la superficie glaciar. Con los datos disponibles se estimó el año en el glaciar pasaría a la 
categoría de parche de hielo o remanente glaciar y cuando podría desaparecer este remanente.
                
Cabe resaltar que el planteamiento original era obtener datos satelitales desde 2015 pero la zona presentó nubosidad muy elevada, lo que dificultó la adquisición de imágenes satelitales viables para el periodo 2015-2018. Por otra parte, los años 2019 y 2026 presentaron una cobertura de nieve estacional muy elevada, por lo que fueron excluídos del análisis más profundo. Estos dos años estuvieron influenciados por una temporada de lluvias especialmente elevada, lo fue confirmado por guías de montaña [UGAM](https://www.instagram.com/ugamvenezuela/) a través de fotografías e información de campo.

Los datos fueron contrastados con estudios similares recientes como Ramírez et al. (2020) y productos Sentinel-2 de la Agencia Espacial Europea (ESA).Teniendo en cuenta las diferencias metodológicas entre estas fuentes, se obtuvieron resultados aceptables para ser un estudio exploratorio, apoyando a la continuidad del 
monitoreo de este emblemático ecosistema andino con herramientas modernas de análisis geoespacial.
""")

st.markdown("---")

# ============================================================================
# 2D INTERACTIVE MAP WITH SELECTOR
# ============================================================================

st.markdown("""
<h2><span class="material-icons">map</span> Mapa Interactivo</h2>
""", unsafe_allow_html=True)

# Year selector
current_year = year_selector('map', 'Seleccionar año')

# Get selected year data for the map
area_map = df.loc[df['year'] == current_year, 'snow_area_km2'].values[0]
ndsi_map = df.loc[df['year'] == current_year, 'ndsi_mean'].values[0]
type_map = df.loc[df['year'] == current_year, 'year_type'].values[0]

# Create map
m = folium.Map(
    location=[8.5505, -70.998],
    zoom_start=15,
    control_scale=True,
    tiles=None
)

# Satellite base layer
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Satélite',
    overlay=False,
    control=True,
    show=True
).add_to(m)

# Topographic layer
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Terreno',
    overlay=False,
    control=True,
    show=False
).add_to(m)

# Glacier polygon
glacier_layer = folium.FeatureGroup(
    name=f'Glaciar {current_year}', 
    overlay=True, 
    control=True, 
    show=True
)

polygon_gdf = load_polygon(current_year)

if polygon_gdf is not None:
    folium.GeoJson(
        polygon_gdf,
        style_function=lambda x: {
            'fillColor': '#3498db',
            'color': '#2980b9',
            'weight': 2,
            'fillOpacity': 0.7
        },
        tooltip=folium.Tooltip(
            f'Glaciar {current_year}: {area_map:.4f} km²', 
            sticky=True
        )
    ).add_to(glacier_layer)
else:
    radius_m = np.sqrt(area_map * 1e6 / np.pi)
    folium.Circle(
        location=[8.5505, -70.998],
        radius=radius_m,
        color='#2980b9',
        fill=True,
        fillColor='#3498db',
        fillOpacity=0.7,
        weight=2,
        tooltip=f'Glaciar {current_year} (aprox.)'
    ).add_to(glacier_layer)

glacier_layer.add_to(m)

# Peak marker
markers_layer = folium.FeatureGroup(
    name='Pico Humboldt', 
    overlay=True, 
    control=True, 
    show=True
)

folium.Marker(
    location=[8.5505, -70.998],
    popup=folium.Popup(f"""
    <div style='font-family: Arial; min-width: 150px;'>
        <h4 style='margin:0; color:#2c3e50;'>Pico Humboldt</h4>
        <hr style='margin: 5px 0;'>
        <b>Año:</b> {current_year}<br>
        <b>Área:</b> {area_map:.4f} km²<br>
        <b>NDSI:</b> {ndsi_map:.2f}<br>
        <b>Tipo:</b> {type_map}
    </div>
    """, max_width=200),
    icon=folium.Icon(color='red', icon='mountain', prefix='fa')
).add_to(markers_layer)

markers_layer.add_to(m)

# Layer control
folium.LayerControl(position='topright', collapsed=True).add_to(m)

# CSS for semi-transparent control
m.get_root().html.add_child(folium.Element("""
<style>
.leaflet-control-layers {
    background: rgba(255, 255, 255, 0.75) !important;
    backdrop-filter: blur(5px);
}
.leaflet-control-layers-expanded {
    background: rgba(255, 255, 255, 0.7) !important;
}
</style>
"""))

# Render
st_folium(m, width=None, height=600, returned_objects=[])

st.caption(
    f"**Año {current_year}:** {area_map:.4f} km² | "
    f"NDSI: {ndsi_map:.2f} | "
    f"Tipo: {type_map}"
)

st.markdown("---")

# ============================================================================
# CHART: TEMPORAL EVOLUTION
# ============================================================================

st.markdown("""
<h2><span class="material-icons">insights</span> Evolución Temporal</h2>
""", unsafe_allow_html=True)
screen_width = streamlit_js_eval(
    js_expressions='window.innerWidth',
    key='SCREEN_WIDTH_TEMPORAL',
    want_output=True
)

if screen_width is None:
    screen_width = 700

is_mobile = screen_width < 700

left_margin = 0 if is_mobile else 70
right_margin = 15 if is_mobile else 50
fig = go.Figure()

# Separate years by type
df_wet = df[df['year_type'] == 'Húmedo']
df_dry = df[df['year_type'] == 'Seco']

# Adjusted scale
y_min = df['snow_area_km2'].min() * 0.85
y_max = df['snow_area_km2'].max() * 1.05

# Trend line
fig.add_trace(go.Scatter(
    x=df['year'],
    y=df['snow_area_km2'],
    mode='lines',
    name='Tendencia observada',
    line=dict(color='#95a5a6', width=1.5, dash='dot'),
    showlegend=False,
    hoverinfo='skip'
))

# Dry years
fig.add_trace(go.Scatter(
    x=df_dry['year'],
    y=df_dry['snow_area_km2'],
    mode='lines+markers+text',
    name='Años secos',
    marker=dict(size=12, color='#e74c3c', symbol='circle'),
    line=dict(color='#e74c3c', width=2, dash='dot'),
    text=None if is_mobile else [f'{area:.4f}' for area in df_dry['snow_area_km2']],
    textposition='top center',
    textfont=dict(size=9, color='#e74c3c'),
    hovertemplate='<b>%{x}</b><br>Área: %{y:.4f} km²<extra></extra>'
))

# Wet years
fig.add_trace(go.Scatter(
    x=df_wet['year'],
    y=df_wet['snow_area_km2'],
    mode='markers+text',
    name='Años húmedos',
    marker=dict(size=8, color='#20959e', symbol='x'),
    text=None if is_mobile else [f'{area:.4f}' for area in df_dry['snow_area_km2']],
    textposition='top center',
    textfont=dict(size=9, color='#20959e'),
    hovertemplate='<b>%{x}</b><br>Área: %{y:.4f} km²<extra></extra>'
))

# Ramírez data points
ramirez_years = [2018, 2019]
ramirez_areas = [0.079, 0.046]

fig.add_trace(go.Scatter(
    x=ramirez_years,
    y=ramirez_areas,
    mode='markers',
    name='Ramírez et al. 2020',
    marker=dict(size=10, color='#27ae60', symbol='diamond'),
    hovertemplate='<b>%{x}</b> (Ramírez)<br>Área: %{y:.3f} km²<extra></extra>'
))

# Model B
if 'parameters' in model_b:
    params = model_b['parameters']
    a, k, t0 = params['a'], params['k'], params['t0']
    
    years_fit = np.linspace(min(model_b_years), max(model_b_years) + 0.6, 100)
    area_fit = a * np.exp(k * (years_fit - t0))
    
    fig.add_trace(go.Scatter(
        x=years_fit,
        y=area_fit,
        mode='lines',
        name=f'Modelo exponencial ({model_b_rate:.2f}%/año)',
        line=dict(color='#f39c12', width=2),
        hovertemplate='Año: %{x:.1f}<br>Proyección: %{y:.4f} km²<extra></extra>'
    ))

# Layout configuration
fig.update_layout(
    xaxis=dict(
        title=dict(
            text='Año',
            font=dict(size=13, color='#1a1a1a')
        ),
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.4)',
        showline=True,
        linewidth=2,
        linecolor='#2c3e50',
        mirror=True,
        range=[2017.7, 2026.6],
        tickfont=dict(size=11, color='#2c3e50', family='Arial')
    ),
    yaxis=dict(
        title=dict(
            text='Área de nieve (km²)' if not is_mobile else '',
            font=dict(size=13, color='#1a1a1a')
        ),
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.4)',
        showline=True,
        linewidth=2,
        linecolor='#2c3e50',
        mirror=True,
        range=[y_min - 0.009, y_max + 0.020],
        tickfont=dict(size=11, color='#2c3e50', family='Arial')
    ),
    hovermode='x unified',
    height=500,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='#2c3e50', size=12, family='Arial'),
    legend=dict(
        orientation="h",
        yanchor="top" if is_mobile else "bottom",
        y=-0.22 if is_mobile else 1.02,
        xanchor="center" if is_mobile else "right",
        x=0.5 if is_mobile else 1,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#34495e',
        borderwidth=2,
        font=dict(size=11, color='#2c3e50', family='Arial')
    ),
    margin=dict(l=left_margin , r=right_margin, t=40, b=50)
)

st.plotly_chart(fig, width='stretch')

st.markdown(
    f"**Nota:** Años húmedos ({', '.join(map(str, df_wet['year'].tolist()))}) "
    f"presentan nieve estacional inusualmente abundante, afectando predicciones. "
    f"Por lo tanto el modelo predictivo fue ajustado a años secos, con buena precisión (R² = {model_b_r2:.2f}).\n\n"
    f"Puntos verdes: Ramírez et al. (2020), fotointerpretación manual submétrica. La diferencia de +28,4% respecto a este estudio (0.058 vs 0.046 km²) es esperable dada la resolución de Sentinel-2 (10-20 m) y la diferencia interanual (2020 vs 2019). Ambos resultados convergen en la tendencia de retroceso acelerado. "
)

st.warning(
    f"Según este estudio, La Corona dejaría de considerarse un Glaciar aproximadamente en 2025 (±2 años) "
    f"para pasar a ser un Parche de Hielo según Benn & Evans (2010).\n\n"
    f"Desaparición total estimada: aproximadamente 2040 (±3 años)."
)

st.markdown("---")

# ============================================================================
# 3D TOPOGRAPHY WITH SELECTOR
# ============================================================================
# Detect browser screen width
screen_width = streamlit_js_eval(js_expressions='window.innerWidth', key='SCREEN_WIDTH', want_output=True)

# Fallback in case value is not returned on first render
if screen_width is None:
    screen_width = 700

is_mobile = screen_width <= 700

st.markdown("""
<h2><span class="material-icons">satellite</span> Topografía 3D</h2>
""", unsafe_allow_html=True)

# Year selector
current_year_3d = year_selector('3d', 'Seleccionar año')

dem, bounds = load_dem()

if dem is not None and bounds is not None:
    rows, cols = dem.shape
    x = np.linspace(bounds.left, bounds.right, cols)
    y = np.linspace(bounds.bottom, bounds.top, rows)
    x_grid, y_grid = np.meshgrid(x, y)

    fig_3d = go.Figure()
    
    # Surface with topographic palette
    fig_3d.add_trace(go.Surface(
        x=x_grid,
        y=y_grid,
        z=dem,
        colorscale='Earth',
        showscale=not is_mobile, # Better UI on mobile
        colorbar=dict(
        title='Elevación (m)',
        x=1.1,
        tickfont=dict(size=10)
    ) if not is_mobile else None,
        hovertemplate='Elev: %{z:.0f} m<extra></extra>',
        name='DEM'
    ))
    
    # Glacier polygon in 3D
    polygon_gdf_3d = load_polygon(current_year_3d)
    if polygon_gdf_3d is not None:
        for geom in polygon_gdf_3d.geometry:
            if geom.geom_type == 'Polygon':
                coords = list(geom.exterior.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                
                # Interpolate elevation
                elevations = []
                for lon, lat in zip(lons, lats):
                    i = np.argmin(np.abs(y - lat))
                    j = np.argmin(np.abs(x - lon))
                    elevations.append(dem[i, j] + 15)
                
                # 3D glacier mesh
                fig_3d.add_trace(go.Mesh3d(
                    x=lons,
                    y=lats,
                    z=elevations,
                    color='cyan',
                    opacity=0.6,
                    name=f'Glaciar {current_year_3d}',
                    showscale=False,
                    hovertemplate=f'Glaciar {current_year_3d}<extra></extra>'
                ))

    # Layout configuration
    fig_3d.update_layout(
        scene=dict(
            zaxis_title='Elevación (m)',
            camera=dict(
                eye=dict(x=-0.7, y=-0.7, z=0.7) if not is_mobile else dict(x=-0.9, y=-0.9, z=0.55),

                center=dict(x=0, y=0, z=-0.1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3)


        ),
        height=600,
        template='plotly_dark',
        showlegend=False
    )

    st.plotly_chart(fig_3d, width='stretch')
    st.caption(f"DEM: Copernicus GLO-30 (30 m) | Polígono cyan: Glaciar {current_year_3d}")
else:
    st.error("DEM no disponible")

st.markdown("---")


# ============================================================================
# COMPARATIVE IMAGE 2020 vs 2025
# ============================================================================

st.markdown("""
<h2><span class="material-icons">photo_library</span> Comparación 2020 vs 2025</h2>
""", unsafe_allow_html=True)

comparative_img_path = 'results/plots/comparison_2020_2025.png'

if Path(comparative_img_path).exists():
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image(
            comparative_img_path,
            caption="Vista previa",
            width="content"
        )
    
   
else:
    st.warning("Imagen comparativa no disponible. Ejecuta Script 02.")

st.markdown("---")

# ============================================================================
# DISAPPEARANCE PROJECTION
# ============================================================================

st.markdown("""
<h2><span class="material-icons">trending_down</span> Proyección de Desaparición</h2>
""", unsafe_allow_html=True)

if 'projections' in model_b:
    proj = model_b['projections']
    
    # Glacier threshold
    if 'glacier_threshold' in proj and proj['glacier_threshold'].get('valid', False):
        glacier_proj = proj['glacier_threshold']
        glacier_year = glacier_proj['year']
        uncertainty = glacier_proj.get('uncertainty_years', 2)
        
        st.warning(
            f"**Proyección:** Con tasa actual de {model_b_rate:.2f}%/año, "
            f"el glaciar dejará de calificar como tal en **{int(glacier_year)}** "
            f"(±{uncertainty} años). Pasará a ser un parche de hielo.",
            icon=":material/warning:"
        )
    
        st.info(
            f"""
            **Metodología:** Modelo exponencial ajustado a años secos ({min(model_b_years)}-{max(model_b_years)})  
            **Precisión:** R² = {model_b_r2:.3f} | RMSE = ±{model_b.get('uncertainty', {}).get('model_rmse_km2', 0.0):.4f} km² | IC 95% = ±{model_b.get('uncertainty', {}).get('confidence_interval_95pct_km2', 0.0):.4f} km²  
            """,
            icon=":material/info:"
        )

    # Total disappearance
    if 'total_disappearance' in proj and proj['total_disappearance'].get('valid', False):
        disappearance_proj = proj['total_disappearance']
        disappearance_year = disappearance_proj['year']
        disappearance_uncertainty = disappearance_proj.get('uncertainty_years', 3)
        
        st.error(
            f"**Desaparición total estimada:** aproximadamente **{int(disappearance_year)}** "
            f"(±{disappearance_uncertainty} años) si continúa la tendencia.",
            icon=":material/error:"
        )
else:
    st.warning(
        "Proyecciones no disponibles. Verifica que Script 02 haya actualizado metadata correctamente.",
        icon=":material/warning:"
    )

# ============================================================================
# MODEL COMPARISON
# ============================================================================

with st.expander("Comparación de modelos alternativos", icon=":material/compare:"):
    st.markdown("""
    Se evaluaron 3 modelos de regresión para proyectar el retroceso glaciar. 
    El modelo exponencial fue seleccionado por presentar el **menor AIC** 
    (Akaike Information Criterion), indicando el mejor balance entre ajuste y complejidad.
    """)
    
    # Check if comparison file exists
    comp_path = Path('results/model_comparison.csv')
    
    if comp_path.exists():
        df_comp = pd.read_csv(comp_path)
        
        # Format table
        df_comp_display = df_comp[['model', 'params', 'r2', 'rmse', 'aic', 'rate_of_change']].copy()
        df_comp_display.columns = ['Modelo', 'Parámetros', 'R²', 'RMSE (km²)', 'AIC', 'Tasa (%/año)']
        
        # Round values
        df_comp_display['R²'] = df_comp_display['R²'].round(4)
        df_comp_display['RMSE (km²)'] = df_comp_display['RMSE (km²)'].round(6)
        df_comp_display['AIC'] = df_comp_display['AIC'].round(2)
        df_comp_display['Tasa (%/año)'] = df_comp_display['Tasa (%/año)'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )
        
        # Highlight selected model
        st.dataframe(
            df_comp_display,
            hide_index=True,
            width='stretch',
            height=150
        )
        
        st.caption(
            "**AIC (Akaike Information Criterion):** Menor valor = mejor modelo. "
            "Penaliza complejidad (número de parámetros) favoreciendo parsimonia."
        )
    else:
        st.warning("Tabla de comparación no disponible.")
    
    # Technical explanation
    st.markdown("""
    ### Interpretación
    
    **Modelo Exponencial (seleccionado):**
    - Representa retroceso acelerado típico de glaciares en fase terminal
    - 3 parámetros: área inicial (a), tasa de cambio (k), año base (t₀)
    - R² = 0.86 indica que explica 86% de la variabilidad observada
    
    **Modelo Lineal:**
    - Asume retroceso constante (no acelerado)
    - Menos parámetros (2) pero peor ajuste
    - No captura dinámica exponencial de derretimiento
    
    **Modelo Polinomial:**
    - Mismo número de parámetros que exponencial
    - AIC mayor indica sobreajuste sin ganancia explicativa
    """)

st.markdown("---")

# ============================================================================
# RAMÍREZ VALIDATION
# ============================================================================

st.markdown("""
<h2><span class="material-icons">verified</span> Validación Comparativa</h2>
""", unsafe_allow_html=True)

st.markdown("""
Este análisis se comparó con **Ramírez et al. (2020)**,
*"The end of the eternal snows: Integrative mapping of 100 years of glacier retreat in the Venezuelan Andes"*,
publicado en *Arctic, Antarctic, and Alpine Research*

La diferencia del +28.4% en área detectada (0.058 km² en 2020 vs. 0.046 km² en 2019 de Ramírez et al. 2020) refleja las limitaciones esperadas de comparar fotogrametría submétrica con análisis espectral (NDSI) con imágenes satelitales Sentinel-2 (10-20m), el cual puede sobrestimar la superficie glaciar debido a nieve estacional. 
            
Sin embargo, ambos estudios **convergen en la conclusión crítica**: 
el glaciar está en fase terminal con retroceso acelerado en un rango equivalente.


Este estudio establece un protocolo de **monitoreo continuo y reproducible** 
que complementa —no reemplaza— estudios de alta precisión, permitiendo alertas tempranas 
sobre cambios críticos que justifiquen campañas especializadas.
""")

st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown(
    """
    <div style="
        font-size: 1rem;
        color: #9aa0a6;
        text-align: center;
        margin-top: 20px;
    ">
        <a href="https://github.com/leomed512/humboldt-glacier" 
           target="_blank" 
           style="text-decoration: none; color: #9aa0a6;">
            GitHub • Leonardo Medina
        </a>
    </div>
    """,
    unsafe_allow_html=True
)