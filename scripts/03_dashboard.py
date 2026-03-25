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
    """Load metadata with model info"""
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
    """Independent year selector for each section."""
    st.markdown(f"**{label}:**")
    state_key = f'selected_year_{key_prefix}'
    selected = st.select_slider(
        label=f"slider_{key_prefix}",
        options=available_years,
        value=st.session_state[state_key],
        key=f'{key_prefix}_slider',
        label_visibility="collapsed"
    )
    st.session_state[state_key] = selected
    return selected

# ============================================================================
# INITIALIZATION
# ============================================================================

df = load_data()
metadata = load_metadata()

available_years = sorted(df['year'].tolist())
year_min = int(df['year'].min())
year_max = int(df['year'].max())

dry_threshold = 0.060
df['year_type'] = df['snow_area_km2'].apply(
    lambda x: 'Húmedo' if x > dry_threshold else 'Seco'
)

df_wet = df[df['year_type'] == 'Húmedo']
df_dry = df[df['year_type'] == 'Seco']

# Load model info
model_info = metadata.get('model', {})
model_rate = model_info.get('rate_pct_year', -30.5)
model_r2 = model_info.get('r2', 0.87)
model_years = model_info.get('years_used', [2020, 2021, 2022, 2023, 2024, 2025])
model_projections = model_info.get('projections', {})

# ============================================================================
# SESSION STATE
# ============================================================================

if 'selected_year_kpi' not in st.session_state:
    st.session_state.selected_year_kpi = year_max
if 'selected_year_map' not in st.session_state:
    st.session_state.selected_year_map = year_max
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
debido al aumento de las temperaturas y cambios en los patrones de precipitaciones en los Andes Tropicales, uniéndose a las extintas *Nieves Eternas*
de la cordillera de Mérida como los existentes en el Pico Bolívar, La Concha, El Toro y León.

Con este escenario Venezuela se convierte en el primer país tropical en despedirse de sus glaciares, 
abriendo interrogantes sobre la dinámica ecológica en estos ecosistemas de alta montaña y cómo se relaciona
con el cambio climático.
            
Este estudio combina el análisis de imágenes satelitales Sentinel-2, pre-máscara de NDSI, desmezclado espectral sub-píxel, DEM de Copernicus GLO-30 y observaciones de campo proporcionadas por guías de la asociación [UGAM](https://www.instagram.com/ugamvenezuela/)
""")

st.markdown("---")


# ============================================================================
# 2D INTERACTIVE MAP WITH SELECTOR
# ============================================================================

st.markdown("""
<h2><span class="material-icons">map</span> Mapa Interactivo</h2>
""", unsafe_allow_html=True)

current_year = year_selector('map', 'Seleccionar año')

area_map = df.loc[df['year'] == current_year, 'snow_area_km2'].values[0]
ndsi_map = df.loc[df['year'] == current_year, 'ndsi_mean'].values[0]
type_map = df.loc[df['year'] == current_year, 'year_type'].values[0]

m = folium.Map(location=[8.5505, -70.998], zoom_start=15, control_scale=True, tiles=None)

folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Satélite', overlay=False, control=True, show=True
).add_to(m)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Terreno', overlay=False, control=True, show=False
).add_to(m)

glacier_layer = folium.FeatureGroup(name=f'Glaciar {current_year}', overlay=True, control=True, show=True)
polygon_gdf = load_polygon(current_year)

if polygon_gdf is not None:
    folium.GeoJson(
        polygon_gdf,
        style_function=lambda x: {'fillColor': '#3498db', 'color': '#2980b9', 'weight': 2, 'fillOpacity': 0.7},
        tooltip=folium.Tooltip(f'Glaciar {current_year}: {area_map:.4f} km²', sticky=True)
    ).add_to(glacier_layer)
else:
    radius_m = np.sqrt(area_map * 1e6 / np.pi)
    folium.Circle(location=[8.5505, -70.998], radius=radius_m, color='#2980b9',
                  fill=True, fillColor='#3498db', fillOpacity=0.7, weight=2,
                  tooltip=f'Glaciar {current_year} (aprox.)').add_to(glacier_layer)

glacier_layer.add_to(m)

markers_layer = folium.FeatureGroup(name='Pico Humboldt', overlay=True, control=True, show=True)
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

folium.LayerControl(position='topright', collapsed=True).add_to(m)
m.get_root().html.add_child(folium.Element("""
<style>
.leaflet-control-layers { background: rgba(255, 255, 255, 0.75) !important; backdrop-filter: blur(5px); }
.leaflet-control-layers-expanded { background: rgba(255, 255, 255, 0.7) !important; }
</style>
"""))

st_folium(m, width=None, height=600, returned_objects=[])

st.caption(
    f"**Año {current_year}:** {area_map:.4f} km² | "
    f"NDSI: {ndsi_map:.2f} | Tipo: {type_map}"
)

st.markdown("---")

# ============================================================================
# DYNAMIC KPIs WITH SELECTOR
# ============================================================================

st.markdown("""
<h2><span class="material-icons">analytics</span> Estadísticas</h2>
""", unsafe_allow_html=True)

selected_year = year_selector('kpi', 'Seleccionar año para análisis')

area_selected = df.loc[df['year'] == selected_year, 'snow_area_km2'].values[0]
type_selected = df.loc[df['year'] == selected_year, 'year_type'].values[0]
ndsi_selected = df.loc[df['year'] == selected_year, 'ndsi_mean'].values[0]
images_selected = df.loc[df['year'] == selected_year, 'images_count'].values[0]

if 2020 in df['year'].values:
    area_2020_ref = df.loc[df['year'] == 2020, 'snow_area_km2'].values[0]
    loss_vs_2020 = area_selected - area_2020_ref
else:
    loss_vs_2020 = 0

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    f"Área {selected_year}", f"{area_selected:.4f} km²", f"{loss_vs_2020:+.4f} km²",
    help=f"Área del glaciar en {selected_year}. Delta respecto a 2020."
)
col2.metric(
    "Tipo de Año", type_selected, None,
    help="Clasificación según umbral de 0.060 km². Húmedo: nieve estacional presente."
)
col3.metric(
    "NDSI Promedio", f"{ndsi_selected:.3f}", f"{images_selected} imágenes",
    help="Índice de Diferencia Normalizada de Nieve. Basado en imágenes Sentinel-2."
)
col4.metric(
    "Modelo (Secos)", f"{model_rate:.2f}%/año", f"R² = {model_r2:.2f}",
    help="Tasa anual de retroceso según modelo exponencial ajustado a años secos."
)

with st.expander("**Metodología del análisis geoespacial**", icon=":material/info:"):
    st.markdown(f"""
Este estudio cuantifica la dinámica reciente del área de hielo y nieve mediante imágenes satelitales **Sentinel-2**, combinando una **pre-máscara espectral basada en NDSI** con **desmezclado espectral lineal totalmente restringido** (*Fully Constrained Linear Spectral Unmixing, FCLSU*) para estimar la cobertura de nieve a nivel subpíxel. El script usa el valor refinado con el FCLSU como métrica principal del estudio y compara ese resultado con el método NDSI tradicional. 

#### **Enfoque inicial: estimación de superficie nival mediante NDSI**

Inicialmente se utilizó el método convencional NDSI (*Normalized Difference Snow Index*). Sin embargo, debido a la naturaleza del índice, al reducido tamaño del glaciar y a la resolución espacial de la banda SWIR (20 m), este enfoque tiende a **sobrestimar la superficie glaciar**, principalmente por la inclusión de nieve estacional y por la presencia de píxeles mixtos en los bordes.

#### **Desmezclado espectral (Spectral Unmixing)**

Para corregir este sesgo, se implementó el **desmezclado espectral**, que descompone cada píxel en fracciones de nieve, roca y vegetación, permitiendo una estimación más realista en zonas del borde irregular del glaciar donde predominan los píxeles mixtos (Painter et al., 2009; Sirguey et al., 2009)

El modelo fue calibrado a partir de puntos de referencia identificados en el área de estudio mediante fotointerpretación en QGIS y validados mediante análisis de separabilidad espectral. Posteriormente, para aumentar su robustez, se aplicó una **pre-máscara NDSI (> 0.2)** con el fin de ejecutar el unmixing únicamente en las zonas con alta probabilidad de presencia de nieve o hielo.

A partir de este procedimiento se derivaron dos métricas:

- **Sub-pixel unmixing:** suma de las fracciones estimadas de cada cobertura multiplicadas por el área del píxel. Esta es la **métrica principal** del estudio.
- **Binary unmixing:** conteo de píxeles con fracción de nieve ≥ 0.5, multiplicados por el área total del píxel.


#### **Mejoría de la estimación mediante FCLSU**

La utilidad del desmezclado espectral se evidencia al comparar sus resultados de 2020 con la estimación original basada en NDSI y con la referencia independiente de Ramírez et al. (2020) para el año 2019. Salvando las diferencias metodológicas y temporales, la comparación sigue siendo valiosa porque el glaciar se encuentra en una fase de retroceso acelerado, por lo que un incremento de superficie entre 2019 y 2020 resultaría físicamente poco probable. 
                
Bajo ese criterio, el valor estimado por FCLSU se aproxima más al orden de magnitud esperado y **reduce en 36.1% la sobreestimación inicialmente observada con el NDSI**.

#### **Análisis multitemporal**

Para la serie multitemporal se analizaron datos entre **2019 y 2026** (8 años) durante la estación seca (diciembre-marzo), cuando la cobertura de hielo es más representativa. Los años 2019 y 2026 fueron excluidos del modelo por presentar nieve estacional elevada, lo cual fue **confirmado por guías de montaña** [UGAM](https://www.instagram.com/ugamvenezuela/). De hecho, un análisis exploratorio de la distribución temporal, clasificó cuantitativamente a estos como años **“atípicos”**, usando un umbral de **0.060 km²**, estimado por IQR.
                
Se buscó además incorporar observaciones desde 2015, pero esos años fueron omitidos por ausencia de imágenes Sentinel-2 utilizables por la alta nubosidad en el área. Se trabaja actualmente en técnicas para obtener información sobre estos años y mejorar el modelo predictivo para mejor monitoreo automatizado a futuro.
                
**Procedimiento detallado en Python disponible en el repositorio de** [GitHub](https://github.com/leomed512/humboldt-glacier)
                
""")

st.markdown("---")


# ============================================================================
# PROJECTION GRAPH
# ============================================================================

st.markdown("""
<h2><span class="material-icons">trending_down</span> Proyección</h2>
""", unsafe_allow_html=True)

# Projection plot image
projection_img_path = 'results/plots/03_proyection.png'
if Path(projection_img_path).exists():
    st.image(projection_img_path, width="stretch")

area_2020_val = df.loc[df['year'] == 2020, 'snow_area_km2'].values[0] if 2020 in df['year'].values else None
if area_2020_val:
    diff_ramirez = ((area_2020_val - 0.046) / 0.046) * 100
    st.markdown(

        f"**Puntos verdes**: Ramírez et al. (2020). "
        f"**Variación entre estudios**: +{diff_ramirez:.1f}% (2020 vs Ramírez 2019)."
        f"** Nota:** Años húmedos ({', '.join(map(str, df_wet['year'].tolist()))}) "
        f"presentan nieve estacional inusual y fueron excluidos del modelo predictivo. "
    )

# Reclassification
reclass_info = model_projections.get('reclassification', {})
reclass_year = reclass_info.get('year')
reclass_unc = reclass_info.get('uncertainty_years', 2)

if reclass_year:
    st.warning(
        f"**Proyección:** Con la tasa actual de {model_rate:.2f}%/año, "
        f"el glaciar dejará de calificar como tal en **~{int(reclass_year)}** "
        f"(±{reclass_unc} años), pasando a ser un **parche de hielo** "
        f"según la clasificación de Huss & Fischer (2016).",
        icon=":material/warning:"
    )
# Model info
uncertainty = model_info.get('uncertainty', {})
st.info(
    f"**Modelo:** Exponencial · Años secos ({min(model_years)}-{max(model_years)}, n={len(model_years)}) · "
    f"R² = {model_r2:.3f} · RMSE = ±{uncertainty.get('model_rmse_km2', 0.0):.4f} km²",
    icon=":material/info:"
)
# Disappearance
disappear_info = model_projections.get('disappearance', {})
disappear_year = disappear_info.get('year')
disappear_unc = disappear_info.get('uncertainty_years', 3)

if disappear_year:
    st.error(
        f"**Desaparición estimada:** El remanente de hielo sería indetectable "
        f"por Sentinel-2 aproximadamente en **~{int(disappear_year)}** "
        f"(±{disappear_unc} años) si continúa la tendencia.",
        icon=":material/error:"
    )



# ── Technical expanders ───────────────────────────────────────────────

with st.expander("Selección del modelo", icon=":material/compare:"):
    st.markdown(f"""
Para explicar el comportamiento actual y a futuro de la dinámica del cuerpo de hielo se evaluaron varios modelos predictivos: **exponencial, lineal y polinomial** sobre los años secos (n={len(model_years)}) ya que la nieve estacional abundante de 2019 y 2026 destruía la confiabilidad de los modelos.
La selección se realizó mediante **AICc** (Akaike Information Criterion corregido para muestras pequeñas, 
Burnham & Anderson, 2002).

El modelo **exponencial** fue seleccionado por su  ajuste estadístico comparable (R²=0.87) y su compatibilidad con la naturaleza de los glaciares en fase terminal, donde
la tasa de pérdida es proporcional al área actual (Huss & Fischer, 2016). El modelo lineal 
presentó un ajuste estadístico comparable (R² similar) pero no permite proyecciones de desaparición 
físicamente coherentes.

**Ecuación:** {model_info.get('equation', 'N/A')}
""")

# ── Analysis plots expander ───────────────────────────────────────────

with st.expander("Gráficos del análisis adicionales", icon=":material/bar_chart:"):
    
    # Serie temporal
    serie_path = 'results/plots/01_time_series.png'
    if Path(serie_path).exists():
        st.markdown("**Evolución temporal**")
        st.image(serie_path, width="stretch")
        st.caption(
            "Serie temporal completa con datos de Ramírez et al. (2020) como referencia. Se observa el retroceso sostenido de la superficie, incrementándose en lo últimos 2 años. \n\n"

            "El modelo exponencial se ajusta a los años secos para explicar su comportamiento y hacer estimaciones de su dinámica a largo plazo. "
            "Los años húmedos muestran el efecto de la nieve estacional y han sido excluídos del modelo."
        )
        st.markdown("---")
    
    # Variabilidad y anomalías
    var_path = 'results/plots/02_annual_variability.png'
    if Path(var_path).exists():
        st.markdown("**Variabilidad interanual y anomalías**")
        st.image(var_path, width="stretch")
        st.caption(
            "**Panel superior:** cambio absoluto año a año. La caída más abrupta ocurrió entre 2019-2020 "
            "(-73%), seguida de una relativa estabilización 2021-2023 y nueva caída en 2024-2025. El año 2026 vuelve a mostrar presencia de gran cantidad de nieve estacional. \n\n"

            "**Panel inferior:** desviación de la estimación de cada año respecto al modelo exponencial. "
            "Los años húmedos (barras azules) representan anomalías positivas por nieve estacional."
        )
        st.markdown("---")
    
    # Comparación de métodos
    comp_path = 'results/plots/04_method_comparision.png'
    if Path(comp_path).exists():
        st.markdown("**Comparación de métodos de estimación**")
        st.image(comp_path, width="stretch")
        st.caption(
            "**Panel superior**: presenta la evolución del área glaciar estimada mediante tres enfoques: NDSI, unmixing binario y FCLSU subpíxel. El método NDSI reporta consistentemente valores superiores, evidenciando una tendencia a sobreestimar la cobertura nival. En contraste, los enfoques basados en desmezclado espectral (especialmente FCLSU -utilizada como métrica principal-) generan estimaciones más conservadoras al modelar explícitamente la fracción de nieve dentro de cada píxel. \n\n"

             "Panel inferior: muestra cuánto corrige el FCLSU respecto al NDSI (%). Todos los valores son negativos, lo que indica que el unmixing reduce el área estimada al tratar los píxeles mixtos de forma fraccional en lugar de contarlos completos. \n\n" 
             "" 

             "La línea punteada marca el promedio (~-16.7%), en que el NDSI sobreestima el área glaciar. Esta corrección es mayor en los últimos años, cuando el glaciar es más pequeño y los bordes (píxeles mixtos) influyen más."
        )

st.markdown("---")

# ============================================================================
# 3D TOPOGRAPHY WITH SELECTOR
# ============================================================================

screen_width = streamlit_js_eval(js_expressions='window.innerWidth', key='SCREEN_WIDTH', want_output=True)
if screen_width is None:
    screen_width = 700
is_mobile = screen_width <= 700

st.markdown("""
<h2><span class="material-icons">satellite</span> Topografía 3D</h2>
""", unsafe_allow_html=True)

current_year_3d = year_selector('3d', 'Seleccionar año')

dem, bounds = load_dem()

if dem is not None and bounds is not None:
    rows, cols = dem.shape
    x = np.linspace(bounds.left, bounds.right, cols)
    y = np.linspace(bounds.bottom, bounds.top, rows)
    x_grid, y_grid = np.meshgrid(x, y)

    fig_3d = go.Figure()
    
    fig_3d.add_trace(go.Surface(
        x=x_grid, y=y_grid, z=dem, colorscale='Earth',
        showscale=not is_mobile,
        colorbar=dict(title='Elevación (m)', x=1.1, tickfont=dict(size=10)) if not is_mobile else None,
        hovertemplate='Elev: %{z:.0f} m<extra></extra>', name='DEM'
    ))
    
    polygon_gdf_3d = load_polygon(current_year_3d)
    if polygon_gdf_3d is not None:
        for geom in polygon_gdf_3d.geometry:
            if geom.geom_type == 'Polygon':
                coords = list(geom.exterior.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                elevations = []
                for lon, lat in zip(lons, lats):
                    i = np.argmin(np.abs(y - lat))
                    j = np.argmin(np.abs(x - lon))
                    elevations.append(dem[i, j] + 15)
                fig_3d.add_trace(go.Mesh3d(
                    x=lons, y=lats, z=elevations, color='cyan', opacity=0.6,
                    name=f'Glaciar {current_year_3d}', showscale=False,
                    hovertemplate=f'Glaciar {current_year_3d}<extra></extra>'
                ))

    fig_3d.update_layout(
        scene=dict(
            zaxis_title='Elevación (m)',
            camera=dict(
                eye=dict(x=-0.7, y=-0.7, z=0.7) if not is_mobile else dict(x=-0.9, y=-0.9, z=0.55),
                center=dict(x=0, y=0, z=-0.1)
            ),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.3)
        ),
        height=600, template='plotly_dark', showlegend=False
    )

    st.plotly_chart(fig_3d, width="stretch")
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
        st.image(comparative_img_path, caption="Vista previa", width="stretch")
else:
    st.warning("Imagen comparativa no disponible. Ejecuta Script 02.")

st.markdown("---")

# ============================================================================
# VALIDATION
# ============================================================================

st.markdown("""
<h2><span class="material-icons">verified</span> Validación Comparativa</h2>
""", unsafe_allow_html=True)

if area_2020_val:
    diff_pct = ((area_2020_val - 0.046) / 0.046) * 100
    st.markdown(f"""
Este análisis se comparó con **Ramírez et al. (2020)**,
*"The end of the eternal snows: Integrative mapping of 100 years of glacier retreat in the Venezuelan Andes"*,
publicado en *Arctic, Antarctic, and Alpine Research*.

La variación de +{diff_pct:.1f}% es consistente con las diferencias de resolución espacial 
(fotogrametría submétrica vs Sentinel-2 a 10-20 m) y año de referencia (2019 vs 2020*), ambos se encuentran en un rango coherente.

- **Ramírez et al. (2020):** Para 2019 reportaron **0.046 km²** (fotointerpretación manual )
- **Este estudio (2020):** **{area_2020_val:.4f} km²** (desmezclado espectral sub-píxel Sentinel-2)
- **Variación:** **+{diff_pct:.1f}%**

* Se eligió 2020 como punto de comparación debido a que 2019 presentó cobertura elevada de nieve 
estacional, **confirmado por reportes de campo de guías de montaña** 
[UGAM](https://www.instagram.com/ugamvenezuela/).

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
