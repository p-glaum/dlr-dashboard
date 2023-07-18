import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import yaml
import plotly.graph_objects as go
from matplotlib.colors import to_rgba
from contextlib import suppress
from bokeh.models import HoverTool
import geopandas as gpd
import networkx as nx
import hvplot.networkx as hvnx
import holoviews as hv
import datetime
import hvplot.pandas
import pypsa

from helpers import prepare_colors, rename_techs_tyndp, get_cmap

CACHE_TTL = 24*3600  # seconds


@st.cache_data(persist=True, ttl=CACHE_TTL)
def load_regions():
    fn = "data/regions_onshore_elec_s_all.geojson"
    gdf = gpd.read_file(fn)
    gdf.geometry = gdf.to_crs(3035).geometry.simplify(1000).to_crs(4326)
    return gdf

# MAIN


with open("data/config.yaml", encoding='utf-8') as file:
    config = yaml.safe_load(file)

colors = prepare_colors(config)

preferred_order = pd.Index(config['preferred_order'])

n = pypsa.Network("../results/de2020_all_nodes_Co2L-BL-Ep_dlr1.0_v.nc")

# DISPLAY

st.set_page_config(
    page_title='Dynamic Line Rating',
    layout="wide"
)

style = '<style>div.block-container{padding-top:.5rem; padding-bottom:0rem; padding-right:1.2rem; padding-left:1.2rem}</style>'
st.write(style, unsafe_allow_html=True)

# SIDEBAR

with st.sidebar:
    st.title(
        "[Leveraging the existing German transmission grid with dynamic line rating](https://www.sciencedirect.com/science/article/abs/pii/S0306261923005639)")

    st.markdown("""
        **Philipp Glaum, Fabian Hofmann**
    """)

    view_list = [
        "Map explorer",
        "Bar plot explorer",
        # "Scenario explorer",
    ]
    display = st.selectbox(
        "View", view_list, help="Choose your view on the system.")

    custom_settings = dict()

    dlr_rating_options = [100, 120, 140, 150]
    custom_settings["dlr_rating"] = st.select_slider(
        "DLR Rating", dlr_rating_options, help="Choose dynamic line rating factor.")

    scenario_list = ["year 2019",
                     "80% renewable share", "100 % renewable share"]
    custom_settings["scenario"] = st.selectbox(
        "Scenario", scenario_list, help="Choose scenario.")


if (display == "Map explorer"):

    st.title("Map explorer")

    gdf = load_regions().drop("name", axis=1)

    opts = dict(
        xaxis=None,
        yaxis=None,
        active_tools=['pan', 'wheel_zoom']
    )

    kwargs = dict(
        color='white',
        line_color='grey',
        alpha=0.7,
    )

    plot = gdf.plot()
