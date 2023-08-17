import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import yaml
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap, hex2color
from contextlib import suppress
from bokeh.models import HoverTool
import geopandas as gpd
import networkx as nx
import hvplot.networkx as hvnx
import holoviews as hv
import datetime
import hvplot.pandas
import pypsa
import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from helpers import prepare_colors, rename_techs_tyndp, get_cmap

CACHE_TTL = 24*3600  # seconds
pd.options.plotting.backend = 'holoviews'


@st.cache_data(persist=True)
def load_regions():
    fn = "data/regions_onshore_elec_s_all.geojson"
    gdf = gpd.read_file(fn)
    gdf.geometry = gdf.to_crs(3035).geometry.simplify(
        1000).to_crs(ccrs.Geodetic())
    return gdf


@st.cache_data(persist=True)
def load_shape(fn):
    gdf = gpd.read_file(fn)
    gdf.geometry = gdf.to_crs(3035).geometry.simplify(1000).to_crs(4326)
    gdf.set_index("name", inplace=True)
    return gdf


@st.cache_data(persist=True)
def open_networks():
    networks = dict()
    for file in os.listdir("./data/networks"):
        if not file.endswith(".nc"):
            continue
        n = pypsa.Network("./data/networks/" + file)
        year = file.split("_")[0][2:]
        constraint = file.split("_")[3].split("-")[1]
        dlr = file.split("_")[4]
        v = file.split("_")[5]
        n.name = f"{year}-{constraint}-{dlr}-{v}"
        networks[f"{n.name}"] = n
    return networks


def network_selector(scenario_selection, dlr_selection):
    if scenario_selection == "year 2019":
        year = "2020"
        share = ""
    elif scenario_selection == "80% renewable share":
        year = "2030"
        share = "0.8"
    elif scenario_selection == "85% renewable share":
        year = "2030"
        share = "0.85"
    elif scenario_selection == "90% renewable share":
        year = "2030"
        share = "0.9"
    elif scenario_selection == "95% renewable share":
        year = "2030"
        share = "0.95"
    elif scenario_selection == "100% renewable share":
        year = "2030"
        share = "1.0"
    if dlr_selection == "unlimited":
        dlr = ""
    else:
        dlr = dlr_selection/100
    return f"{year}-\\w*{share}-dlr{dlr}-\\w*"


# MAIN
with open("data/config.yaml", encoding='utf-8') as file:
    config = yaml.safe_load(file)

colors = prepare_colors(config)

preferred_order = pd.Index(config['preferred_order'])


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
    dlr_rating_options = [100, 130, 150, 180, 200, "unlimited"]
    scenario_list = ["year 2019"] + \
        [f"{x}% renewable share" for x in range(80, 105, 5)]
    tech_list = ["Onshore Wind", "Solar",
                 "Offshore Wind (AC)", "Offshore Wind (DC)"]
    barplot_parameters = ["optimal capacities",
                          "generation", "curtailment", "cost"]


networks = pd.Series(open_networks())
regions_shapes = load_shape("data/regions_onshore_elec_s_all.geojson")
country_shape = load_shape("data/country_shapes.geojson")

if (display == "Map explorer"):

    st.title("Map explorer")

    _, col1, col2, _ = st.columns([1, 50, 50, 1])

    with col1:
        custom_settings["dlr_rating"] = st.select_slider(
            "DLR Rating", dlr_rating_options, help="Choose dynamic line rating factor.")

    with col2:
        custom_settings["scenario"] = st.selectbox(
            "Scenario", scenario_list, help="Choose scenario.")

    custom_settings["tech"] = st.selectbox(
        "Technology", tech_list, help="Choose technology.")

    n = networks.filter(regex=network_selector(
        custom_settings["scenario"], custom_settings["dlr_rating"]))

    if n.empty:
        st.error("No data available for this selection.")
        st.stop()
    else:
        n = n.iloc[0]

        opts = dict(
            xaxis=None,
            yaxis=None,
            active_tools=['pan']
        )
        crs = "4326"

        carrier = n.carriers[n.carriers.nice_name ==
                             custom_settings["tech"]].squeeze()
        capacities = n.statistics.optimal_capacity(comps=["Generator"], groupby=[
                                                   "carrier", "bus"]).droplevel(0)
        cmap = get_cmap(carrier.name)

        regions_shapes[carrier.nice_name] = capacities[carrier.name].div(1e3)
        regions_shapes[carrier.nice_name] = regions_shapes[carrier.nice_name].fillna(
            0)
        plot = regions_shapes.hvplot(
            height=720,
            width=600,
            color=carrier.nice_name,
            alpha=0.7,
            crs=crs,
            hover_cols=[carrier.nice_name],
            cmap=cmap,
            tiles=config["tiles"],
        ).opts(**opts)

        # plot lines
        lines = n.lines
        edges = lines.loc[:, ["bus0", "bus1"]]
        attr = ["Total Capacity (GW)", "Length (km)"]
        edges.loc[:, "Total Capacity (GW)"] = lines.s_nom_opt.clip(
            lower=1e-3)/1e3
        edges.loc[:, "Length (km)"] = lines.length
        pos = n.buses[["x", "y"]].apply(tuple, axis=1).to_dict()
        G = nx.from_pandas_edgelist(edges, 'bus0', 'bus1', edge_attr=attr)
        line_plot = hvnx.draw(
            G,
            pos=pos,
            node_size=0,
            edge_color='Total Capacity (GW)',
            inspection_policy="edges",
            crs=crs,
        )

        plot *= line_plot

        st.bokeh_chart(hv.render(plot, backend='bokeh'),
                       use_container_width=True)

        rating_plot = (n.lines_t.p0/n.lines.s_nom_opt).abs().mean(
            axis=1).rename("Rating").hvplot(title="Average Line Rating").opts(active_tools=['pan'])
        st.bokeh_chart(hv.render(rating_plot, backend='bokeh'),
                       use_container_width=True)

        generation_plot = n.statistics.dispatch(comps=["Generator"], aggregate_time=False).droplevel(
            0).loc[carrier.nice_name].div(1e3).hvplot(title=f"{carrier.nice_name} Generation in GW").opts(active_tools=['pan'])
        st.bokeh_chart(hv.render(generation_plot, backend='bokeh'),
                       use_container_width=True)


if (display == "Bar plot explorer"):

    st.title("Bar plot explorer")

    # custom_settings["barplot_parameters"] = st.selectbox(
    #     "Parameter", barplot_parameters, help="Choose a parameter you want to plot as bar plot.")

    _, col1, col2, _ = st.columns([1, 50, 50, 1])

    with col1:
        custom_settings["dlr_rating"] = st.select_slider(
            "DLR Rating", dlr_rating_options, help="Choose dynamic line rating factor.")

    with col2:
        custom_settings["scenario"] = st.selectbox(
            "Scenario", scenario_list, help="Choose scenario.")

    n = networks.filter(regex=network_selector(
        custom_settings["scenario"], custom_settings["dlr_rating"]))
    if n.empty:
        st.error("No data available for this selection.")
        st.stop()
    n = n.iloc[0]

    plots = [(n.statistics.optimal_capacity(), 'Optimal Capacity'),
             (n.statistics.dispatch(), 'Dispatch'),
             (n.statistics.curtailment(), 'Curtailment'),
             (pd.concat([n.statistics.capex(), n.statistics.opex()], axis=1, keys=[
                 "capex", "opex"]), 'Capex and Opex')]

    # Loop over the plots and create and plot each one
    for data, title in plots:
        plot = data.hvplot.bar(title=title, stacked=True,
                               height=400, width=600).opts(active_tools=['pan'])

        st.bokeh_chart(hv.render(plot, backend='bokeh'),
                       use_container_width=True)

if (display == "Network comparison"):

    st.title("Bar plot explorer")

    n = networks.iloc[0]
    plot = n.statistics.dispatch().hvplot.bar()
    st.bokeh_chart(hv.render(plot, backend='bokeh'), use_container_width=True)
