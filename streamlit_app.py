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
import os
import cartopy.crs as ccrs

from helpers import prepare_colors, rename_techs_tyndp, get_cmap

CACHE_TTL = 24*3600  # seconds


@st.cache_data(persist=True)
def load_regions():
    fn = "data/regions_onshore_elec_s_all.geojson"
    gdf = gpd.read_file(fn)
    gdf.geometry = gdf.to_crs(3035).geometry.simplify(
        1000).to_crs(ccrs.Geodetic())
    return gdf


@st.cache_data(persist=True)
def load_country_shape():
    fn = "data/country_shapes.geojson"
    gdf = gpd.read_file(fn)
    gdf.geometry = gdf.to_crs(3035).geometry.simplify(1000).to_crs(4326)
    return gdf


@st.cache_data(persist=True)
def open_networks():
    networks = dict()
    for file in os.listdir("./data/"):
        if not file.endswith(".nc"):
            continue
        n = pypsa.Network("./data/" + file)
        year = file.split("_")[0][2:]
        constraint = file.split("_")[3].split("-")[1]
        dlr = file.split("_")[4]
        v = file.split("_")[5]
        n.name = f"{year}-{constraint}-{dlr}-{v}"
        networks[f"{n.name}"] = n
    return networks


def scale_bus_sizes(bus_sizes, bus_size_factor):
    if bus_size_factor is None:
        max_size_factor = 0.16
        bus_size_factor = max_size_factor / \
            bus_sizes.groupby("bus").sum().max()
    return bus_size_factor


def scale_line_widths(line_widths, line_width_factor):
    if line_width_factor is None:
        max_line_factor = 5
        line_width_factor = max_line_factor / line_widths.max()
    return line_width_factor


def plot_capacity_map(ax, n, bounds, bus_size_factor=None, line_width_factor=None, with_colormap=False
                      ):
    # Plots capacity of all generators in MW, existing and newly built
    g = n.generators.p_nom_opt
    bus_sizes = g.groupby([n.generators.bus, n.generators.carrier]).sum()
    bus_sizes.drop("load", errors="ignore", level=1, inplace=True)
    bus_size_factor = scale_bus_sizes(bus_sizes, bus_size_factor)
    link_widths = n.links.p_nom_opt
    line_widths = n.lines.s_nom_opt
    line_width_factor = scale_line_widths(line_widths, line_width_factor)
    # color map lines
    cmap = cm.get_cmap("viridis", 256)
    line_colors = n.get_switchable_as_dense("Line", "s_max_pu").mean() / 0.7
    norm = colors.Normalize(vmin=line_colors.min(), vmax=line_colors.max())

    with plt.rc_context({"patch.linewidth": 0}):
        collection = n.plot(
            ax=ax,
            line_widths=line_widths * line_width_factor,
            link_widths=link_widths * line_width_factor,
            line_colors=line_colors,
            line_cmap=cmap,
            bus_sizes=bus_sizes * bus_size_factor,
            bus_alpha=0.7,
            color_geomap=False,
            boundaries=bounds,
        )

    if with_colormap:
        collection[1].set(norm=norm)
        plt.colorbar(collection[1], ax=ax, fraction=0.04,
                     pad=0.004, label="DLR / SLR")


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

    dlr_rating_options = [100, 120, 140, 150]
    custom_settings["dlr_rating"] = st.select_slider(
        "DLR Rating", dlr_rating_options, help="Choose dynamic line rating factor.")

    scenario_list = ["year 2019",
                     "80% renewable share", "100 % renewable share"]
    custom_settings["scenario"] = st.selectbox(
        "Scenario", scenario_list, help="Choose scenario.")

networks = pd.Series(open_networks())

if (display == "Map explorer"):

    st.title("Map explorer")

    gdf = load_country_shape().drop("name", axis=1)

    opts = dict(
        active_tools=['pan', 'wheel_zoom']
    )
    crs = ccrs.Geodetic()

    # plot german shapefile
    plot = gdf.hvplot(
        # geo=True,
        height=720,
        tiles=config["tiles"],
        color='white',
        alpha=0.7,
    ).opts(**opts)

    # plot lines
    n = networks.iloc[0]
    lines = n.lines
    edges = lines.loc[:, ["bus0", "bus1"]]
    attr = ["Total Capacity (GW)", "Length (km)"]
    edges.loc[:, "Total Capacity (GW)"] = lines.s_nom_opt.clip(lower=1e-3)/1e3
    edges.loc[:, "Length (km)"] = lines.length
    pos = n.buses[["x", "y"]].apply(tuple, axis=1).to_dict()
    G = nx.from_pandas_edgelist(edges, 'bus0', 'bus1', edge_attr=attr)
    # line_plot = hvnx.draw(
    #     G,
    #     pos=pos,
    #     # node_size=4,
    #     # node_color='grey',
    #     edge_color='Total Capacity (GW)',
    #     inspection_policy="edges",
    #     # edge_width=4,
    #     crs=ccrs.Geodetic(),
    # ).opts(**opts)

    # plot *= line_plot

    # plot buses
    buses = gpd.GeoDataFrame(index=n.buses.index, geometry=gpd.points_from_xy(
        n.buses.x, n.buses.y), crs=crs)
    bus_size = n.statistics.optimal_capacity(groupby=n.statistics.get_bus_and_carrier)[
        "Generator"]
    # buses = buses.merge(bus_size.unstack(), left_index=True, right_index=True)
    node_plot = buses.hvplot(
        # geo=True,
        alpha=0.7,
        # hover_cols=list(bus_size.index.get_level_values(1).unique()),
    )
    plot *= node_plot

    # wanted to show an additional plot when hovering over a bus showing the capacities of the bus
    # def capacity_plot(index):
    #     plot = bus_size[index].hvplot.bar()
    #     return plot
    # capacity_dmap = hv.DynamicMap(capacity_plot, streams=[
    #                               hv.streams.Selection1D(source=node_plot)])

    # plot *= capacity_dmap

    st.bokeh_chart(hv.render(plot, backend='bokeh'),
                   use_container_width=True)

if (display == "Bar plot explorer"):

    st.title("Bar plot explorer")

    n = networks.iloc[0]
    plot = n.statistics.dispatch().hvplot.bar()
    st.bokeh_chart(hv.render(plot, backend='bokeh'), use_container_width=True)
