# %%
import pandas as pd
import pypsa
import os
# %%
path_to_files = "../results/"
for file in os.listdir(path_to_files):
    if not file.endswith(".nc"):
        continue
    n = pypsa.Network(path_to_files + file)
    # %%
    if not isinstance(n.snapshots, pd.DatetimeIndex):
        n.set_snapshots(pd.date_range(
            f'{file[2:6]}', periods=len(n.snapshots), freq='H'))

    # %%
    resample = "1W"
    resample_method = "mean"
    keep_components_list = ["Line", "Link",
                            "StorageUnit", "Generator", "Bus"]
    static_attributes = ["x", "y", "country", "carrier", "bus", "p_nom_opt", "p_nom", "p_nom_max",
                         "efficiency", "marginal_cost", "capital_cost", "bus0", "bus1", "s_nom_opt", "length"]
    dynamic_attributes = ["s_max_pu", "s_min_pu",
                          "p", "p0", "p_max_pu", "marginal_price"]

    # %%
    # wanted to delete all components which are not needed but cannot export it to netcdf afterwards
    # network_components = list(n.components.keys())
    # for component in network_components:
    for component in keep_components_list:
        # if component not in keep_components_list:
        #     del n.components[component]
        #     continue
        attribute_list = n.df(component).columns
        for attribute in attribute_list:
            if attribute not in static_attributes:
                n.df(component).drop(columns=attribute, inplace=True)
        attribute_list = list(n.pnl(component).keys())
        for attribute in attribute_list:
            if attribute in dynamic_attributes:
                n.pnl(component)[attribute] = n.pnl(component)[attribute].resample(
                    resample).mean()

    n.set_snapshots(pd.date_range(
        n.snapshots.min(), n.snapshots.max(), freq=resample))

    for component in keep_components_list:
        if attribute not in dynamic_attributes:
            del n.pnl(component)[attribute]
    # %%
    ds = n.export_to_netcdf(f"./data/{file}", float32=True, compression={
        "zlib": True, "complevel": 9, "least_significant_digit": 5})
    # %%
    memory_usage = pd.Series(
        {var: ds[var].data.nbytes/1e6 for var in ds.data_vars})
    memory_usage.sort_values(inplace=True, ascending=False)

    # %%
