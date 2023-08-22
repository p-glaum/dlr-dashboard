import pandas as pd


def rename_techs(label):

    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral "
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch"
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        "battery": "battery storage",
        "SMR": "steam methane reforming",
        "SMR CC": "steam methane reforming CC",
    }

    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "offwind-ac": "offshore wind (AC)",
        "offwind-dc": "offshore wind (DC)",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "NH3": "ammonia",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines"
    }

    for ptr in prefix_to_remove:
        if label[:len(ptr)] == ptr:
            label = label[len(ptr):]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old, new in rename.items():
        if old == label:
            label = new
    return label


def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    # if "heat pump" in tech or "resistive heater" in tech:
    #    return "power-to-heat"
    # elif tech in ["H2 Electrolysis", "methanation", "helmeth", "H2 liquefaction"]:
    #    return "power-to-gas"
    if tech == "H2":
        return "H2 storage"
    # elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
    #    return "gas-to-power/heat"
    # elif "solar" in tech:
    #    return "solar"
    # elif tech == "Fischer-Tropsch":
    #    return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    #    if "heat pump" in tech:
    #        return "heat pump"
    elif tech == "gas":
        return "fossil gas"
    # elif "CC" in tech or "sequestration" in tech:
    #    return "CCS"
    elif tech in ["industry electricity", "agriculture electricity"]:
        return "industry electricity"
    elif "oil emissions" in tech:
        return "oil emissions"
    elif "agriculture" in tech:
        return "agriculture"
    elif "H2 for" in tech or tech == "land transport fuel cell":
        return "H2"
    elif "EV" in tech or tech in ['Li ion', "V2G"]:
        return "land transport EV"
    elif tech == "electricity" or tech == "industry electricity":
        return "electricity"
    elif "co2" in tech:
        return "CCS"
    else:
        return tech


def prepare_colors(config):
    colors = pd.Series(config["tech_colors"])
    return colors.rename(config["nice_names"])


def get_cmap(c):
    if "solar" in c:
        return "Oranges"
    elif "hydrogen" in c:
        return "Purples"
    elif "bio" in c or "battery" in c:
        return "Greens"
    else:
        return 'Blues'
