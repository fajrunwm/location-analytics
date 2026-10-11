"""Stage 3 — aggregate node/segment metrics to H3 hexagons and to outlets.

Outputs (in <out>/<area>/):
  hex_features.parquet      one row per H3 cell (default res 9, ~0.1 km2): network features,
                            population, transit, and POI-count targets
  outlet_features.parquet   one row per brand outlet inside the area: features of the
                            nearest walk/drive node + its hex, joined to outlets.csv

Column prefixes:
  walk_* / drive_* / ang_*   street-network structure (allowed as model predictors)
  acc_*                      walk-distance access to POIs/transit — derived from the same POIs
                             as the targets, so NOT used to predict POI density (leakage);
                             used for outlet scoring only
  ctx_*                      context (population, transit stops)
  y_*                        targets: Overture POI counts (confidence >= 0.5)
"""
import argparse
import time
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

from areas import AREAS, utm_epsg

COMMERCIAL = ["food_and_drink", "shopping", "services_and_business", "lodging", "arts_and_entertainment"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_h3(df, res):
    df = df.copy()
    df["h3"] = [h3.latlng_to_cell(la, lo, res) for la, lo in zip(df.lat.values, df.lon.values)]
    return df


def split_cols(cols):
    """Separate structural centrality columns from POI/pop accessibility columns."""
    struct, access, pop = [], [], []
    for c in cols:
        if not c.startswith(("cc_", "nk_")):
            continue
        if c.startswith("cc_pop_"):
            pop.append(c)
        elif c.startswith("nk_") or any(c.startswith(f"cc_{k}") for k in
                                        ("density", "harmonic", "beta", "betweenness", "cycles", "farness", "hillier")):
            struct.append(c)
        else:
            access.append(c)
    return struct, access, pop


def rename(c, prefix):
    return prefix + c.replace("cc_", "", 1)


def agg_nodes(df, cols, prefix):
    g = df.groupby("h3")[cols]
    mean = g.mean().rename(columns=lambda c: rename(c, prefix) + "_mean")
    mx = g.max().rename(columns=lambda c: rename(c, prefix) + "_max")
    return mean.join(mx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", required=True, choices=AREAS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outlets", required=True, help="path to outlets.csv")
    ap.add_argument("--res", type=int, default=9)
    a = ap.parse_args()

    base = Path(a.out) / a.area
    raw = base / "raw"
    area = AREAS[a.area]
    w, s, e, n = area["bbox"]

    walk = add_h3(pd.read_parquet(base / "nodes_walk.parquet"), a.res)
    drive = add_h3(pd.read_parquet(base / "nodes_drive.parquet"), a.res)
    seg = add_h3(pd.read_parquet(base / "segments_walk_angular.parquet"), a.res)

    w_struct, w_access, w_pop = split_cols(walk.columns)
    d_struct, _, _ = split_cols(drive.columns)
    ang = [c for c in seg.columns if c.startswith("cc_")]

    parts = [
        walk.groupby("h3").size().rename("walk_node_count").to_frame(),
        agg_nodes(walk, w_struct, "walk_"),
        agg_nodes(drive, d_struct, "drive_"),
        agg_nodes(seg, ang, "ang_"),
        walk.groupby("h3")[w_pop].mean().rename(columns=lambda c: rename(c, "ctx_")),
        walk.groupby("h3")[w_access].mean().rename(columns=lambda c: rename(c, "acc_")),
    ]
    hexes = parts[0].join(parts[1:], how="left")

    pop = add_h3(pd.read_parquet(raw / "population.parquet"), a.res)
    hexes = hexes.join(pop.groupby("h3")["pop"].sum().rename("ctx_pop_hex"), how="left")

    transit = gpd.read_parquet(raw / "transit.parquet")
    transit = add_h3(transit.assign(lon=transit.geometry.x, lat=transit.geometry.y), a.res)
    hexes = hexes.join(pd.crosstab(transit.h3, transit.transit_type).add_prefix("ctx_n_"), how="left")

    places = gpd.read_parquet(raw / "places.parquet")
    places = places[places.confidence >= 0.5]
    places = add_h3(places.assign(lon=places.geometry.x, lat=places.geometry.y), a.res)
    counts = pd.crosstab(places.h3, places.category_top)
    y = pd.DataFrame(index=counts.index)
    y["y_fnb"] = counts.get("food_and_drink", 0)
    y["y_shopping"] = counts.get("shopping", 0)
    y["y_commercial"] = counts[[c for c in COMMERCIAL if c in counts]].sum(axis=1)
    y["y_all"] = counts.sum(axis=1)
    hexes = hexes.join(y, how="left")

    fill0 = [c for c in hexes.columns if c.startswith(("y_", "ctx_n_", "ctx_pop_hex"))]
    hexes[fill0] = hexes[fill0].fillna(0)
    ll = np.array([h3.cell_to_latlng(h) for h in hexes.index])
    hexes["lat"], hexes["lon"] = ll[:, 0], ll[:, 1]
    hexes = hexes[(hexes.lon.between(w, e)) & (hexes.lat.between(s, n))]
    hexes["h3_parent6"] = [h3.cell_to_parent(h, 6) for h in hexes.index]
    hexes["h3_parent7"] = [h3.cell_to_parent(h, 7) for h in hexes.index]
    hexes["area"] = a.area
    hexes.index.name = "h3"
    hexes.to_parquet(base / "hex_features.parquet")
    log(f"hex_features: {hexes.shape}, targets: {hexes[['y_fnb', 'y_commercial']].sum().to_dict()}")

    # Outlets inside this area: nearest walk/drive node features + hex features.
    outlets = pd.read_csv(a.outlets)
    outlets = outlets[outlets.longitude.between(w, e) & outlets.latitude.between(s, n)].copy()
    if outlets.empty:
        log("no outlets in area")
        return
    epsg = utm_epsg(area["bbox"])
    og = gpd.GeoDataFrame(outlets, geometry=gpd.points_from_xy(outlets.longitude, outlets.latitude), crs=4326).to_crs(epsg)

    def nearest(nodes_path, cols, prefix):
        nodes = gpd.read_parquet(nodes_path)
        j = gpd.sjoin_nearest(og[["outlet_id", "geometry"]], nodes[cols + ["geometry"]], how="left",
                              max_distance=300, distance_col=f"{prefix}snap_m")
        j = j.drop_duplicates("outlet_id").set_index("outlet_id").drop(columns=["geometry", "index_right"], errors="ignore")
        return j.rename(columns=lambda c: rename(c, prefix) if c.startswith(("cc_", "nk_")) else c)

    nw = nearest(base / "nodes_walk.parquet", w_struct + w_access + w_pop, "node_walk_")
    nd = nearest(base / "nodes_drive.parquet", d_struct, "node_drive_")
    outlets["h3"] = [h3.latlng_to_cell(la, lo, a.res) for la, lo in zip(outlets.latitude, outlets.longitude)]
    of = outlets.set_index("outlet_id").join(nw).join(nd)
    of = of.join(hexes.drop(columns=["lat", "lon", "area"]).add_prefix("hex_"), on="h3")
    of["area"] = a.area
    of.to_parquet(base / "outlet_features.parquet")
    log(f"outlet_features: {of.shape} ({len(of)} outlets)")


if __name__ == "__main__":
    main()
