"""Stage 2 — street-network centrality + network-distance accessibility per node.

For one area and one network (walk | drive), writes <out>/<area>/nodes_<net>.parquet
with one row per street intersection:

  cc_*_{d}            cityseer localized centrality within network distance d metres
                      (density/harmonic closeness/betweenness; walk also angular "simplest-path")
  cc_pop_sum_{d}      WorldPop population reachable within d (network distance, walk only)
  cc_<cat>_{d}_wt     distance-weighted access to Overture POI categories / transit (walk only)
  nk_betweenness      global betweenness (networkit; exact or sampled for big graphs)
  nk_closeness        global harmonic closeness (networkit)

Localized centrality is the standard space-syntax / urban-network-analysis way to
separate "local street life" (400–800 m) from "city-wide through-movement" (5–10 km).
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("CITYSEER_QUIET_MODE", "1")

import geopandas as gpd
import networkit as nk
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, io

from areas import AREAS, DRIVE_DISTANCES, WALK_DISTANCES, utm_epsg

POI_KEYS = ["food_and_drink", "shopping", "services_and_business", "education", "lodging",
            "health_care", "arts_and_entertainment", "community_and_government", "travel_and_transportation"]
TRANSIT_KEYS = ["rail_station", "bus_station", "bus_stop"]
EXACT_BETWEENNESS_MAX_NODES = int(os.environ.get("EXACT_BETWEENNESS_MAX_NODES", 250_000))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def global_networkit(G_proj):
    """Global betweenness & harmonic closeness on the length-weighted undirected graph."""
    U = nx.Graph()
    for u, v, d in G_proj.edges(data=True):
        w = float(d.get("length", 1.0))
        if U.has_edge(u, v):
            w = min(w, U[u][v]["weight"])
        U.add_edge(u, v, weight=w)
    idx = {n: i for i, n in enumerate(U.nodes())}
    g = nk.Graph(len(idx), weighted=True, directed=False)
    for u, v, d in U.edges(data=True):
        g.addEdge(idx[u], idx[v], d["weight"])
    n = g.numberOfNodes()
    nk.setNumberOfThreads(os.cpu_count())

    t = time.time()
    if n <= EXACT_BETWEENNESS_MAX_NODES:
        bc = nk.centrality.Betweenness(g, normalized=True)
        mode = "exact"
    else:
        samples = int(os.environ.get("BETWEENNESS_SAMPLES", 20_000))
        bc = nk.centrality.EstimateBetweenness(g, samples, normalized=True, parallel=True)
        mode = f"sampled({samples})"
    bc.run()
    log(f"networkit betweenness {mode} on {n:,} nodes: {time.time() - t:.0f}s")

    t = time.time()
    hc = nk.centrality.HarmonicCloseness(g, normalized=True)
    hc.run()
    log(f"networkit harmonic closeness: {time.time() - t:.0f}s")

    keys = list(idx)
    return pd.DataFrame({"node": [str(k) for k in keys], "nk_betweenness": bc.scores(),
                         "nk_closeness": hc.scores()}).set_index("node"), mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", required=True, choices=AREAS)
    ap.add_argument("--net", required=True, choices=["walk", "drive"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-global", action="store_true")
    ap.add_argument("--global-exact-only", action="store_true",
                    help="only exact global betweenness/closeness -> nodes_<net>_global_exact.parquet")
    a = ap.parse_args()

    base = Path(a.out) / a.area
    raw = base / "raw"
    epsg = utm_epsg(AREAS[a.area]["bbox"])
    distances = WALK_DISTANCES if a.net == "walk" else DRIVE_DISTANCES

    G = ox.load_graphml(raw / f"{a.net}.graphml")
    G_proj = ox.project_graph(G, to_crs=epsg)
    log(f"{a.area}/{a.net}: {len(G_proj):,} nodes, {G_proj.number_of_edges():,} edges, EPSG:{epsg}, cpus={os.cpu_count()}")

    if a.global_exact_only:
        global EXACT_BETWEENNESS_MAX_NODES
        EXACT_BETWEENNESS_MAX_NODES = 10**9
        glob, _ = global_networkit(G_proj)
        glob.rename(columns=lambda c: c + "_exact").to_parquet(base / f"nodes_{a.net}_global_exact.parquet")
        log(f"wrote nodes_{a.net}_global_exact.parquet: {glob.shape}")
        return

    G_cs = io.nx_from_osm_nx(G_proj)
    G_cs = graphs.nx_remove_filler_nodes(G_cs)
    nodes_gdf, _, ns = io.network_structure_from_nx(G_cs)
    log(f"cityseer graph: {len(nodes_gdf):,} nodes")

    t = time.time()
    nodes_gdf = networks.node_centrality_shortest(ns, nodes_gdf, distances=distances)
    log(f"shortest-path centrality {distances}: {time.time() - t:.0f}s")

    if a.net == "walk":
        # Angular analysis runs on the dual graph (street segments as nodes) -> separate table.
        t = time.time()
        dual_gdf, _, ns_dual = io.network_structure_from_nx(graphs.nx_to_dual(G_cs))
        dual_gdf = networks.node_centrality_simplest(ns_dual, dual_gdf, distances=distances)
        mid = dual_gdf.geometry.interpolate(0.5, normalized=True)
        seg = gpd.GeoDataFrame(dual_gdf[[c for c in dual_gdf.columns if c.startswith("cc_")]],
                               geometry=mid.values, crs=epsg)
        ll = seg.geometry.to_crs(4326)
        seg["lon"], seg["lat"] = ll.x, ll.y
        seg.reset_index(drop=True).to_parquet(base / "segments_walk_angular.parquet")
        log(f"angular (simplest-path) centrality on {len(seg):,} segments: {time.time() - t:.0f}s")

        pop = pd.read_parquet(raw / "population.parquet")
        pop_gdf = gpd.GeoDataFrame(pop[["pop"]], geometry=gpd.points_from_xy(pop.lon, pop.lat), crs=4326).to_crs(epsg)
        t = time.time()
        nodes_gdf = layers.compute_stats(pop_gdf, ["pop"], nodes_gdf, ns, distances=distances, max_netw_assign_dist=150)
        log(f"population reach: {time.time() - t:.0f}s")

        places = gpd.read_parquet(raw / "places.parquet")
        places = places[places.confidence >= 0.5].to_crs(epsg)
        t = time.time()
        nodes_gdf = layers.compute_accessibilities(places, "category_top", POI_KEYS, nodes_gdf, ns,
                                                   distances=distances, max_netw_assign_dist=150)
        transit = gpd.read_parquet(raw / "transit.parquet").to_crs(epsg)
        nodes_gdf = layers.compute_accessibilities(transit, "transit_type", TRANSIT_KEYS, nodes_gdf, ns,
                                                   distances=distances, max_netw_assign_dist=150)
        log(f"POI / transit accessibility: {time.time() - t:.0f}s")

    nodes_gdf.index = nodes_gdf.index.astype(str)
    if not a.skip_global:
        glob, mode = global_networkit(G_proj)
        nodes_gdf = nodes_gdf.join(glob, how="left")
        nodes_gdf.attrs["betweenness_mode"] = mode

    ll = nodes_gdf.geometry.to_crs(4326)
    nodes_gdf["lon"], nodes_gdf["lat"] = ll.x, ll.y
    keep = [c for c in nodes_gdf.columns if c.startswith(("cc_", "nk_")) or c in ("lon", "lat", "live")]
    out = gpd.GeoDataFrame(nodes_gdf[keep], geometry=nodes_gdf.geometry.values, crs=epsg)
    out.to_parquet(base / f"nodes_{a.net}.parquet")
    log(f"wrote {base / f'nodes_{a.net}.parquet'}: {out.shape}")


if __name__ == "__main__":
    main()
