"""Huff model with real street-network distance (and drive time) instead of a circuity guess.

site_scoring_huff.py turns straight-line distance into road distance with assumed circuity
factors (1.25-1.35 per city, "placeholder, NOT fitted"). The OSM drive graphs built for the
centrality work let us replace that guess with measured shortest paths:

  1. circuity  = network km / straight km for every hex-outlet pair, per metro
                 -> the factor the existing model should have used
  2. Huff       = same alpha, beta per outlet_type, attractiveness and same-brand/same-city
                 competition as site_scoring_huff.py, but D_ij = network km (variant
                 "network") vs straight km x assumed city factor (variant "assumed")
  3. drive time = minutes from each hex to each outlet (OSMnx-imputed speeds), reported for
                 the demand each outlet captures

Demand = WorldPop 2025 population per H3 hex (real), identical in both variants, so every
difference comes from distance alone.

Outputs (in --out/<area>/): huff_outlets.csv, huff_pairs.csv, circuity.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from areas import AREAS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:  # reuse the existing model's parameters so the comparison is like-for-like
    import site_scoring_huff as base
except ImportError:  # job snapshot without the repo root: copy of the same constants
    base = None

ALPHA = getattr(base, "ALPHA", 1.0)
MIN_DIST_KM = getattr(base, "MIN_DIST_KM", 0.15)
BETA_BY_OUTLET_TYPE = getattr(base, "BETA_BY_OUTLET_TYPE", {"destination_hub": 1.1, "transit_adjacent": 1.5, "neighborhood": 2.3})
DEFAULT_BETA = getattr(base, "DEFAULT_BETA", 1.8)
TYPE_MULT = getattr(base, "OUTLET_TYPE_ATTRACT_MULTIPLIER", {"destination_hub": 1.6, "transit_adjacent": 1.2, "neighborhood": 1.0})
TIER_MULT = getattr(base, "TIER_MULTIPLIER", {"flagship": 1.3, "standard": 1.0})
ASSUMED_CIRCUITY = getattr(base, "ROAD_CIRCUITY_BY_CITY", {"Jakarta": 1.35, "Bandung": 1.25, "Surabaya": 1.30})
ASSUMED_DEFAULT = 1.3
OVERLAP_P = 0.15  # same threshold as compute_cannibalization()
MAX_KM = 60  # paths longer than this carry negligible Huff weight


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def graph_arrays(G):
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    u, v, length, ttime = [], [], [], []
    for a, b, d in G.edges(data=True):
        u.append(idx[a]); v.append(idx[b])
        length.append(float(d.get("length", 1.0)))
        ttime.append(float(d.get("travel_time", d.get("length", 1.0) / 8.3)))
    n = len(nodes)
    # duplicate edges between the same pair: csr sums them, so keep the minimum explicitly
    df = pd.DataFrame({"u": u, "v": v, "len": length, "tt": ttime}).groupby(["u", "v"], as_index=False).min()
    L = csr_matrix((df["len"].values, (df.u.values, df.v.values)), shape=(n, n))
    T = csr_matrix((df["tt"].values, (df.u.values, df.v.values)), shape=(n, n))
    xs = np.array([G.nodes[k]["x"] for k in nodes]); ys = np.array([G.nodes[k]["y"] for k in nodes])
    return nodes, idx, L, T, xs, ys


def nearest(xs, ys, lon, lat):
    from sklearn.neighbors import BallTree
    tree = BallTree(np.radians(np.c_[ys, xs]), metric="haversine")
    d, i = tree.query(np.radians(np.c_[lat, lon]), k=1)
    return i[:, 0], d[:, 0] * 6371.0


def huff(dist_km, attract, beta, pop):
    """dist_km: hexes x outlets. Returns P (hexes x outlets) and captured demand per outlet."""
    d = np.maximum(dist_km, MIN_DIST_KM)
    w = attract[None, :] ** ALPHA / d ** beta[None, :]
    w[~np.isfinite(d)] = 0.0
    tot = w.sum(1, keepdims=True)
    P = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
    return P, (P * pop[:, None]).sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", required=True, choices=AREAS)
    ap.add_argument("--data", required=True, help="network-v1 folder (raw graphml + hex_features)")
    ap.add_argument("--outlets", required=True, help="outlets.csv (Java outlets with area/tier/type)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    area = AREAS[a.area]
    w_, s_, e_, n_ = area["bbox"]
    out = Path(a.out) / a.area
    out.mkdir(parents=True, exist_ok=True)

    outlets = pd.read_csv(a.outlets)
    outlets = outlets[outlets.longitude.between(w_, e_) & outlets.latitude.between(s_, n_)
                      & outlets.gross_floor_area_sqm.notna()].reset_index(drop=True)
    tier = outlets.store_tier.map(TIER_MULT).fillna(1.0)
    otype = outlets.outlet_type.map(TYPE_MULT).fillna(1.0)
    attract = (outlets.gross_floor_area_sqm * tier * otype).values.astype(float)
    beta = outlets.outlet_type.map(BETA_BY_OUTLET_TYPE).fillna(DEFAULT_BETA).values.astype(float)
    log(f"{a.area}: {len(outlets)} outlets")

    hexes = pd.read_parquet(Path(a.data) / a.area / "hex_features.parquet", columns=["h3", "lat", "lon", "ctx_pop_hex"])
    if "h3" in hexes.columns:
        hexes = hexes.set_index("h3")
    hexes = hexes[hexes.ctx_pop_hex > 0]
    pop = hexes.ctx_pop_hex.values.astype(float)
    log(f"{len(hexes):,} populated hexes, {pop.sum():,.0f} people")

    t = time.time()
    G = ox.load_graphml(Path(a.data) / a.area / "raw" / "drive.graphml")
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    nodes, idx, L, T, xs, ys = graph_arrays(G)
    log(f"drive graph {len(nodes):,} nodes loaded with speeds ({time.time() - t:.0f}s)")

    hex_node, hex_snap = nearest(xs, ys, hexes.lon.values, hexes.lat.values)
    out_node, out_snap = nearest(xs, ys, outlets.longitude.values, outlets.latitude.values)

    # Shortest paths TO each outlet = from the outlet on the reversed graph.
    t = time.time()
    dl = dijkstra(L.T.tocsr(), directed=True, indices=out_node, limit=MAX_KM * 1000)  # outlets x nodes, metres
    dt = dijkstra(T.T.tocsr(), directed=True, indices=out_node, limit=2 * 3600)       # seconds
    log(f"dijkstra from {len(outlets)} outlets: {time.time() - t:.0f}s")

    # hexes x outlets; add the straight snap distances at both ends (walk/drive onto the network)
    net_km = dl[:, hex_node].T / 1000 + hex_snap[:, None] + out_snap[None, :]
    drive_min = dt[:, hex_node].T / 60 + (hex_snap[:, None] + out_snap[None, :]) / 0.5  # snap at ~30 km/h
    straight_km = haversine_km(hexes.lat.values[:, None], hexes.lon.values[:, None],
                               outlets.latitude.values[None, :], outlets.longitude.values[None, :])
    factor = ASSUMED_CIRCUITY.get(area["city"], ASSUMED_DEFAULT)
    assumed_km = straight_km * factor

    # 1. Measured circuity (pairs 1-20 km apart, reachable)
    m = (straight_km >= 1) & (straight_km <= 20) & np.isfinite(net_km)
    ratio = (net_km / straight_km)[m]
    bands = {}
    for lo, hi in [(1, 3), (3, 7), (7, 12), (12, 20)]:
        mm = m & (straight_km >= lo) & (straight_km < hi)
        r = (net_km / straight_km)[mm]
        bands[f"{lo}-{hi} km"] = round(float(np.median(r)), 3) if r.size else None
    circ = {"area": a.area, "city": area["city"], "assumed_factor": factor,
            "measured_median": round(float(np.median(ratio)), 3),
            "measured_p25": round(float(np.percentile(ratio, 25)), 3),
            "measured_p75": round(float(np.percentile(ratio, 75)), 3),
            "measured_by_straight_distance": bands, "pairs": int(ratio.size),
            "unreachable_share": round(float((~np.isfinite(net_km)).mean()), 4)}
    (out / "circuity.json").write_text(json.dumps(circ, indent=2))
    log(f"circuity: assumed {factor} vs measured median {circ['measured_median']} (IQR {circ['measured_p25']}-{circ['measured_p75']})")

    # 2. Huff per brand (same-brand competition within the area, as in the base model)
    rows, pairs = [], []
    for brand, grp in outlets.groupby("brand"):
        j = grp.index.values
        res = {}
        for name, D in (("assumed", assumed_km[:, j]), ("network", net_km[:, j])):
            P, cap = huff(D, attract[j], beta[j], pop)
            res[name] = (P, cap)
        P_net = res["network"][0]
        for k, oid in enumerate(j):
            o = outlets.loc[oid]
            cap_a, cap_n = res["assumed"][1][k], res["network"][1][k]
            share = P_net[:, k] * pop
            tmin = drive_min[:, oid]
            ok = np.isfinite(tmin) & (share > 0)
            med_min = float(np.average(tmin[ok], weights=share[ok])) if ok.any() else np.nan
            rows.append({"outlet_id": o.outlet_id, "brand": brand, "outlet_name": o.outlet_name,
                         "location_type": o.location_type, "outlet_type": o.outlet_type,
                         "captured_assumed": round(cap_a), "captured_network": round(cap_n),
                         "change_pct": round((cap_n / cap_a - 1) * 100, 1) if cap_a else np.nan,
                         "share_assumed_pct": round(cap_a / res["assumed"][1].sum() * 100, 1),
                         "share_network_pct": round(cap_n / res["network"][1].sum() * 100, 1),
                         "mean_drive_min_captured": round(med_min, 1),
                         "outlet_snap_m": round(out_snap[oid] * 1000)})
        # 3. Cannibalisation: demand both outlets pull with P >= threshold
        for x in range(len(j)):
            for y in range(x + 1, len(j)):
                rec = {"brand": brand, "outlet_a": outlets.loc[j[x]].outlet_id, "outlet_b": outlets.loc[j[y]].outlet_id}
                for name in ("assumed", "network"):
                    P = res[name][0]
                    both = (P[:, x] >= OVERLAP_P) & (P[:, y] >= OVERLAP_P)
                    rec[f"contested_pop_{name}"] = round(float(pop[both].sum()))
                pairs.append(rec)
    res_df = pd.DataFrame(rows)
    for col in ("assumed", "network"):
        res_df[f"rank_{col}"] = res_df.groupby("brand")[f"captured_{col}"].rank(ascending=False).astype(int)
    res_df.to_csv(out / "huff_outlets.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "huff_pairs.csv", index=False)
    log(f"outlets: median change {res_df.change_pct.median():+.1f}%, rank changes {(res_df.rank_assumed != res_df.rank_network).sum()}/{len(res_df)}")
    log(res_df[["outlet_name", "captured_assumed", "captured_network", "change_pct", "mean_drive_min_captured"]].to_string(index=False))


if __name__ == "__main__":
    main()
