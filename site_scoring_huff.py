"""
Site Scoring Engine — Huff Gravity Model
=========================================
Bagian "analytics engine" dari arsitektur location analytics.

Model:  P_ij = (A_j^alpha / D_ij^beta_j) / sum_k(A_k^alpha / D_ik^beta_k)

  P_ij   = probabilitas demand di grid cell i "tertarik" ke outlet j
  A_j    = attractiveness outlet j (luas outlet x tier x outlet_type multiplier)
  D_ij   = jarak EFEKTIF (km) antara grid cell i dan outlet j — jarak lurus
           (haversine) dikali circuity factor (kota x land use), BUKAN jarak
           lurus mentah. Lihat ROAD_CIRCUITY_BY_CITY / LAND_USE_CIRCUITY_ADJ
           dan SITE_SCORING_METHODOLOGY.md §Road-distance proxy untuk rasional
           dan batasannya (ini masih proxy, bukan travel-time dari routing API).
  alpha  = 1.0     (elastisitas attractiveness)
  beta_j = per outlet_type, BUKAN satu angka global (lihat BETA_BY_OUTLET_TYPE
           di bawah) — outlet `destination_hub` (mis. flagship di Grand
           Indonesia) menarik demand dari radius jauh & catchment tidak
           simetris, sedangkan outlet `neighborhood` didominasi proximity.
           Memakai satu beta untuk semua outlet akan meremehkan jangkauan
           hub dan melebih-lebihkan jangkauan neighborhood. Lihat
           SITE_SCORING_METHODOLOGY.md §Heterogeneous pull untuk rasional.

Kompetisi dimodelkan HANYA antar outlet brand yang SAMA di kota yang sama
(pelanggan Marugame tidak "bersaing" melawan pilihan The Harvest dalam model
ini — beda kategori kebutuhan). Kalau nanti mau model cross-brand/cross-
category, cukup ubah filter `same_brand` di compute_huff_scores().

Output:
  1. outlet_site_scores.csv          - estimasi demand tertangkap per outlet
  2. cannibalization_pairs.csv       - pasangan outlet brand sama yang overlap,
                                        DIRECTIONAL (a_demand_at_risk_pct vs
                                        b_demand_at_risk_pct bisa berbeda jauh)
  3. cannibalization_network.csv     - agregat per outlet: total demand at
                                        risk dari SELURUH tetangga yang
                                        overlap (bukan cuma pasangan terburuk)
  4. whitespace_candidates.csv       - grid cell dengan demand tinggi tapi
                                        exposure ke outlet existing rendah
  5. outlet_catchment_validation.csv - dispersion jarak customer OBSERVED
                                        (dari customer_origin_sample.csv) per
                                        outlet, dibandingkan dengan asumsi
                                        beta/outlet_type di atas — dipakai untuk
                                        flag outlet yang mungkin salah klasifikasi

Run: python3 site_scoring_huff.py
"""

import math
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(__file__)

ALPHA = 1.0
MIN_DIST_KM = 0.15  # floor jarak, hindari division blow-up untuk cell sangat dekat outlet

# Distance-decay steepness per outlet_type. Landai (kecil) = customer mau
# menempuh jarak jauh (hub); curam (besar) = demand didominasi proximity
# (neighborhood). Nilai ini paralel dengan mean_km di ORIGIN_TYPE_PARAMS pada
# generate_dummy_data.py — keduanya harus dikalibrasi bersama saat data riil
# tersedia, bukan diubah sendiri-sendiri.
BETA_BY_OUTLET_TYPE = {
    "destination_hub": 1.1,
    "transit_adjacent": 1.5,
    "neighborhood": 2.3,
}
DEFAULT_BETA = 1.8  # fallback kalau outlet_type tidak ada (mis. data lama)

# Attractiveness tidak hanya soal luas fisik — outlet destination_hub punya
# daya tarik ekstra (assembly point, transit/event traffic) yang tidak
# tercermin dari gross_floor_area_sqm saja.
OUTLET_TYPE_ATTRACT_MULTIPLIER = {
    "destination_hub": 1.6,
    "transit_adjacent": 1.2,
    "neighborhood": 1.0,
}

# Expected mean home-to-outlet distance (km) per outlet_type, mirrored from
# generate_dummy_data.py's ORIGIN_TYPE_PARAMS — used only as a sanity-check
# reference in compute_catchment_validation(), not fed back into the model.
EXPECTED_MEAN_KM_BY_TYPE = {
    "destination_hub": 6.0,
    "transit_adjacent": 3.2,
    "neighborhood": 1.3,
}

TIER_MULTIPLIER = {"flagship": 1.3, "standard": 1.0}
INCOME_SPEND_MULTIPLIER = {
    "low": 0.6, "lower_middle": 0.8, "middle": 1.0, "upper_middle": 1.3, "high": 1.8,
}

# --- Road-distance proxy -----------------------------------------------
# Straight-line (as-the-crow-flies) distance systematically understates real
# travel distance/time in Indonesian metros — grid-locked roads, toll
# detours, one-way systems, rivers. Until a routing API/OSRM is wired in,
# approximate this with a circuity factor: effective_distance = straight_km
# x city_factor x land_use_factor. Typical urban circuity indices are
# ~1.2-1.4x; these are placeholder estimates, NOT fitted to real trip data.
ROAD_CIRCUITY_BY_CITY = {
    "Jakarta": 1.35,   # dense grid, toll detours, one-ways
    "Bandung": 1.25,   # hillier but smaller/more direct grid
    "Surabaya": 1.30,
}
# Adjustment on top of the city factor based on the ORIGIN cell's land use —
# dense CBD/office grids add detour, residential streets tend more direct.
LAND_USE_CIRCUITY_ADJ = {
    "office": 1.10,
    "commercial": 1.05,
    "mixed": 1.00,
    "residential": 0.95,
}


def effective_distance_km(straight_km, city, land_use_type):
    city_factor = ROAD_CIRCUITY_BY_CITY.get(city, 1.3)
    land_use_factor = LAND_USE_CIRCUITY_ADJ.get(land_use_type, 1.0)
    return straight_km * city_factor * land_use_factor


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def attractiveness(outlet_row):
    tier_mult = TIER_MULTIPLIER.get(outlet_row["store_tier"], 1.0)
    type_mult = OUTLET_TYPE_ATTRACT_MULTIPLIER.get(outlet_row.get("outlet_type"), 1.0)
    return outlet_row["gross_floor_area_sqm"] * tier_mult * type_mult


def outlet_beta(outlet_row):
    return BETA_BY_OUTLET_TYPE.get(outlet_row.get("outlet_type"), DEFAULT_BETA)


def demand_potential(cell_row):
    """Synthetic demand index — NOT a currency value. population x spend multiplier."""
    mult = INCOME_SPEND_MULTIPLIER.get(cell_row["income_bracket"], 1.0)
    return cell_row["population_est"] * mult


# ---------------------------------------------------------------------------
# 1. Huff probabilities + captured demand per outlet
# ---------------------------------------------------------------------------
def compute_huff_scores(outlets, grid):
    outlets = outlets.copy()
    outlets["attractiveness"] = outlets.apply(attractiveness, axis=1)
    grid = grid.copy()
    grid["demand_potential"] = grid.apply(demand_potential, axis=1)

    outlet_captured = {oid: 0.0 for oid in outlets["outlet_id"]}
    whitespace_rows = []

    for city in grid["city"].unique():
        city_grid = grid[grid["city"] == city]
        for brand in outlets["brand"].unique():
            city_outlets = outlets[(outlets["city"] == city) & (outlets["brand"] == brand)]
            if city_outlets.empty:
                continue

            for _, cell in city_grid.iterrows():
                weights = []
                for _, o in city_outlets.iterrows():
                    straight_d = haversine_km(cell["centroid_lat"], cell["centroid_lon"],
                                               o["latitude"], o["longitude"])
                    d = max(effective_distance_km(straight_d, city, cell["land_use_type"]), MIN_DIST_KM)
                    w = (o["attractiveness"] ** ALPHA) / (d ** outlet_beta(o))
                    weights.append((o["outlet_id"], w))

                total_w = sum(w for _, w in weights)
                if total_w <= 0:
                    continue

                for outlet_id, w in weights:
                    p_ij = w / total_w
                    outlet_captured[outlet_id] += p_ij * cell["demand_potential"]

                # exposure = total_w itself (unnormalized gravity sum) — used for whitespace
                whitespace_rows.append({
                    "grid_id": cell["grid_id"], "city": city, "brand": brand,
                    "centroid_lat": cell["centroid_lat"], "centroid_lon": cell["centroid_lon"],
                    "demand_potential": cell["demand_potential"],
                    "total_exposure": total_w,
                })

    scores = outlets[["outlet_id", "brand", "city", "outlet_name", "location_type"]].copy()
    scores["estimated_captured_demand"] = scores["outlet_id"].map(outlet_captured)

    # market share within city+brand
    scores["city_brand_total"] = scores.groupby(["city", "brand"])["estimated_captured_demand"].transform("sum")
    scores["market_share_pct"] = (scores["estimated_captured_demand"] / scores["city_brand_total"] * 100).round(1)
    scores = scores.drop(columns="city_brand_total").sort_values(
        ["brand", "estimated_captured_demand"], ascending=[True, False]
    )

    return scores, pd.DataFrame(whitespace_rows)


# ---------------------------------------------------------------------------
# 2. Cannibalization — directional pairwise overlap + network-level exposure
# ---------------------------------------------------------------------------
def compute_cannibalization(outlets, grid, overlap_threshold=0.15):
    """
    For each same-brand, same-city outlet pair, estimate contested cells (both
    outlets get Huff probability >= overlap_threshold) AND the DIRECTIONAL
    demand at risk: what share of outlet A's own captured demand comes from
    cells it contests with B, vs the share of B's captured demand contested
    by A. These two shares are rarely equal — a destination_hub with a large
    total captured demand can share the same contested cells as a small
    neighborhood outlet next door, but lose only a sliver of its own demand
    while the neighborhood outlet loses a large fraction of its (much
    smaller) base. A single symmetric "overlap %" hides this.

    Also aggregates a network-level view per outlet: total demand at risk
    across ALL contested neighbors (not just the single worst pair), since
    an outlet boxed in by three moderate overlaps can be under more real
    competitive pressure than one outlet with one large overlap.

    Returns (pairs_df, network_df).
    """
    pair_rows = []
    network_rows = []
    outlets = outlets.copy()
    outlets["attractiveness"] = outlets.apply(attractiveness, axis=1)
    outlet_type_by_id = dict(zip(outlets["outlet_id"], outlets["outlet_type"]))
    grid = grid.copy()
    grid["demand_potential"] = grid.apply(demand_potential, axis=1)

    for city in grid["city"].unique():
        city_grid = grid[grid["city"] == city]
        for brand in outlets["brand"].unique():
            city_outlets = outlets[(outlets["city"] == city) & (outlets["brand"] == brand)]
            ids = city_outlets["outlet_id"].tolist()
            if len(ids) < 2:
                continue

            # P matrix (rows=cells, cols=outlets) + matching demand potential per cell
            p_matrix, demand_list = [], []
            for _, cell in city_grid.iterrows():
                weights = {}
                for _, o in city_outlets.iterrows():
                    straight_d = haversine_km(cell["centroid_lat"], cell["centroid_lon"],
                                               o["latitude"], o["longitude"])
                    d = max(effective_distance_km(straight_d, city, cell["land_use_type"]), MIN_DIST_KM)
                    weights[o["outlet_id"]] = (o["attractiveness"] ** ALPHA) / (d ** outlet_beta(o))
                total_w = sum(weights.values())
                p_matrix.append({k: v / total_w for k, v in weights.items()})
                demand_list.append(cell["demand_potential"])

            captured = {oid: 0.0 for oid in ids}
            for p, dem in zip(p_matrix, demand_list):
                for oid, pij in p.items():
                    captured[oid] += pij * dem

            demand_at_risk = {oid: {} for oid in ids}  # oid -> {neighbor_id: at_risk_pct}
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    contested_cells = 0
                    a_contested_demand = b_contested_demand = 0.0
                    for p, dem in zip(p_matrix, demand_list):
                        pa, pb = p.get(a, 0), p.get(b, 0)
                        if pa >= overlap_threshold and pb >= overlap_threshold:
                            contested_cells += 1
                            a_contested_demand += pa * dem
                            b_contested_demand += pb * dem
                    if contested_cells == 0:
                        continue

                    overlap_pct = round(contested_cells / len(p_matrix) * 100, 1)
                    a_risk_pct = round(a_contested_demand / captured[a] * 100, 1) if captured[a] > 0 else 0.0
                    b_risk_pct = round(b_contested_demand / captured[b] * 100, 1) if captured[b] > 0 else 0.0
                    pair_rows.append({
                        "brand": brand, "city": city,
                        "outlet_a": a, "outlet_b": b,
                        "contested_cells_pct": overlap_pct,
                        "a_demand_at_risk_pct": a_risk_pct,
                        "b_demand_at_risk_pct": b_risk_pct,
                        # positive => A is more exposed to B than B is to A
                        "net_asymmetry_pct": round(a_risk_pct - b_risk_pct, 1),
                    })
                    demand_at_risk[a][b] = a_risk_pct
                    demand_at_risk[b][a] = b_risk_pct

            for oid in ids:
                neighbors = demand_at_risk[oid]
                if not neighbors:
                    continue
                worst_neighbor, worst_pct = max(neighbors.items(), key=lambda kv: kv[1])
                network_rows.append({
                    "outlet_id": oid, "brand": brand, "city": city,
                    "outlet_type": outlet_type_by_id.get(oid),
                    "estimated_captured_demand": round(captured[oid], 1),
                    "n_contested_neighbors": len(neighbors),
                    # sum across neighbors: cumulative competitive pressure, can exceed
                    # 100% if the same demand is separately contested by multiple neighbors
                    "total_demand_at_risk_pct": round(sum(neighbors.values()), 1),
                    "most_threatening_neighbor": worst_neighbor,
                    "most_threatening_neighbor_risk_pct": worst_pct,
                })

    pairs_df = pd.DataFrame(pair_rows)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values("contested_cells_pct", ascending=False)

    network_df = pd.DataFrame(network_rows)
    if not network_df.empty:
        network_df = network_df.sort_values("total_demand_at_risk_pct", ascending=False)

    return pairs_df, network_df


# ---------------------------------------------------------------------------
# 3. Whitespace — high demand, low existing exposure
# ---------------------------------------------------------------------------
def compute_whitespace(whitespace_df, top_n=5):
    results = []
    for (city, brand), sub in whitespace_df.groupby(["city", "brand"]):
        sub = sub.copy()
        # normalize both metrics 0-1 so they're comparable
        sub["demand_norm"] = (sub["demand_potential"] - sub["demand_potential"].min()) / (
            sub["demand_potential"].max() - sub["demand_potential"].min() + 1e-9
        )
        sub["exposure_norm"] = (sub["total_exposure"] - sub["total_exposure"].min()) / (
            sub["total_exposure"].max() - sub["total_exposure"].min() + 1e-9
        )
        # whitespace score: high demand, low exposure
        sub["whitespace_score"] = (sub["demand_norm"] * (1 - sub["exposure_norm"])).round(3)
        top = sub.sort_values("whitespace_score", ascending=False).head(top_n)
        results.append(top[["grid_id", "city", "brand", "centroid_lat", "centroid_lon",
                             "demand_potential", "whitespace_score"]])
    return pd.concat(results, ignore_index=True).sort_values(
        ["brand", "whitespace_score"], ascending=[True, False]
    )


# ---------------------------------------------------------------------------
# 4. Catchment validation — model assumption vs. observed origin dispersion
# ---------------------------------------------------------------------------
def compute_catchment_validation(outlets, origin_sample, deviation_flag_pct=0.5):
    """
    Sanity-check the beta/attractiveness assumptions above against the
    observed spread of customer_origin_sample.csv's distance_km per outlet.
    Flags outlets whose observed median distance deviates a lot from the
    outlet_type's expected mean — a signal the outlet may be misclassified
    (e.g. a "neighborhood" outlet that is actually behaving like a hub).
    """
    stats = origin_sample.groupby("outlet_id")["distance_km"].agg(
        observed_median_km="median", observed_p90_km=lambda s: s.quantile(0.9), n_sampled_customers="count"
    ).reset_index()

    validation = outlets[["outlet_id", "brand", "outlet_type"]].merge(stats, on="outlet_id", how="left")
    validation["expected_mean_km"] = validation["outlet_type"].map(EXPECTED_MEAN_KM_BY_TYPE)
    validation["assumed_beta"] = validation["outlet_type"].map(BETA_BY_OUTLET_TYPE)
    validation["deviation_pct"] = (
        (validation["observed_median_km"] - validation["expected_mean_km"]) / validation["expected_mean_km"]
    ).round(2)
    validation["reclassify_review"] = validation["deviation_pct"].abs() >= deviation_flag_pct

    return validation.sort_values("deviation_pct", ascending=False)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    outlets = pd.read_csv(os.path.join(DATA_DIR, "outlets.csv"))
    grid = pd.read_csv(os.path.join(DATA_DIR, "demographics_grid.csv"))

    print("Menghitung Huff site scores...")
    scores, whitespace_raw = compute_huff_scores(outlets, grid)
    scores.to_csv(os.path.join(DATA_DIR, "outlet_site_scores.csv"), index=False)
    print(scores.to_string(index=False))

    print("\nMenghitung cannibalization antar outlet sebrand (directional + network)...")
    cannib_pairs, cannib_network = compute_cannibalization(outlets, grid)
    cannib_pairs.to_csv(os.path.join(DATA_DIR, "cannibalization_pairs.csv"), index=False)
    cannib_network.to_csv(os.path.join(DATA_DIR, "cannibalization_network.csv"), index=False)
    print(cannib_pairs.head(10).to_string(index=False) if not cannib_pairs.empty else "(tidak ada overlap signifikan)")
    print("\nOutlet dengan total demand-at-risk tertinggi (network-level):")
    print(cannib_network.head(10).to_string(index=False) if not cannib_network.empty else "(tidak ada exposure signifikan)")

    print("\nMenghitung whitespace candidates...")
    whitespace = compute_whitespace(whitespace_raw)
    whitespace.to_csv(os.path.join(DATA_DIR, "whitespace_candidates.csv"), index=False)
    print(whitespace.to_string(index=False))

    origin_path = os.path.join(DATA_DIR, "customer_origin_sample.csv")
    if os.path.exists(origin_path):
        print("\nMemvalidasi asumsi beta/outlet_type terhadap observed origin dispersion...")
        origin_sample = pd.read_csv(origin_path)
        validation = compute_catchment_validation(outlets, origin_sample)
        validation.to_csv(os.path.join(DATA_DIR, "outlet_catchment_validation.csv"), index=False)
        flagged = validation[validation["reclassify_review"]]
        print(validation.to_string(index=False))
        if not flagged.empty:
            print(f"\n{len(flagged)} outlet menyimpang >= 50% dari expected_mean_km outlet_type-nya "
                  "— pertimbangkan review klasifikasi outlet_type-nya.")
    else:
        print("\n(customer_origin_sample.csv tidak ditemukan — lewati catchment validation)")
