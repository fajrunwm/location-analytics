"""
Location Analytics Data Generator — F&B (Marugame Udon & The Harvest)

Mixed-provenance data: outlet-level LOCATION fields (outlet_name, city,
district, location_type, latitude, longitude) plus `rating`/`review_count`
are REAL — scraped from Google Maps (via Claude Desktop, 2026-09-10),
filtered to Java island only. See REAL_OUTLETS_MARUGAME /
REAL_OUTLETS_HARVEST below for the exact list. Everything else —
demographics, competitor POIs, foot traffic, customer origin, and the
remaining outlet attributes (floor area, seating, open date) — is synthetic,
generated here. `store_tier` and `outlet_type` are DERIVED from the real
`review_count`/`location_type` signal rather than pure random draws (see
assign_store_tier / assign_outlet_type below) — a light synthesis on top of
real data, not invented from scratch.

Generates 5 linked datasets:

    1. outlets.csv                 - REAL locations/rating + synthetic attributes
    2. demographics_grid.csv       - synthetic 500m grid population/income/age profile
    3. competitor_pois.csv         - synthetic competitor & complementary POIs
    4. foot_traffic_transactions.csv - simulated daily/daypart visit & sales pattern
    5. customer_origin_sample.csv  - synthetic per-customer home-to-outlet trips,
                                      used to derive each outlet's catchment
                                      "pull" (origin dispersion), not just its size

outlet_type captures that outlets are not flat/uniform draws: a real
assembly-point location (a mega mall, an airport terminal) pulls customers
from a much wider and more irregular area than a neighborhood street-front
outlet, independent of floor size. customer_origin_sample.csv lets that pull
be measured directly (spread of home locations) rather than assumed from a
single global distance-decay parameter.

demographics_grid/competitor_pois/foot_traffic/customer_origin are still
fully synthetic and scattered around each metro's city center (jittered,
not tied to the real outlet coordinates above) — safe to treat as such; no
real person or transaction is represented anywhere in this repo.

Run:  python3 generate_dummy_data.py
Output written to ./output/
"""

import numpy as np
import pandas as pd
import os
import math
import random
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 0. Reference geography — city centers & rough metro radius (km)
# ---------------------------------------------------------------------------
# Weights are proportional to how many REAL outlets (both brands combined)
# landed in each metro below (36/72 Jakarta, 10/72 Bandung, 11/72 Surabaya,
# 6/72 Semarang, 5/72 Yogyakarta, 4/72 Malang) — used only for scattering
# synthetic competitor_pois, not for placing outlets (those are fixed real
# points).
CITIES = {
    "Jakarta":    {"lat": -6.2088, "lon": 106.8456, "radius_km": 18, "weight": 0.50},
    "Bandung":    {"lat": -6.9175, "lon": 107.6191, "radius_km": 10, "weight": 0.14},
    "Surabaya":   {"lat": -7.2575, "lon": 112.7521, "radius_km": 12, "weight": 0.15},
    "Semarang":   {"lat": -6.9932, "lon": 110.4203, "radius_km": 9,  "weight": 0.08},
    "Yogyakarta": {"lat": -7.7956, "lon": 110.3695, "radius_km": 8,  "weight": 0.07},
    "Malang":     {"lat": -7.9666, "lon": 112.6326, "radius_km": 7,  "weight": 0.06},
}

DISTRICTS = {
    "Jakarta":    ["Jakarta Selatan", "Jakarta Pusat", "Jakarta Barat", "Jakarta Timur", "Jakarta Utara", "Tangerang", "Bekasi", "Depok", "Bogor"],
    "Bandung":    ["Bandung Wetan", "Coblong", "Sukajadi", "Cidadap", "Buah Batu", "Batununggal", "Gedebage", "Cicendo"],
    "Surabaya":   ["Gubeng", "Wonokromo", "Rungkut", "Tegalsari", "Sukolilo", "Mulyorejo", "Bubutan", "Lakarsantri"],
    "Semarang":   ["Semarang Tengah", "Semarang Barat", "Gajahmungkur", "Pedurungan", "Candisari"],
    "Yogyakarta": ["Gondokusuman", "Sleman", "Umbulharjo", "Kotagede", "Depok (Sleman)"],
    "Malang":     ["Klojen", "Lowokwaru", "Blimbing", "Sukun", "Kedungkandang"],
}

def km_to_deg_lat(km):
    return km / 111.0

def km_to_deg_lon(km, lat):
    return km / (111.0 * math.cos(math.radians(lat)))

def random_point_in_city(city):
    c = CITIES[city]
    # sample radius with bias toward center (sqrt for uniform area density)
    r = c["radius_km"] * math.sqrt(random.random())
    theta = random.uniform(0, 2 * math.pi)
    dlat = km_to_deg_lat(r * math.sin(theta))
    dlon = km_to_deg_lon(r * math.cos(theta), c["lat"])
    return c["lat"] + dlat, c["lon"] + dlon

def pick_city():
    cities = list(CITIES.keys())
    weights = [CITIES[c]["weight"] for c in cities]
    return random.choices(cities, weights=weights, k=1)[0]

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))

def point_at_distance_bearing(lat, lon, distance_km, bearing_deg):
    dlat = km_to_deg_lat(distance_km * math.cos(math.radians(bearing_deg)))
    dlon = km_to_deg_lon(distance_km * math.sin(math.radians(bearing_deg)), lat)
    return lat + dlat, lon + dlon

def assign_outlet_type(location_type, review_count):
    """Classify an outlet's catchment behaviour from REAL signals.

    destination_hub    - a busy real mall (review_count >= 500) — assembly-point
                          behaviour, wide/irregular pull
    transit_adjacent    - a real mall but with a smaller/quieter footprint
                          (review_count < 500) — moderate pull
    neighborhood        - street-front/local stop, tight pull

    review_count is a genuine popularity proxy from Google Maps, not invented —
    using it (rather than guessing from the outlet name) is why this split
    ended up 3-way balanced instead of collapsing to all-hub or all-neighborhood.
    """
    if location_type != "mall":
        return "neighborhood"
    return "destination_hub" if (review_count or 0) >= 500 else "transit_adjacent"

# Manual overrides for cases where this outlet's own review_count understates
# the real scale of the mall it's in — confirmed via external sources (Ayo
# Bandung's "4 mal terbesar di Bandung", Tripadvisor), not just the address
# text. Applied AFTER assign_outlet_type(); keep this list short and cited,
# it's an exception path, not the default classification mechanism.
OUTLET_TYPE_OVERRIDES = {
    "Marugame Udon Paris Van Java": "destination_hub",  # one of Bandung's biggest/best-known malls; this branch's own review_count (417) undercounts that
}

def assign_store_tier(outlet_type, review_count):
    """flagship more likely for busy hubs — informed by real review_count,
    not pure random, though the flagship/standard label itself is still a
    synthesized attribute (Google Maps doesn't expose store tier)."""
    p_flagship = 0.45 if outlet_type == "destination_hub" else (0.20 if (review_count or 0) >= 500 else 0.08)
    return random.choices(["flagship", "standard"], weights=[p_flagship, 1 - p_flagship])[0]

# ---------------------------------------------------------------------------
# 1. OUTLETS — REAL locations, scraped from Google Maps (Java island only)
# ---------------------------------------------------------------------------
# outlet_name, city, district, location_type, latitude, longitude, rating,
# review_count are REAL — scraped from Google Maps via Claude Desktop on
# 2026-09-10, filtered here to Java island (Jabodetabek/Jakarta metro,
# Bandung, Surabaya, Semarang, Yogyakarta, Malang) from the full-Indonesia
# scrape in marugame_outlets_indonesia.csv / harvest_outlets_indonesia.csv
# (which also cover Medan, Makassar, Palembang, Denpasar, Balikpapan —
# excluded here, not because they don't exist, but to keep this tool's
# geographic scope to one island). `district` is parsed from the real
# address text (kecamatan-level where the address included one, else the
# city-administrative name) — real, but not independently verified.
# `location_type` is classified from mall-name keywords in the real
# name/address (see the source notebook); `outlet_type`/`store_tier` are
# then derived from that plus the real `review_count` (see
# assign_outlet_type/assign_store_tier above) rather than pure random draws.
#
# Result: 44 real Marugame Udon + 28 real The Harvest = 72 outlets on Java.
REAL_OUTLETS_MARUGAME = [
    # outlet_name,                                        city,          district,           location_type,      lat,        lon,         rating, review_count
    ("Marugame Udon Lippo Mall Nusantara",                 "Jakarta",     "Jakarta Selatan",  "mall",             -6.2194624, 106.8146084, 4.9,    311),
    ("Marugame Udon Mall Ciputra Grogol",                  "Jakarta",     "Jakarta Barat",    "mall",             -6.1683965, 106.7864982, 4.7,    45),
    ("Marugame Udon WTC Sudirman",                         "Jakarta",     "Jakarta Selatan",  "mall",             -6.2152035, 106.8204371, 4.9,    85),
    ("Marugame Udon Grand Indonesia",                      "Jakarta",     "Jakarta Pusat",    "mall",             -6.1951619, 106.8201593, 4.6,    1506),
    ("Marugame Udon Danau Sunter Utara",                   "Jakarta",     "Jakarta Utara",    "standalone_ruko",  -6.1405963, 106.8572463, 4.6,    2409),
    ("Marugame Udon Mall Kelapa Gading 3",                 "Jakarta",     "Jakarta Utara",    "mall",             -6.1573418, 106.9087133, 4.6,    254),
    ("Marugame Udon Living World Kota Wisata Cibubur",     "Jakarta",     "Gn. Putri",        "mall",             -6.3696850, 106.9596988, 4.3,    187),
    ("Marugame Udon Mall Of Indonesia",                    "Jakarta",     "Jakarta Utara",    "mall",             -6.1509416, 106.8917794, 4.6,    196),
    ("Marugame Udon Green Pramuka",                        "Jakarta",     "Jakarta Pusat",    "mall",             -6.1896767, 106.8742483, 4.5,    895),
    ("Marugame Udon AEON Mall Jakarta Garden City",        "Jakarta",     "Jakarta Timur",    "mall",             -6.1726170, 106.9529259, 4.8,    825),
    ("Marugame Udon Margonda Raya",                        "Jakarta",     "Depok",            "standalone_ruko",  -6.3727952, 106.8331649, 4.6,    2811),
    ("Marugame Udon Tangerang City Mall",                  "Jakarta",     "Tangerang",        "mall",             -6.1945875, 106.6343438, 3.9,    181),
    ("Marugame Udon Summarecon Mall Serpong",               "Jakarta",     "Kelapa Dua",       "mall",             -6.2413492, 106.6280082, 4.5,    2359),
    ("Marugame Udon Bogor Trade Mall",                     "Jakarta",     "Kota Bogor",       "mall",             -6.6050383, 106.7955848, 4.4,    110),
    ("Marugame Udon Supermall Karawaci",                   "Jakarta",     "Tangerang",        "mall",             -6.2268282, 106.6071930, 4.5,    920),
    ("Marugame Udon Botani Square",                        "Jakarta",     "Kota Bogor",       "mall",             -6.6012848, 106.8073393, 4.5,    620),
    ("Marugame Udon Living World Alam Sutera",             "Jakarta",     "Tangerang",        "mall",             -6.2443655, 106.6536326, 4.3,    305),
    ("Marugame Udon Soekarno-Hatta T3",                    "Jakarta",     "Terminal 3",       "mall",             -6.1185669, 106.6657750, 4.5,    1332),
    ("Marugame Udon Trans Studio Cibubur",                 "Jakarta",     "Depok",            "mall",             -6.3753519, 106.9017382, 4.5,    203),
    ("Marugame Udon Cibinong City Mall",                   "Jakarta",     "Kab. Bogor",       "mall",             -6.4842788, 106.8426341, 4.5,    409),
    ("Marugame Udon Riau Bandung",                         "Bandung",     "Bandung Wetan",    "standalone_ruko",  -6.9099827, 107.6256284, 4.6,    6811),
    ("Marugame Udon Summarecon Mall Bandung",              "Bandung",     "Gedebage",         "mall",             -6.9554052, 107.6970883, 4.7,    1221),
    ("Marugame Udon Tenth Avenue Mall",                    "Bandung",     "Buah Batu",        "mall",             -6.9468236, 107.6409540, 4.8,    110),
    ("Marugame Udon Paskal Hyper Square",                  "Bandung",     "Cicendo",          "mall",             -6.9150499, 107.5958171, 4.5,    845),
    ("Marugame Udon Trans Studio Bandung",                 "Bandung",     "Batununggal",      "mall",             -6.9248958, 107.6368870, 4.6,    695),
    ("Marugame Udon Paris Van Java",                       "Bandung",     "Sukajadi",         "mall",             -6.8903644, 107.5965799, 4.5,    417),
    ("Marugame Udon Darmo Surabaya",                       "Surabaya",    "Wonokromo",        "standalone_ruko",  -7.2900595, 112.7392210, 4.7,    862),
    ("Marugame Udon MERR Surabaya",                        "Surabaya",    "Mulyorejo",        "standalone_ruko",  -7.2987185, 112.7816787, 4.4,    360),
    ("Marugame Udon Pakuwon City Mall",                    "Surabaya",    "Mulyorejo",        "mall",             -7.2756380, 112.8052552, 5.0,    12),
    ("Marugame Udon Tunjungan Plaza",                      "Surabaya",    "Tegalsari",        "mall",             -7.2633495, 112.7395080, 4.6,    1870),
    ("Marugame Udon BG Junction Mall",                     "Surabaya",    "Bubutan",          "mall",             -7.2547722, 112.7330883, 4.5,    107),
    ("Marugame Udon Royal Plaza",                          "Surabaya",    "Wonokromo",        "mall",             -7.3094787, 112.7344589, 4.5,    673),
    ("Marugame Udon Pakuwon Supermall",                    "Surabaya",    "Lakarsantri",      "mall",             -7.2887795, 112.6754902, 4.5,    874),
    ("Marugame Udon Galaxy Mall",                          "Surabaya",    "Mulyorejo",        "mall",             -7.2747300, 112.7824347, 4.4,    354),
    ("Marugame Udon Majapahit Semarang",                   "Semarang",    "Semarang",         "standalone_ruko",  -6.9954263, 110.4357404, 4.7,    605),
    ("Marugame Udon Queen City Semarang",                  "Semarang",    "Semarang",         "mall",             -6.9728663, 110.4205609, 4.7,    242),
    ("Marugame Udon Paragon Mall Semarang",                "Semarang",    "Semarang",         "mall",             -6.9788296, 110.4154723, 4.6,    1733),
    ("Marugame Udon DP Mall Semarang",                     "Semarang",    "Semarang",         "mall",             -6.9835909, 110.4126166, 4.6,    370),
    ("Marugame Udon Kaliurang Yogyakarta",                 "Yogyakarta",  "Sleman",           "standalone_ruko",  -7.7576199, 110.3821667, 4.8,    465),
    ("Marugame Udon Plaza Ambarukmo",                      "Yogyakarta",  "Sleman",           "mall",             -7.7828492, 110.4016829, 4.6,    2245),
    ("Marugame Udon Pakuwon Mall Jogja",                   "Yogyakarta",  "Sleman",           "mall",             -7.7592517, 110.3989700, 4.7,    1097),
    ("Marugame Udon Jogja City Mall",                      "Yogyakarta",  "Sleman",           "mall",             -7.7524794, 110.3609946, 4.6,    610),
    ("Marugame Udon Soekarno Hatta Malang",                "Malang",      "Malang",           "standalone_ruko",  -7.9448836, 112.6189486, 4.7,    387),
    ("Marugame Udon Semeru Malang",                        "Malang",      "Malang",           "standalone_ruko",  -7.9731441, 112.6232959, 4.7,    3992),
]

REAL_OUTLETS_HARVEST = [
    ("The Harvest Cakes - Thamrin",                        "Jakarta",     "Tanah Abang",      "standalone_ruko",  -6.1871779, 106.8218720, 4.3, 366),
    ("The Harvest Cakes - Mangga Besar",                   "Jakarta",     "Taman Sari",       "standalone_ruko",  -6.1482221, 106.8295448, 4.5, 709),
    ("The Harvest Cakes - Salemba",                        "Jakarta",     "Senen",            "standalone_ruko",  -6.1901064, 106.8475648, 4.4, 427),
    ("The Harvest Cakes - Kebon Jeruk",                    "Jakarta",     "Kb. Jeruk",        "standalone_ruko",  -6.1999489, 106.7694018, 4.5, 511),
    ("The Harvest Cakes - Senopati",                       "Jakarta",     "Kby. Baru",        "standalone_ruko",  -6.2313781, 106.8099844, 4.2, 697),
    ("The Harvest Cakes - Radio Dalam",                    "Jakarta",     "Kby. Baru",        "standalone_ruko",  -6.2508682, 106.7914162, 4.3, 317),
    ("The Harvest Cakes - Kelapa Gading",                  "Jakarta",     "Klp. Gading Tim.", "standalone_ruko",  -6.1724847, 106.8975346, 4.2, 644),
    ("The Harvest Cakes - Bendungan Hilir",                "Jakarta",     "Tanah Abang",      "standalone_ruko",  -6.2147080, 106.8147157, 4.2, 358),
    ("The Harvest Cakes - Bekasi",                         "Jakarta",     "Bekasi Utara",     "standalone_ruko",  -6.2274402, 107.0039993, 4.5, 495),
    ("The Harvest Cakes - Pajajaran Bogor",                "Jakarta",     "Bogor Utara",      "standalone_ruko",  -6.5781793, 106.8074234, 4.4, 950),
    ("The Harvest Cakes - Grand Wisata",                   "Jakarta",     "Tambun Sel.",      "standalone_ruko",  -6.2815184, 107.0464786, 4.4, 327),
    ("The Harvest - Cibubur",                              "Jakarta",     "Gn. Putri",        "standalone_ruko",  -6.3882233, 106.9429229, 4.3, 887),
    ("The Harvest Cakes - Depok",                          "Jakarta",     "Beji",             "standalone_ruko",  -6.3777795, 106.8312243, 4.4, 1223),
    ("The Harvest Cakes - Tajur",                          "Jakarta",     "Bogor Tim.",       "standalone_ruko",  -6.6255419, 106.8213064, 4.5, 355),
    ("The Harvest Cakes - Sawangan",                       "Jakarta",     "Bojongsari",       "standalone_ruko",  -6.4004536, 106.7421740, 4.4, 221),
    ("The Harvest Cakes - Jatiwaringin",                   "Jakarta",     "Pd. Gede",         "standalone_ruko",  -6.2714677, 106.9119073, 4.5, 575),
    ("The Harvest Dago",                                   "Bandung",     "Bandung Wetan",    "standalone_ruko",  -6.9053289, 107.6102687, 4.5, 1627),
    ("The Harvest Cakes - Buah Batu",                      "Bandung",     "Lengkong",         "standalone_ruko",  -6.9450102, 107.6300887, 4.5, 986),
    ("The Harvest Cakes - Setiabudi Bandung",               "Bandung",     "Sukasari",         "standalone_ruko",  -6.8659269, 107.5938968, 4.7, 139),
    ("The Harvest Cakes - Burangrang",                     "Bandung",     "Lengkong",         "standalone_ruko",  -6.9259490, 107.6198315, 4.7, 498),
    ("The Harvest Cakes - Bengawan",                       "Surabaya",    "Wonokromo",        "standalone_ruko",  -7.2901214, 112.7377761, 4.5, 2584),
    ("The Harvest Cakes - Dharmahusada",                   "Surabaya",    "Mulyorejo",        "standalone_ruko",  -7.2722908, 112.7821872, 4.6, 281),
    ("The Harvest Cakes - Graha Famili",                   "Surabaya",    "Dukuhpakis",       "standalone_ruko",  -7.2920622, 112.6764679, 4.3, 332),
    ("The Harvest Semarang",                               "Semarang",    "Gajahmungkur",     "standalone_ruko",  -6.9961751, 110.4077057, 4.5, 1288),
    ("The Harvest Cakes - Semarang Majapahit",              "Semarang",    "Pedurungan",       "standalone_ruko",  -7.0064535, 110.4558510, 4.3, 230),
    ("The Harvest Cakes - Yogyakarta",                     "Yogyakarta",  "Gondokusuman",     "standalone_ruko",  -7.7827950, 110.3721849, 4.3, 797),
    ("The Harvest Cakes - Malang Soekarno Hatta",          "Malang",      "Lowokwaru",        "standalone_ruko",  -7.9393431, 112.6252518, 4.6, 1287),
    ("The Harvest Cakes - Malang JA Suprapto",             "Malang",      "Klojen",           "standalone_ruko",  -7.9663944, 112.6341350, 4.2, 184),
]

def generate_outlets():
    rows = []

    for i, (outlet_name, city, district, location_type, lat, lon, rating, review_count) in enumerate(REAL_OUTLETS_MARUGAME, start=1):
        outlet_type = OUTLET_TYPE_OVERRIDES.get(outlet_name, assign_outlet_type(location_type, review_count))
        store_tier = assign_store_tier(outlet_type, review_count)
        rows.append({
            "outlet_id": f"MRG-{i:03d}",
            "brand": "Marugame Udon",
            "outlet_name": outlet_name,
            "city": city,
            "district": district,
            "location_type": location_type,
            "outlet_type": outlet_type,
            "latitude": lat,
            "longitude": lon,
            "rating": rating,
            "review_count": review_count,
            "gross_floor_area_sqm": random.randint(120, 320),
            "seating_capacity": random.randint(40, 110),
            "store_tier": store_tier,
            "open_date": (datetime(2018, 1, 1) + timedelta(days=random.randint(0, 2800))).date().isoformat(),
        })

    for i, (outlet_name, city, district, location_type, lat, lon, rating, review_count) in enumerate(REAL_OUTLETS_HARVEST, start=1):
        outlet_type = OUTLET_TYPE_OVERRIDES.get(outlet_name, assign_outlet_type(location_type, review_count))
        store_tier = assign_store_tier(outlet_type, review_count)
        rows.append({
            "outlet_id": f"HVT-{i:03d}",
            "brand": "The Harvest",
            "outlet_name": outlet_name,
            "city": city,
            "district": district,
            "location_type": location_type,
            "outlet_type": outlet_type,
            "latitude": lat,
            "longitude": lon,
            "rating": rating,
            "review_count": review_count,
            "gross_floor_area_sqm": random.randint(35, 90) if location_type == "mall" else random.randint(60, 180),
            "seating_capacity": None,
            "store_tier": store_tier,
            "open_date": (datetime(2015, 1, 1) + timedelta(days=random.randint(0, 4100))).date().isoformat(),
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 2. DEMOGRAPHICS GRID (synthetic 500m cells around each outlet's metro area)
# ---------------------------------------------------------------------------
def generate_demographics_grid(outlets_df, cell_km=0.5, base_cells=140):
    """base_cells is scaled per city by CITIES[city]['weight'] (relative to
    Jakarta's 0.50) so a 6-city model doesn't just multiply grid size by 2x —
    smaller metros get proportionally fewer cells, with a floor so whitespace
    analysis still has enough resolution to be meaningful."""
    rows = []
    grid_id = 1
    for city, c in CITIES.items():
        cells_per_city = max(60, round(base_cells * c["weight"] / 0.50))
        for _ in range(cells_per_city):
            lat, lon = random_point_in_city(city)
            district = random.choice(DISTRICTS[city])
            land_use = random.choices(
                ["residential", "commercial", "mixed", "office"],
                weights=[0.45, 0.2, 0.25, 0.1]
            )[0]

            base_pop = {
                "residential": random.randint(1800, 5200),
                "mixed": random.randint(1200, 3600),
                "commercial": random.randint(300, 1400),
                "office": random.randint(150, 900),
            }[land_use]

            income_bracket = random.choices(
                ["low", "lower_middle", "middle", "upper_middle", "high"],
                weights=[0.15, 0.25, 0.32, 0.20, 0.08]
            )[0]

            # age distribution shifts slightly by land use
            if land_use == "residential":
                age_young, age_adult, age_mid, age_senior = 0.24, 0.30, 0.32, 0.14
            elif land_use == "office":
                age_young, age_adult, age_mid, age_senior = 0.10, 0.45, 0.38, 0.07
            else:
                age_young, age_adult, age_mid, age_senior = 0.20, 0.35, 0.33, 0.12

            noise = lambda: random.uniform(-0.03, 0.03)
            rows.append({
                "grid_id": f"GRD-{grid_id:04d}",
                "city": city,
                "district": district,
                "centroid_lat": round(lat, 6),
                "centroid_lon": round(lon, 6),
                "cell_size_km": cell_km,
                "land_use_type": land_use,
                "population_est": base_pop,
                "household_count_est": round(base_pop / random.uniform(3.2, 4.5)),
                "income_bracket": income_bracket,
                "age_0_14_pct": round(max(age_young + noise(), 0.05), 3),
                "age_15_34_pct": round(max(age_adult + noise(), 0.05), 3),
                "age_35_54_pct": round(max(age_mid + noise(), 0.05), 3),
                "age_55plus_pct": round(max(age_senior + noise(), 0.02), 3),
            })
            grid_id += 1
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 3. COMPETITOR & COMPLEMENTARY POIs
# ---------------------------------------------------------------------------
def generate_competitor_pois(n=90):
    poi_types = {
        "competitor_noodle_asian": ["Marutama Ramen", "Hakata Ikkousha", "Bakmi GM", "Mie Gacoan"],
        "competitor_bakery_cake": ["Holland Bakery", "BreadTalk", "Union Bakery", "Sarinah Cake"],
        "mall": ["Mall X", "Plaza Y", "Trade Center Z"],
        "office_building": ["Menara A", "Tower B", "Corporate Park C"],
        "school_campus": ["Sekolah D", "Universitas E"],
        "transit_station": ["MRT Stasiun F", "KRL Stasiun G", "Halte Transjakarta H"],
        "residential_complex": ["Perumahan I", "Apartemen J", "Cluster K"],
    }
    rows = []
    poi_id = 1
    for _ in range(n):
        city = pick_city()
        lat, lon = random_point_in_city(city)
        ptype = random.choice(list(poi_types.keys()))
        name = f"{random.choice(poi_types[ptype])} {random.randint(1,20)}"
        rows.append({
            "poi_id": f"POI-{poi_id:04d}",
            "poi_type": ptype,
            "name": name,
            "city": city,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
        })
        poi_id += 1
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 4. FOOT TRAFFIC / TRANSACTIONS (90-day daypart-level simulation)
# ---------------------------------------------------------------------------
DAYPARTS = ["breakfast", "lunch", "afternoon", "dinner", "late_night"]

def daypart_profile(brand):
    """Relative weight of each daypart, brand-specific behaviour."""
    if brand == "Marugame Udon":
        return {"breakfast": 0.05, "lunch": 0.38, "afternoon": 0.12, "dinner": 0.35, "late_night": 0.10}
    else:  # The Harvest — steadier all day, occasion spikes
        return {"breakfast": 0.15, "lunch": 0.20, "afternoon": 0.25, "dinner": 0.25, "late_night": 0.15}

def generate_foot_traffic(outlets_df, n_days=90, start_date="2026-06-01"):
    rows = []
    start = datetime.fromisoformat(start_date)
    for _, o in outlets_df.iterrows():
        brand = o["brand"]
        profile = daypart_profile(brand)
        base_daily_visits = (
            random.randint(650, 1400) if brand == "Marugame Udon" and o["location_type"] == "mall"
            else random.randint(150, 500)
        )
        avg_basket = random.randint(55000, 95000) if brand == "Marugame Udon" else random.randint(80000, 260000)

        for d in range(n_days):
            date = start + timedelta(days=d)
            dow = date.strftime("%A")
            weekend_mult = 1.35 if dow in ("Saturday", "Sunday") else 1.0
            # simulate occasional "event day" spike for The Harvest (birthdays/holidays)
            event_mult = 1.0
            if brand == "The Harvest" and random.random() < 0.04:
                event_mult = random.uniform(1.6, 2.4)

            day_visits_total = int(base_daily_visits * weekend_mult * event_mult * random.uniform(0.85, 1.15))

            for part, weight in profile.items():
                visits = max(int(day_visits_total * weight * random.uniform(0.8, 1.2)), 0)
                conversion = random.uniform(0.55, 0.85)  # visits -> transactions
                transactions = int(visits * conversion)
                basket = int(avg_basket * random.uniform(0.85, 1.2))
                rows.append({
                    "outlet_id": o["outlet_id"],
                    "brand": brand,
                    "date": date.date().isoformat(),
                    "day_of_week": dow,
                    "daypart": part,
                    "estimated_visits": visits,
                    "estimated_transactions": transactions,
                    "avg_basket_size_idr": basket,
                    "estimated_revenue_idr": transactions * basket,
                })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 5. CUSTOMER ORIGIN SAMPLE (per-customer home -> outlet trips)
# ---------------------------------------------------------------------------
# Distance-from-home distribution parameters per outlet_type. These stand in
# for what would, with real data, be derived from loyalty/POS user_id + home
# address (or telco/GPS panel) — here we assume outlet_type causes a
# characteristic catchment shape rather than inferring it, since there is no
# real observed behaviour yet.
ORIGIN_TYPE_PARAMS = {
    # (lognormal mean_km, sigma, corridor_anisotropy 0=circular..1=very elongated,
    #  n_unique_customers baseline)
    "destination_hub":  {"mean_km": 6.0, "sigma": 0.9, "anisotropy": 0.3, "n_customers": 260},
    "transit_adjacent": {"mean_km": 3.2, "sigma": 0.7, "anisotropy": 0.7, "n_customers": 160},
    "neighborhood":      {"mean_km": 1.3, "sigma": 0.5, "anisotropy": 0.1, "n_customers": 110},
}

def generate_customer_origin_sample(outlets_df, demographics_df, n_days=90, start_date="2026-06-01"):
    rows = []
    start = datetime.fromisoformat(start_date)
    user_seq = 1

    for _, o in outlets_df.iterrows():
        params = ORIGIN_TYPE_PARAMS[o["outlet_type"]]
        n_customers = int(params["n_customers"] * random.uniform(0.8, 1.2))
        # size scales a bit with brand's typical volume (Marugame busier than Harvest boutique)
        if o["brand"] == "Marugame Udon":
            n_customers = int(n_customers * 1.15)

        city_grid = demographics_df[demographics_df["city"] == o["city"]]
        corridor_bearing = random.uniform(0, 360)  # fixed "commute axis" for this outlet

        for _ in range(n_customers):
            user_id = f"USR-{user_seq:06d}"
            user_seq += 1

            distance_km = float(np.random.lognormal(mean=math.log(params["mean_km"]), sigma=params["sigma"]))
            distance_km = min(distance_km, 40.0)  # clip unrealistic tail

            # anisotropy biases bearing toward the outlet's corridor axis instead of uniform circle
            if random.random() < params["anisotropy"]:
                bearing = corridor_bearing + random.uniform(-25, 25)
            else:
                bearing = random.uniform(0, 360)

            home_lat, home_lon = point_at_distance_bearing(o["latitude"], o["longitude"], distance_km, bearing)

            # snap to nearest existing demographic grid cell in the same city (proxy for a real home_grid_id)
            dists = city_grid.apply(
                lambda g: haversine_km(home_lat, home_lon, g["centroid_lat"], g["centroid_lon"]), axis=1
            )
            home_grid_id = city_grid.loc[dists.idxmin(), "grid_id"]

            # repeat-customer behaviour: most customers visit once or twice in 90 days,
            # a smaller loyal tail visits much more often (esp. neighborhood/proximity-driven)
            loyalty_boost = 1.6 if o["outlet_type"] == "neighborhood" else 1.0
            n_visits = min(int(np.random.geometric(p=0.55 / loyalty_boost)), 12)

            for _ in range(n_visits):
                visit_day = start + timedelta(days=random.randint(0, n_days - 1))
                daypart = random.choices(
                    list(daypart_profile(o["brand"]).keys()),
                    weights=list(daypart_profile(o["brand"]).values()),
                )[0]
                rows.append({
                    "user_id": user_id,
                    "outlet_id": o["outlet_id"],
                    "brand": o["brand"],
                    "outlet_type": o["outlet_type"],
                    "visit_date": visit_day.date().isoformat(),
                    "daypart": daypart,
                    "home_grid_id": home_grid_id,
                    "home_lat": round(home_lat, 6),
                    "home_lon": round(home_lon, 6),
                    "distance_km": round(distance_km, 2),
                })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    outlets = generate_outlets()
    demographics = generate_demographics_grid(outlets)
    competitors = generate_competitor_pois()
    foot_traffic = generate_foot_traffic(outlets)
    customer_origin = generate_customer_origin_sample(outlets, demographics)

    outlets.to_csv(os.path.join(OUT_DIR, "outlets.csv"), index=False)
    demographics.to_csv(os.path.join(OUT_DIR, "demographics_grid.csv"), index=False)
    competitors.to_csv(os.path.join(OUT_DIR, "competitor_pois.csv"), index=False)
    foot_traffic.to_csv(os.path.join(OUT_DIR, "foot_traffic_transactions.csv"), index=False)
    customer_origin.to_csv(os.path.join(OUT_DIR, "customer_origin_sample.csv"), index=False)

    print("Generated:")
    print(f"  outlets.csv                  : {len(outlets)} rows")
    print(f"  demographics_grid.csv        : {len(demographics)} rows")
    print(f"  competitor_pois.csv          : {len(competitors)} rows")
    print(f"  foot_traffic_transactions.csv: {len(foot_traffic)} rows")
    print(f"  customer_origin_sample.csv   : {len(customer_origin)} rows")
