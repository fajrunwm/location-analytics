"""
Dummy Data Generator — F&B Location Analytics Tool
Brands: Marugame Udon (mall-based, fast-casual) & The Harvest (bakery, mixed mall/standalone)

Generates 4 linked datasets that mimic what would eventually come from the
brands' real POS / GIS / demographic vendors:

    1. outlets.csv                 - store locations & attributes
    2. demographics_grid.csv       - synthetic 500m grid population/income/age profile
    3. competitor_pois.csv         - competitor & complementary points of interest
    4. foot_traffic_transactions.csv - simulated daily/daypart visit & sales pattern

All coordinates are randomly jittered around REAL city centers (Jakarta, Bandung,
Surabaya) so the spatial distribution looks plausible, but no real store,
address, or person is represented. Safe to treat as fully synthetic.

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
CITIES = {
    "Jakarta":  {"lat": -6.2088, "lon": 106.8456, "radius_km": 18, "weight": 0.55},
    "Bandung":  {"lat": -6.9175, "lon": 107.6191, "radius_km": 10, "weight": 0.20},
    "Surabaya": {"lat": -7.2575, "lon": 112.7521, "radius_km": 12, "weight": 0.25},
}

DISTRICTS = {
    "Jakarta":  ["Jakarta Selatan", "Jakarta Pusat", "Jakarta Barat", "Jakarta Timur", "Jakarta Utara", "Tangerang", "Bekasi", "Depok"],
    "Bandung":  ["Bandung Wetan", "Coblong", "Sukajadi", "Cidadap", "Buah Batu"],
    "Surabaya": ["Gubeng", "Wonokromo", "Rungkut", "Tegalsari", "Sukolilo"],
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

# ---------------------------------------------------------------------------
# 1. OUTLETS
# ---------------------------------------------------------------------------
def generate_outlets(n_marugame=18, n_harvest=24):
    rows = []
    outlet_id = 1

    # Marugame Udon — overwhelmingly mall-based, fewer but larger footprints
    mall_names = ["Central Park", "Grand Indonesia", "Kota Kasablanka", "Senayan City",
                  "Pondok Indah Mall", "Summarecon Mall", "Living World", "Trans Studio Mall",
                  "Ciputra World", "Pakuwon Mall", "Tunjungan Plaza", "Paris Van Java",
                  "Cihampelas Walk", "AEON Mall", "Bintaro Xchange", "Gandaria City",
                  "PIK Avenue", "Lippo Mall Puri"]
    for i in range(n_marugame):
        city = pick_city()
        lat, lon = random_point_in_city(city)
        rows.append({
            "outlet_id": f"MRG-{outlet_id:03d}",
            "brand": "Marugame Udon",
            "outlet_name": f"Marugame Udon {mall_names[i % len(mall_names)]}",
            "city": city,
            "district": random.choice(DISTRICTS[city]),
            "location_type": "mall",
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "gross_floor_area_sqm": random.randint(120, 320),
            "seating_capacity": random.randint(40, 110),
            "store_tier": random.choices(["flagship", "standard"], weights=[0.2, 0.8])[0],
            "open_date": (datetime(2018, 1, 1) + timedelta(days=random.randint(0, 2800))).date().isoformat(),
        })
        outlet_id += 1

    # The Harvest — mix of mall counters and standalone/ruko boutiques
    for i in range(n_harvest):
        city = pick_city()
        lat, lon = random_point_in_city(city)
        loc_type = random.choices(["mall", "standalone_ruko"], weights=[0.4, 0.6])[0]
        rows.append({
            "outlet_id": f"HVT-{outlet_id:03d}",
            "brand": "The Harvest",
            "outlet_name": f"The Harvest {'Counter' if loc_type=='mall' else 'Boutique'} {i+1:02d}",
            "city": city,
            "district": random.choice(DISTRICTS[city]),
            "location_type": loc_type,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "gross_floor_area_sqm": random.randint(35, 90) if loc_type == "mall" else random.randint(60, 180),
            "seating_capacity": None,
            "store_tier": random.choices(["flagship", "standard"], weights=[0.15, 0.85])[0],
            "open_date": (datetime(2015, 1, 1) + timedelta(days=random.randint(0, 4100))).date().isoformat(),
        })
        outlet_id += 1

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 2. DEMOGRAPHICS GRID (synthetic 500m cells around each outlet's metro area)
# ---------------------------------------------------------------------------
def generate_demographics_grid(outlets_df, cell_km=0.5, cells_per_city=140):
    rows = []
    grid_id = 1
    for city, c in CITIES.items():
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
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    outlets = generate_outlets()
    demographics = generate_demographics_grid(outlets)
    competitors = generate_competitor_pois()
    foot_traffic = generate_foot_traffic(outlets)

    outlets.to_csv(os.path.join(OUT_DIR, "outlets.csv"), index=False)
    demographics.to_csv(os.path.join(OUT_DIR, "demographics_grid.csv"), index=False)
    competitors.to_csv(os.path.join(OUT_DIR, "competitor_pois.csv"), index=False)
    foot_traffic.to_csv(os.path.join(OUT_DIR, "foot_traffic_transactions.csv"), index=False)

    print("Generated:")
    print(f"  outlets.csv                 : {len(outlets)} rows")
    print(f"  demographics_grid.csv       : {len(demographics)} rows")
    print(f"  competitor_pois.csv         : {len(competitors)} rows")
    print(f"  foot_traffic_transactions.csv: {len(foot_traffic)} rows")
