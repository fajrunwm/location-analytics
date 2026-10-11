"""Stage 1 — fetch raw, REAL inputs for one study area.

Outputs (in <out>/<area>/raw/):
  walk.graphml, drive.graphml   OSM street networks (OSMnx, simplified, unprojected)
  transit.parquet               OSM rail/MRT/LRT/KRL stations, BRT & bus stops
  places.parquet                Overture Maps places (POIs) with top-level category
  population.parquet            WorldPop 2025 constrained 100m cells (lon, lat, pop)

Two OSM sources:
  --osm-source pbf       Geofabrik Java extract + osmium-tool (Azure ML; no rate limits)
  --osm-source overpass  OSMnx Overpass queries (local dev on small areas)
"""
import argparse
import subprocess
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import overturemaps.core as overture
import pandas as pd
import rasterio
import requests
from rasterio.windows import from_bounds

from areas import AREAS

PBF_URL = "https://download.geofabrik.de/asia/indonesia/{region}-latest.osm.pbf"
WORLDPOP_URL = ("https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/IDN/v1/100m/"
                "constrained/idn_pop_2025_CN_100m_R2025A_v1.tif")

WALK_HIGHWAYS = ["trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link",
                 "tertiary", "tertiary_link", "unclassified", "residential", "living_street",
                 "pedestrian", "footway", "path", "steps", "service", "track", "cycleway"]
DRIVE_HIGHWAYS = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                  "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
                  "residential", "living_street"]
TRANSIT_TAGS = {"railway": ["station", "halt", "tram_stop"], "public_transport": ["station"],
                "amenity": ["bus_station"], "highway": ["bus_stop"]}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    log(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def osmium(*args):
    subprocess.run(["osmium", *args, "--overwrite"], check=True)


def osm_from_pbf(bbox, raw, cache, region="java"):
    pbf = download(PBF_URL.format(region=region), cache / f"{region}-latest.osm.pbf")
    area_pbf = raw / "area.osm.pbf"
    osmium("extract", "-b", ",".join(map(str, bbox)), "--strategy=complete_ways", str(pbf), "-o", str(area_pbf))

    for name, highways in [("walk", WALK_HIGHWAYS), ("drive", DRIVE_HIGHWAYS)]:
        xml = raw / f"{name}.osm"
        osmium("tags-filter", str(area_pbf), "w/highway=" + ",".join(highways), "-o", str(xml))
        log(f"building {name} graph from {xml.stat().st_size / 1e6:.0f} MB XML")
        G = ox.graph_from_xml(xml, bidirectional=(name == "walk"), simplify=True, retain_all=False)
        G = ox.truncate.largest_component(ox.truncate.truncate_graph_bbox(G, bbox))
        ox.save_graphml(G, raw / f"{name}.graphml")
        log(f"{name}: {len(G):,} nodes, {G.number_of_edges():,} edges")
        xml.unlink()

    expr = [f"nwr/{k}={','.join(v)}" for k, v in TRANSIT_TAGS.items()]
    osmium("tags-filter", str(area_pbf), *expr, "-o", str(raw / "transit.osm.pbf"))
    osmium("export", str(raw / "transit.osm.pbf"), "-f", "geojsonseq", "-o", str(raw / "transit.geojsonseq"))
    transit = gpd.read_file(raw / "transit.geojsonseq")
    area_pbf.unlink()
    return transit


def osm_from_overpass(bbox, raw):
    for name in ["walk", "drive"]:
        G = ox.graph_from_bbox(bbox, network_type=name, simplify=True, retain_all=False)
        ox.save_graphml(G, raw / f"{name}.graphml")
        log(f"{name}: {len(G):,} nodes, {G.number_of_edges():,} edges")
    return ox.features_from_bbox(bbox, TRANSIT_TAGS).reset_index()


def classify_transit(df):
    def kind(r):
        if r.get("railway") in ("station", "halt", "tram_stop") or (
                r.get("public_transport") == "station" and r.get("bus") != "yes" and r.get("highway") != "bus_stop"):
            return "rail_station"
        if r.get("amenity") == "bus_station":
            return "bus_station"
        return "bus_stop"
    df = df.copy()
    df["transit_type"] = df.apply(kind, axis=1)
    df["geometry"] = df.geometry.representative_point()
    keep = [c for c in ["name", "transit_type", "railway", "station", "network", "geometry"] if c in df]
    return gpd.GeoDataFrame(df[keep], geometry="geometry", crs=4326)


def fetch_places(bbox):
    g = overture.geodataframe("place", bbox=bbox)
    tax = g["taxonomy"]
    out = gpd.GeoDataFrame({
        "place_id": g["id"],
        "name": g["names"].map(lambda n: n.get("primary") if isinstance(n, dict) else None),
        "category_top": tax.map(lambda t: t["hierarchy"][0] if isinstance(t, dict) and t.get("hierarchy") is not None and len(t["hierarchy"]) else "unknown"),
        "category": tax.map(lambda t: t.get("primary") if isinstance(t, dict) else None),
        "basic_category": g["basic_category"],
        "brand": g["brand"].map(lambda b: (b.get("names") or {}).get("primary") if isinstance(b, dict) else None),
        "confidence": g["confidence"].astype(float),
    }, geometry=g.geometry, crs=4326)
    return out


def fetch_population(bbox, cache):
    tif = download(WORLDPOP_URL, cache / "idn_pop_2025_100m.tif")
    with rasterio.open(tif) as src:
        win = from_bounds(*bbox, transform=src.transform)
        arr = src.read(1, window=win, boundless=True, fill_value=0).astype("float64")
        tr = src.window_transform(win)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = 0
    arr[~np.isfinite(arr) | (arr < 0)] = 0
    rows, cols = np.nonzero(arr)
    xs, ys = rasterio.transform.xy(tr, rows, cols)
    return pd.DataFrame({"lon": xs, "lat": ys, "pop": arr[rows, cols]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", required=True, choices=AREAS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="dir for the Java PBF / WorldPop raster (shared across areas)")
    ap.add_argument("--osm-source", choices=["pbf", "overpass"], default="pbf")
    a = ap.parse_args()

    bbox = AREAS[a.area]["bbox"]
    raw = Path(a.out) / a.area / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache or Path(a.out) / "_cache")
    cache.mkdir(parents=True, exist_ok=True)

    log(f"area={a.area} bbox={bbox}")
    transit = (osm_from_pbf(bbox, raw, cache, AREAS[a.area].get("pbf", "java")) if a.osm_source == "pbf"
               else osm_from_overpass(bbox, raw))
    transit = classify_transit(transit)
    transit.to_parquet(raw / "transit.parquet")
    log(f"transit: {transit.transit_type.value_counts().to_dict()}")

    places = fetch_places(bbox)
    places.to_parquet(raw / "places.parquet")
    log(f"places: {len(places):,}; top categories: {places.category_top.value_counts().head(8).to_dict()}")

    pop = fetch_population(bbox, cache)
    pop.to_parquet(raw / "population.parquet")
    log(f"population: {len(pop):,} cells, total {pop['pop'].sum():,.0f}")


if __name__ == "__main__":
    main()
