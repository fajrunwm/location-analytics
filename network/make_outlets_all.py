"""Build network/outlets_all.csv = outlets.csv (72 Java outlets) + 26 real outlets outside Java.

The non-Java rows come from the raw Google Maps scrapes (marugame_outlets_indonesia.csv /
harvest_outlets_indonesia.csv) that generate_dummy_data.py filters out. Only real fields are
kept (name, coordinates, rating, review count); location_type uses the same idea as the Java
set — mall keywords in the name/address — plus "airport" for the Makassar airport outlet.
Synthetic attributes (floor area, tier, open date) are left empty on purpose.
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MALL = re.compile(r"\b(mall|mal|plaza|sun plaza|centre point|deli park|living world|galeria|e-walk|icon|trans studio)\b", re.I)


def main():
    java = pd.read_csv(ROOT / "outlets.csv")
    raw = pd.concat([pd.read_csv(ROOT / f).assign(brand=b) for f, b in
                     [("marugame_outlets_indonesia.csv", "Marugame Udon"), ("harvest_outlets_indonesia.csv", "The Harvest")]])
    on_java = raw.Latitude.between(-8.9, -5.8) & raw.Longitude.between(105.0, 114.7)
    ext = raw[~on_java].copy()

    def loc_type(r):
        text = f"{r['Nama Outlet']} {r['Alamat']}"
        if re.search(r"bandara|airport", text, re.I):
            return "airport"
        return "mall" if MALL.search(text) else "standalone_ruko"

    prefix = ext.brand.map({"Marugame Udon": "MRG-X", "The Harvest": "HVT-X"})
    ext["n"] = ext.groupby("brand").cumcount() + 1
    out = pd.DataFrame({
        "outlet_id": prefix + ext.n.astype(str).str.zfill(2),
        "brand": ext.brand,
        "outlet_name": ext["Nama Outlet"],
        "city": ext["Kota/Wilayah"].str.replace(", Bali", "", regex=False),
        "location_type": ext.apply(loc_type, axis=1),
        "latitude": ext.Latitude,
        "longitude": ext.Longitude,
        "rating": ext.Rating,
        "review_count": ext["Jumlah Ulasan"],
    })
    allo = pd.concat([java, out], ignore_index=True)
    allo.to_csv(Path(__file__).with_name("outlets_all.csv"), index=False)
    print(out.groupby(["city", "location_type"]).size().to_string())
    print(f"wrote outlets_all.csv: {len(java)} Java + {len(out)} outside Java")


if __name__ == "__main__":
    main()
