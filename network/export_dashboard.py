"""Stage 5 — bake network results into a standalone NETWORK_DASHBOARD.html.

Reads <data>/<area>/hex_features.parquet, <data>/<area>/outlet_features.parquet and the
train_model.py outputs (hex_predictions, metrics, feature_importance, outlet_correlations),
and injects them as JSON into network/dashboard_template.html.
"""
import argparse
import json
import os
import re
from pathlib import Path

# Must be set before numpy loads: export needs no BLAS threads, and OpenBLAS's per-thread
# buffers can fail to allocate on a laptop that is short on memory.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import h3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from areas import AREAS

# Map layers: key -> (label, column candidates in hex table, scale kind, description)
LAYERS = [
    ("actual", "Hotspot aktual", ["y_commercial"], "seq",
     "Jumlah POI komersial (F&B, retail, jasa, hotel, hiburan; Overture, confidence ≥ 0,5) per hex"),
    ("potential", "Potensi jaringan", ["y_commercial_pred_log"], "seq",
     "Prediksi model (struktur jalan + populasi + transit) — dari spatial CV, bukan data latih"),
    ("gap", "Gap potensi", ["y_commercial_gap_log"], "div",
     "Biru = lokasi strategis tapi usahanya masih sedikit (peluang); merah = ramai karena sengaja dituju, "
     "seperti mall atau kawasan wisata (destinasi). Makin pekat, makin besar selisihnya."),
    ("confident", "Peluang & destinasi (stabil)", ["gap_confident_code"], "cat",
     "Biru = peluang: lokasi strategis tapi usahanya masih sedikit. "
     "Merah = destinasi: ramai karena sengaja dituju (mall, wisata, kantor); hanya hex dengan hasil konsisten yang diwarnai."),
    ("ang2400", "Angular betweenness 2,4 km", ["ang_betweenness_2400_ang_mean"], "seq",
     "Seberapa sering segmen jalan dilewati rute paling 'lurus' (sedikit belokan) dalam radius 2,4 km jalan kaki"),
    ("drive10k", "Koridor kendaraan 10 km", ["drive_betweenness_10000_mean"], "seq",
     "Betweenness jaringan kendaraan radius 10 km — koridor arteri pergerakan lintas kawasan"),
    ("close800", "Integrasi lokal 800 m", ["walk_harmonic_800_mean"], "seq",
     "Harmonic closeness jalan kaki 800 m — seberapa 'terhubung' kawasan dalam jarak 10 menit jalan"),
    ("pop1200", "Penduduk terjangkau 1,2 km", ["ctx_pop_sum_1200_nw", "ctx_pop_sum_1200_wt"], "seq",
     "Penduduk (WorldPop 2025) dalam 1,2 km jarak jalan kaki"),
]


def smooth(series):
    """Mean over each hex and its 6 neighbours — per-hex residuals are noisy; areas are the signal."""
    lookup = series.to_dict()
    out = []
    for h in series.index:
        vals = [lookup[n] for n in h3.grid_disk(h, 1) if n in lookup and pd.notna(lookup[n])]
        out.append(np.mean(vals) if vals else np.nan)
    return pd.Series(out, index=series.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--areas", required=True)
    ap.add_argument("--template", default=str(Path(__file__).with_name("dashboard_template.html")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", default=None, help="folder with hex_bootstrap.parquet (optional layer)")
    ap.add_argument("--huff", default=None, help="huff-network output folder (<area>/huff_outlets.csv), optional")
    a = ap.parse_args()

    model = Path(a.model)
    preds = pd.read_parquet(model / "hex_predictions.parquet")
    metrics = json.loads((model / "metrics.json").read_text())
    gnn_path = model / "gnn_metrics.json"
    if gnn_path.exists():
        metrics.update(json.loads(gnn_path.read_text()))
    imp = pd.read_csv(model / "feature_importance.csv", index_col=0).iloc[:, 0]
    corr = pd.read_csv(model / "outlet_correlations.csv") if (model / "outlet_correlations.csv").exists() else None
    huff = {}
    if a.huff:
        for f in Path(a.huff).glob("*/huff_outlets.csv"):
            for r in pd.read_csv(f).itertuples():
                huff[r.outlet_id] = {"huff_change_pct": r.change_pct, "drive_min": r.mean_drive_min_captured}
    boot = None
    if a.bootstrap and (Path(a.bootstrap) / "hex_bootstrap.parquet").exists():
        boot = pd.read_parquet(Path(a.bootstrap) / "hex_bootstrap.parquet", columns=["gap_confident"])
        boot["gap_confident_code"] = boot.gap_confident.map({"opportunity": 1.0, "destination": -1.0})

    areas_out, layers_meta = {}, []
    for key, label, cands, kind, desc in LAYERS:
        if key == "confident" and boot is None:
            continue
        layers_meta.append({"key": key, "label": label, "kind": kind, "desc": desc})

    for ar in a.areas.split(","):
        # Read only the columns the page needs (hex tables are ~576 columns wide).
        hp = Path(a.data) / ar / "hex_features.parquet"
        avail = set(pq.read_schema(hp).names)
        need = {"h3", "walk_node_count", "y_commercial"} | {c for _, _, cands, _, _ in LAYERS for c in cands}
        hexes = pd.read_parquet(hp, columns=sorted(need & avail))
        if "h3" in hexes:
            hexes = hexes.set_index("h3")
        hexes = hexes[hexes.walk_node_count >= 3]
        hexes = hexes.join(preds[[c for c in preds.columns if c.endswith(("_pred_log", "_gap_log"))]], how="left")
        if boot is not None:
            hexes = hexes.join(boot[["gap_confident_code"]], how="left")
        vals = {}
        for key, _, cands, _, _ in LAYERS:
            col = next((c for c in cands if c in hexes), None)
            if col is None:
                continue
            v = hexes[col].astype(float)
            if key == "potential":
                v = np.expm1(v)  # back to POI-count scale
            if key == "gap":
                v = smooth(v)
            vals[key] = [None if not np.isfinite(x) else round(float(x), 3) for x in v]

        outlets = []
        op = Path(a.data) / ar / "outlet_features.parquet"
        if op.exists():
            of = pd.read_parquet(op)
            for oid, r in of.iterrows():
                h = r.get("h3")
                row = {"id": oid, "name": r["outlet_name"], "brand": r["brand"], "lat": r["latitude"],
                       "lon": r["longitude"], "reviews": int(r["review_count"]), "rating": float(r["rating"]),
                       "location_type": r.get("location_type"), "outlet_type": r.get("outlet_type")}
                for key, col in (("potential", "y_commercial_pred_log"), ("ang2400", "ang_betweenness_2400_ang_mean"),
                                 ("drive10k", "drive_betweenness_10000_mean"), ("actual", "y_commercial")):
                    if col in hexes and h in hexes.index:
                        row[f"{key}_pct"] = float((hexes[col] <= hexes.at[h, col]).mean().round(3) * 100)
                if "gap_confident_code" in hexes and h in hexes.index:
                    c = hexes.at[h, "gap_confident_code"]
                    row["confident"] = {1.0: "peluang", -1.0: "destinasi"}.get(c, "")
                row.update({k: (None if pd.isna(v) else float(v)) for k, v in huff.get(oid, {}).items()})
                if "y_commercial_gap_log" in hexes and h in hexes.index:
                    g = hexes.at[h, "y_commercial_gap_log"]
                    row["gap"] = None if pd.isna(g) else round(float(g), 3)
                outlets.append(row)

        w, s, e, n = AREAS[ar]["bbox"]
        areas_out[ar] = {"label": AREAS[ar]["label"], "bbox": [w, s, e, n], "h3": list(hexes.index),
                         "values": vals, "outlets": outlets}

    # Group SHAP importance into readable families (all radii / stats / aggregations summed).
    def family(c):
        src = c.split("_", 1)[0]
        if src == "ctx":
            return "Transit (halte/stasiun)" if c.startswith("ctx_n_") else "Populasi terjangkau"
        if c == "walk_node_count":
            return "Kepadatan simpang"
        net = {"walk": "Jalan kaki", "drive": "Kendaraan", "ang": "Angular (jalan kaki)"}[src]
        if "nk_betweenness" in c:
            return f"{net}: betweenness global"
        if "nk_closeness" in c:
            return f"{net}: closeness global"
        for key, name in (("betweenness", "betweenness lokal"), ("harmonic", "closeness lokal"), ("hillier", "closeness lokal"),
                          ("farness", "closeness lokal"), ("beta", "closeness lokal"), ("density", "jangkauan jaringan"),
                          ("cycles", "jangkauan jaringan")):
            if f"_{key}_" in c or c.startswith(f"{src}_{key}"):
                return f"{net}: {name}"
        return c
    fam = imp.groupby(imp.index.map(family)).sum().sort_values(ascending=False)
    payload = {
        "layers": layers_meta,
        "areas": areas_out,
        "metrics": metrics,
        "importance": [{"feature": k, "value": round(float(v), 4)} for k, v in fam.head(15).items()],
        "outlet_corr": [] if corr is None else corr.head(12).round(4).replace({np.nan: None}).to_dict("records"),
    }
    def clean(o):  # NaN (e.g. too few dense-urban hexes) -> null for strict JSON
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return None if isinstance(o, float) and not np.isfinite(o) else o
    payload = clean(payload)
    html = Path(a.template).read_text(encoding="utf-8")
    html = html.replace("/*__NETWORK_DATA__*/null", json.dumps(payload, separators=(",", ":"), allow_nan=False))
    Path(a.out).write_text(html, encoding="utf-8")
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB), areas={list(areas_out)}")


if __name__ == "__main__":
    main()
