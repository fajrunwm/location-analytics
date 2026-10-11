"""Study areas for the street-network analysis.

Bboxes are (west, south, east, north) in WGS84. Metro boxes cover every outlet
in outlets.csv for that city plus a few km of surrounding street network, so
centrality near the edge of the box (edge effect) does not land on an outlet.
`city` links an area back to outlets.csv.
"""

AREAS = {
    # Fast iteration area — Jakarta Pusat (approximate admin extent).
    "jakpus": {"bbox": (106.790, -6.220, 106.880, -6.135), "city": "Jakarta", "label": "Jakarta Pusat"},
    # DKI Jakarta mainland (excludes Kepulauan Seribu).
    "dki": {"bbox": (106.685, -6.375, 106.975, -6.085), "city": "Jakarta", "label": "DKI Jakarta"},
    # Jabodetabek — matches the "Jakarta" metro in outlets.csv (Bogor/Depok/Tangerang/Bekasi).
    "jabodetabek": {"bbox": (106.450, -6.680, 107.150, -6.040), "city": "Jakarta", "label": "Jabodetabek"},
    "bandung": {"bbox": (107.500, -7.050, 107.780, -6.800), "city": "Bandung", "label": "Bandung Raya"},
    "surabaya": {"bbox": (112.580, -7.420, 112.860, -7.150), "city": "Surabaya", "label": "Surabaya"},
    "semarang": {"bbox": (110.330, -7.100, 110.520, -6.920), "city": "Semarang", "label": "Semarang"},
    "yogyakarta": {"bbox": (110.300, -7.880, 110.470, -7.690), "city": "Yogyakarta", "label": "Yogyakarta"},
    "malang": {"bbox": (112.560, -8.050, 112.700, -7.880), "city": "Malang", "label": "Malang"},
    # Outside Java — out-of-sample test for a model trained on Java (`pbf` = Geofabrik region).
    "medan": {"bbox": (98.550, 3.470, 98.780, 3.720), "city": "Medan", "label": "Medan", "pbf": "sumatra"},
    "palembang": {"bbox": (104.650, -3.080, 104.850, -2.880), "city": "Palembang", "label": "Palembang", "pbf": "sumatra"},
    "denpasar": {"bbox": (115.100, -8.800, 115.320, -8.560), "city": "Denpasar", "label": "Denpasar & Badung", "pbf": "nusa-tenggara"},
    "makassar": {"bbox": (119.360, -5.250, 119.600, -5.030), "city": "Makassar", "label": "Makassar", "pbf": "sulawesi"},
    "balikpapan": {"bbox": (116.800, -1.320, 116.950, -1.180), "city": "Balikpapan", "label": "Balikpapan", "pbf": "kalimantan"},
}

# Areas that together cover all 72 outlets without double counting.
METROS = ["jabodetabek", "bandung", "surabaya", "semarang", "yogyakarta", "malang"]
NON_JAVA = ["medan", "palembang", "denpasar", "makassar", "balikpapan"]

# Centrality radii (metres of network distance).
WALK_DISTANCES = [400, 800, 1200, 1600, 2400, 3200, 4800]
DRIVE_DISTANCES = [1000, 2000, 5000, 10000, 20000]

def utm_epsg(bbox):
    """WGS84 UTM zone of the bbox centre (Java: 48S west of 108E, 49S east; Medan is 47N)."""
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone
