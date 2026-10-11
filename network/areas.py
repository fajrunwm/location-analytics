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
}

# Areas that together cover all 72 outlets without double counting.
METROS = ["jabodetabek", "bandung", "surabaya", "semarang", "yogyakarta", "malang"]

# Centrality radii (metres of network distance).
WALK_DISTANCES = [400, 800, 1200, 1600, 2400, 3200, 4800]
DRIVE_DISTANCES = [1000, 2000, 5000, 10000, 20000]

# UTM zones for Java: 48S west of 108E, 49S east of it.
def utm_epsg(bbox):
    return 32748 if (bbox[0] + bbox[2]) / 2 < 108 else 32749
