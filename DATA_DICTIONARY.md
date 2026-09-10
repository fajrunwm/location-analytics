# Data Dictionary — F&B Location Analytics (Dummy Data)

Semua data di sini **sintetis**, di-generate dengan `generate_dummy_data.py`
(seed=42, reproducible). Titik koordinat di-jitter di sekitar pusat kota asli
(Jakarta, Bandung, Surabaya) agar distribusi spasial realistis, tapi tidak
merepresentasikan outlet, alamat, atau orang sungguhan.

Saat data asli dari brand sudah tersedia, tinggal mapping ke skema yang sama
— pipeline analitik & integrasi GenAI di tahap berikutnya tidak perlu berubah.

---

## 1. `outlets.csv` (42 baris)
Master data lokasi toko.

| Kolom | Tipe | Keterangan |
|---|---|---|
| outlet_id | string | ID unik (`MRG-xxx` / `HVT-xxx`) |
| brand | string | `Marugame Udon` / `The Harvest` |
| outlet_name | string | Nama outlet |
| city | string | Jakarta / Bandung / Surabaya |
| district | string | Kecamatan/area |
| location_type | string | `mall`, `standalone_ruko` |
| outlet_type | string | `destination_hub` / `transit_adjacent` / `neighborhood` — lihat rasional di bawah |
| latitude, longitude | float | Koordinat |
| gross_floor_area_sqm | int | Luas outlet |
| seating_capacity | int/null | Hanya relevan untuk Marugame (dine-in) |
| store_tier | string | `flagship` / `standard` |
| open_date | date | Tanggal buka |

**Rasional desain:** Marugame 100% `mall`-based (fast-casual dine-in, butuh
foot-traffic mall tinggi). The Harvest mix `mall` (counter kecil) vs
`standalone_ruko` (butik, occasion-based) — mencerminkan pola bisnis nyata
kedua brand.

**`outlet_type` — outlet bukan titik yang flat/seragam.** Dua outlet dengan
`location_type=mall` dan luas serupa bisa punya "power" (jangkauan tarik
customer) yang jauh berbeda tergantung peran mall-nya:
- `destination_hub`: mall besar yang jadi assembly point/meeting point (mis.
  Grand Indonesia, Senayan City, Central Park) atau outlet flagship — menarik
  customer dari radius jauh & catchment tidak simetris (dipengaruhi
  transit/corridor, bukan lingkaran rapi).
- `transit_adjacent`: dekat jalur commuter/transit — catchment memanjang
  mengikuti sumbu koridor perjalanan, bukan melebar ke segala arah.
- `neighborhood`: outlet lokal (ruko/counter kecil) — catchment sempit &
  didominasi proximity ke rumah/kantor sekitarnya.

`outlet_type` inilah yang menentukan bentuk & lebar catchment tiap outlet di
`customer_origin_sample.csv` (lihat §5) dan, di tahap analitik berikutnya,
akan dipakai untuk beda-kan attractiveness dan distance-decay (beta) per tipe
alih-alih satu parameter global untuk semua outlet.

## 2. `demographics_grid.csv` (420 baris)
Grid sintetis 500m menutupi area metro tiap kota — pengganti data sensus/BPS
riil untuk keperluan POC.

| Kolom | Keterangan |
|---|---|
| grid_id | ID grid cell |
| centroid_lat/lon | Titik tengah cell |
| land_use_type | `residential` / `commercial` / `mixed` / `office` — memengaruhi pola populasi & usia |
| population_est, household_count_est | Estimasi populasi |
| income_bracket | `low` → `high` (kategorikal, bukan nominal rupiah — by design, untuk hindari angka finansial presisi yang menyesatkan di data dummy) |
| age_0_14_pct ... age_55plus_pct | Distribusi usia, total ≈ 1.0 |

**Rasional:** land use memengaruhi profil usia & populasi secara berbeda
(area residential lebih muda & padat KK; office lebih didominasi usia kerja).

## 3. `competitor_pois.csv` (90 baris)
POI kompetitor & kontekstual di sekitar outlet.

| Kolom | Keterangan |
|---|---|
| poi_type | `competitor_noodle_asian`, `competitor_bakery_cake`, `mall`, `office_building`, `school_campus`, `transit_station`, `residential_complex` |
| name | Nama generik/sintetis (bukan brand asli, kecuali kategori kompetitor umum sbg placeholder) |

## 4. `foot_traffic_transactions.csv` (18,900 baris)
Simulasi harian x daypart selama 90 hari per outlet.

| Kolom | Keterangan |
|---|---|
| daypart | `breakfast`, `lunch`, `afternoon`, `dinner`, `late_night` |
| estimated_visits | Simulasi jumlah pengunjung |
| estimated_transactions | visits × conversion rate acak (0.55–0.85) |
| avg_basket_size_idr | Rata-rata belanja |
| estimated_revenue_idr | transactions × basket |

**Pola brand-specific yang disimulasikan:**
- **Marugame Udon**: puncak di `lunch` & `dinner` (0.38 & 0.35 dari total), weekend multiplier 1.35x
- **The Harvest**: lebih rata sepanjang hari, plus "event day" acak (~4% hari) dengan spike 1.6–2.4x (simulasi hampers/ulang tahun/hari raya)

## 5. `customer_origin_sample.csv` (~15,000 baris)
Sampel sintetis per-customer: dari mana pelanggan datang untuk mengunjungi
tiap outlet, dipakai untuk mengukur **origin dispersion** (seberapa jauh &
seberapa tersebar wilayah asal pelanggan tiap outlet) — proxy untuk "power"
outlet yang disebut di §1, dan pengganti sementara loyalty/POS `user_id` +
alamat rumah riil yang belum tersedia.

| Kolom | Keterangan |
|---|---|
| user_id | ID pelanggan sintetis (satu user bisa punya beberapa baris kunjungan — repeat customer) |
| outlet_id | Outlet yang dikunjungi |
| outlet_type | Disalin dari `outlets.csv` untuk kemudahan agregasi |
| visit_date, daypart | Sama seperti `foot_traffic_transactions.csv` |
| home_grid_id | Grid cell terdekat (dari `demographics_grid.csv`) ke lokasi rumah sintetis pelanggan |
| home_lat, home_lon | Lokasi rumah sintetis (hasil offset jarak+bearing dari outlet) |
| distance_km | Jarak lurus rumah → outlet |

**Rasional desain:** jarak & sebaran bearing digambar dari distribusi
lognormal yang parameternya berbeda per `outlet_type` (mean 6km untuk
`destination_hub`, 3.2km `transit_adjacent`, 1.3km `neighborhood`), dengan
`destination_hub` & `transit_adjacent` diberi anisotropy (bias ke satu arah
koridor) supaya catchment-nya tidak berbentuk lingkaran sempurna. Ini adalah
**asumsi**, bukan hasil observasi — sekali data user_id + alamat riil (POS/
loyalty) tersedia, kolom `distance_km`/`home_grid_id` harus dihitung dari data
asli, dan parameter dispersion per tipe di atas dikalibrasi ulang.

---

## Cara pakai ulang / extend
```bash
python3 generate_dummy_data.py
```
Ubah `n_marugame`, `n_harvest`, `n_days`, atau bobot di `daypart_profile()`
untuk skenario lain. Ganti isi `CITIES` dict untuk menambah kota lain.

## Batasan yang perlu diingat
- Ini **bukan** data riil — jangan dipakai untuk keputusan bisnis, hanya untuk
  membangun & menguji pipeline (schema, join logic, model analitik, dashboard).
- Saat data asli (POS, GIS outlet, data demografi vendor/BPS) tersedia, mapping
  ke kolom yang sama di atas → downstream code tidak perlu ditulis ulang.
