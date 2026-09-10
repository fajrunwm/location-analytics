# Data Dictionary — F&B Location Analytics (Real Locations + Synthetic Data)

Data di sini **provenance campuran** — lihat §Provenance data lokasi di bawah
untuk mana yang real dan mana yang sintetis, jangan asumsikan semuanya
sintetis. Semua digenerate lewat `generate_dummy_data.py` (seed=42,
reproducible untuk bagian yang sintetis).

Saat data POS/GIS asli dari brand sudah tersedia (lengkap, bukan cuma yang
terindeks di Google Maps), tinggal mapping ke skema yang sama — pipeline
analitik & integrasi GenAI di tahap berikutnya tidak perlu berubah.

## Provenance data lokasi

`outlets.csv` kolom `outlet_name`, `city`, `district`, `location_type`,
`latitude`, `longitude`, `rating`, `review_count` adalah **REAL** — di-scrape
dari Google Maps (via Claude Desktop, 2026-09-10) untuk brand "Marugame
Udon" dan "The Harvest" di Indonesia. Sumber mentahnya, sebelum difilter,
ada di `marugame_outlets_indonesia.csv` (60 baris, seluruh Indonesia) dan
`harvest_outlets_indonesia.csv` (38 baris) — kolomnya: Nama Outlet,
Kota/Wilayah, Alamat, Latitude, Longitude, Rating, Jumlah Ulasan, Telepon,
Jam Operasional, Website, Google Maps URL. `generate_dummy_data.py`
memfilter kedua file ini ke **pulau Jawa saja** (lihat di bawah), lalu
mem-parsing `district` dari teks Alamat dan mengklasifikasi `location_type`
dari kata kunci mall pada nama/alamat. Kolom lain di `outlets.csv` (luas,
seating, tier, tanggal buka) tetap sintetis — Google Maps tidak menyimpan
data itu. `outlet_type` dan `store_tier` diturunkan dari `review_count` real
(lihat rasional §1 di bawah), bukan pure random. Seluruh file lain
(`demographics_grid.csv`, `competitor_pois.csv`, `foot_traffic_transactions.csv`,
`customer_origin_sample.csv`) 100% sintetis, tidak merepresentasikan lokasi,
transaksi, atau orang sungguhan.

**Cakupan & yang di-exclude:** data mentah Google Maps mencakup seluruh
Indonesia (Jabodetabek, Bandung, Surabaya, Semarang, Yogyakarta, Malang,
Medan, Makassar, Palembang, Denpasar/Bali, Balikpapan). Repo ini sengaja
**membatasi ke pulau Jawa** — 6 metro: Jakarta (termasuk Jabodetabek),
Bandung, Surabaya, Semarang, Yogyakarta, Malang — supaya scope geografis
tool tetap satu pulau yang koheren. Outlet di Medan, Makassar, Palembang,
Denpasar, dan Balikpapan **di-exclude**, bukan disembunyikan — datanya tetap
ada di kedua CSV mentah kalau suatu saat scope mau diperluas.

Hasilnya: **44 outlet Marugame Udon real + 28 outlet The Harvest real = 72
outlet** di Jawa, dengan sebaran (Marugame/Harvest per kota): Jakarta metro
20/16, Bandung 6/4, Surabaya 8/3, Semarang 4/2, Yogyakarta 4/1, Malang 2/2.
Jakarta metro paling padat (36 dari 72) — realistis untuk brand F&B rantai
yang basis ekspansinya dari Jabodetabek.

`location_type` (`mall`/`standalone_ruko`) diklasifikasi dari kata kunci di
teks nama+alamat real (mis. "mall", "plaza", "square", "trans studio",
"wtc") — bukan diverifikasi ke denah toko sungguhan, jadi anggap sebagai
heuristik dari data real, bukan fakta yang sudah tervalidasi 100%. `district`
di-parse dari struktur alamat (ambil segmen kecamatan kalau ada, kalau tidak
fallback ke nama kota administratif) — juga real, dengan presisi yang
bervariasi tergantung selengkap apa format alamat aslinya.

---

## 1. `outlets.csv` (72 baris — REAL lokasi & rating, sintetis atribut lain)
Master data lokasi toko.

| Kolom | Tipe | Provenance | Keterangan |
|---|---|---|---|
| outlet_id | string | derived | ID unik (`MRG-xxx` / `HVT-xxx`) |
| brand | string | real | `Marugame Udon` / `The Harvest` |
| outlet_name | string | **real** | Nama outlet, dari Google Maps |
| city | string | **real** | Jakarta / Bandung / Surabaya / Semarang / Yogyakarta / Malang (metro bucket) |
| district | string | **real** (parsed) | Kecamatan/area asli dari alamat Google Maps |
| location_type | string | **real** (heuristik) | `mall`, `standalone_ruko` |
| outlet_type | string | derived dari real | `destination_hub` / `transit_adjacent` / `neighborhood` — lihat rasional di bawah |
| latitude, longitude | float | **real** | Koordinat Google Maps |
| rating | float | **real** | Rating Google Maps (1–5) |
| review_count | int | **real** | Jumlah ulasan Google Maps — dipakai sebagai proxy popularitas |
| gross_floor_area_sqm | int | sintetis | Luas outlet |
| seating_capacity | int/null | sintetis | Hanya relevan untuk Marugame (dine-in) |
| store_tier | string | derived dari real | `flagship` / `standard` — probabilitas flagship naik kalau `outlet_type=destination_hub` atau `review_count` tinggi |
| open_date | date | sintetis | Tanggal buka |

**Rasional desain:** dari 44 outlet real Marugame, 34 memang mall (kata
kunci mall di nama/alamat), 10 street-front — mengoreksi asumsi awal
"Marugame 100% mall-based": data real menunjukkan modelnya campuran, meski
tetap mall-dominan. The Harvest yang real 26 dari 28 street-front, 2 di
komplek ruko/plaza kecil — konsisten dengan narasi awal brand sebagai chain
boutique/occasion-based.

**`outlet_type` — outlet bukan titik yang flat/seragam.** Diklasifikasi dari
`location_type` REAL + `review_count` REAL (lihat `assign_outlet_type()` di
`generate_dummy_data.py`) — bukan tebak-tebakan dari nama:

- `destination_hub`: `location_type=mall` **dan** `review_count >= 500` —
  mall yang benar-benar ramai secara riil (Grand Indonesia 1.506 ulasan,
  Tunjungan Plaza 1.870, dst.). Menarik customer dari radius jauh & catchment
  tidak simetris.
- `transit_adjacent`: `location_type=mall` tapi `review_count < 500` — mall
  yang lebih kecil/sepi secara riil, pull-nya moderat.
- `neighborhood`: `location_type=standalone_ruko` — catchment sempit &
  didominasi proximity ke rumah/kantor sekitarnya.

Karena pembaginya `review_count` riil (bukan kata kunci nama yang bisa
kosong), ketiga tipe muncul proporsional di data ini (Marugame: 17 hub / 17
transit / 10 neighborhood; The Harvest: 0 hub / 2 transit / 26
neighborhood) — tidak ada kategori yang kosong seperti versi data
sebelumnya.

`outlet_type` inilah yang menentukan bentuk & lebar catchment tiap outlet di
`customer_origin_sample.csv` (lihat §5) dan dipakai untuk beda-kan
attractiveness dan distance-decay (beta) per tipe di analytics engine.

## 2. `demographics_grid.csv` (~440 baris)
Grid sintetis 500m menutupi area 6 metro — pengganti data sensus/BPS riil
untuk keperluan POC. Jumlah cell per kota proporsional terhadap bobot metro
di `CITIES` (Jakarta terbanyak, Malang/Yogyakarta paling sedikit, dengan
lantai minimum 60 cell supaya whitespace analysis tetap bermakna).

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

## 4. `foot_traffic_transactions.csv` (~32,400 baris)
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

## 5. `customer_origin_sample.csv` (~27,600 baris)
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
Tambah/ubah outlet real dengan mengedit `REAL_OUTLETS_MARUGAME` /
`REAL_OUTLETS_HARVEST` di `generate_dummy_data.py`. Untuk memasukkan kota di
luar Jawa (Medan, Makassar, Palembang, Denpasar, Balikpapan — sudah ada di
`marugame_outlets_indonesia.csv`/`harvest_outlets_indonesia.csv` mentah),
tambah entrinya ke `CITIES`/`DISTRICTS` dict lalu filter ulang dari kedua
CSV sumber itu. Ubah `n_days` atau bobot di `daypart_profile()` untuk
skenario simulasi lain.

## Batasan yang perlu diingat
- Lokasi outlet real, tapi **jumlahnya bukan daftar lengkap** — cuma yang
  berhasil ter-scrape dari Google Maps per 2026-09-10 dan berada di Jawa,
  kemungkinan besar brand punya outlet lain (termasuk di luar Jawa, atau
  yang belum terindeks Google Maps). Jangan dipakai sebagai jumlah outlet
  resmi kedua brand.
- Seluruh angka demand/transaksi/demografi **bukan** data riil — jangan
  dipakai untuk keputusan bisnis, hanya untuk membangun & menguji pipeline
  (schema, join logic, model analitik, dashboard).
- Saat data POS/GIS asli tersedia, mapping ke kolom yang sama di atas →
  downstream code tidak perlu ditulis ulang.
