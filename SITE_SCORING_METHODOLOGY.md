# Site scoring engine — metodologi (Huff gravity model)

Bagian "analytics engine" pada diagram arsitektur. Implementasi:
`site_scoring_huff.py`.

## Model

```
P_ij = (A_j^alpha / D_ij^beta_j) / sum_k (A_k^alpha / D_ik^beta_k)
```

- `P_ij` — probabilitas demand di grid cell *i* tertarik ke outlet *j*
- `A_j` — attractiveness outlet = luas outlet × multiplier tier (flagship 1.3x,
  standard 1.0x) × multiplier `outlet_type` (lihat §Heterogeneous pull)
- `D_ij` — jarak haversine (km), floor 0.15km untuk hindari division blow-up
- `alpha = 1.0` untuk semua outlet
- `beta_j` — **per outlet, bukan satu angka global** (lihat §Heterogeneous pull)

Kompetisi dimodelkan hanya **antar outlet brand yang sama, di kota yang sama**
— asumsinya pelanggan Marugame tidak "kompetisi" melawan pilihan The Harvest
dalam satu keputusan makan.

## Heterogeneous pull — outlet bukan titik yang flat

Versi awal model ini memakai satu `beta` global (1.8) untuk semua outlet.
Masalahnya: outlet mall besar yang jadi assembly point (mis. flagship di
Grand Indonesia) secara nyata menarik customer dari radius jauh dan bentuk
catchment yang tidak simetris (mengikuti sumbu transit/komuter), sementara
outlet ruko/neighborhood didominasi proximity murni. Satu beta global akan
**meremehkan jangkauan hub** dan **melebih-lebihkan jangkauan neighborhood**.

Solusinya: setiap outlet punya `outlet_type` (`destination_hub` /
`transit_adjacent` / `neighborhood`, ditentukan di `generate_dummy_data.py`
dari kombinasi jenis mall & tier), dengan attractiveness dan beta berbeda:

| outlet_type | attractiveness multiplier | beta (decay jarak) |
|---|---|---|
| `destination_hub` | 1.6x | 1.1 (landai — orang mau tempuh jauh) |
| `transit_adjacent` | 1.2x | 1.5 |
| `neighborhood` | 1.0x | 2.3 (curam — didominasi proximity) |

Nilai-nilai ini **diasumsikan**, paralel dengan parameter dispersion jarak di
`ORIGIN_TYPE_PARAMS` (generator) — bukan hasil fit terhadap data riil.
`compute_catchment_validation()` di `site_scoring_huff.py` membandingkan
asumsi ini dengan dispersion jarak *observed* dari
`customer_origin_sample.csv` (median & p90 `distance_km` per outlet) dan
menghasilkan `outlet_catchment_validation.csv`, flag outlet yang median
observed-nya menyimpang ≥50% dari `expected_mean_km` tipe-nya — sinyal untuk
review klasifikasi `outlet_type`-nya. Saat data POS/loyalty riil tersedia,
langkah yang sama dipakai untuk **mengkalibrasi ulang** beta & multiplier per
tipe, bukan cuma memvalidasi.

`demand_potential` per grid cell = `population_est x income_spend_multiplier`
— ini **indeks sintetis**, bukan nilai rupiah riil (dan datanya sendiri dummy).

## Tiga output

**1. `outlet_site_scores.csv`** — estimasi captured demand & market share
per outlet dalam kota+brand yang sama. Berguna untuk ranking performa lokasi
relatif, bukan angka absolut.

**2. `cannibalization_pairs.csv`** — pasangan outlet sebrand di kota yang
sama di mana kedua outlet punya Huff probability ≥15% dari demand cell yang
sama ("contested"). `contested_cells_pct` tetap simetris (persentase cell
yang sama-sama contested), tapi **demand at risk-nya dihitung directional**:
`a_demand_at_risk_pct` = persentase dari captured demand outlet A sendiri
yang berasal dari cell yang dikontes dengan B, dan sebaliknya untuk
`b_demand_at_risk_pct`. Keduanya jarang sama — outlet kecil bisa kehilangan
porsi jauh lebih besar dari basisnya sendiri dibanding tetangga besarnya,
meski overlap cell-nya identik. `net_asymmetry_pct` (`a - b`) positif berarti
A lebih rentan terhadap B daripada sebaliknya. Lihat §Heterogeneous pull
untuk kenapa ini penting dicek sebelum keputusan ekspansi/tutup outlet.

**3. `cannibalization_network.csv`** — agregat per outlet: `n_contested_neighbors`
(berapa outlet sebrand yang overlap dengannya) dan `total_demand_at_risk_pct`
(jumlah `demand_at_risk_pct` dari SEMUA tetangga yang overlap, bisa >100%
kalau demand yang sama dikontes lebih dari satu tetangga sekaligus — itu
memang menunjukkan tekanan kompetitif kumulatif, bukan bug). Dipakai untuk
menjawab "outlet mana yang paling terkepung", bukan cuma "pasangan mana yang
overlap paling besar" — dua hal yang bisa beda jawabannya kalau satu outlet
dikepung 3 tetangga sedang-sedang saja sementara pasangan tunggal terbesar
ada di tempat lain.

**4. `whitespace_candidates.csv`** — grid cell dengan demand tinggi tapi
total exposure ke outlet existing rendah (`whitespace_score` = demand_norm ×
(1 - exposure_norm)). Ini kandidat area untuk lokasi baru.

**5. `outlet_catchment_validation.csv`** — median/p90 `distance_km` observed
per outlet (dari `customer_origin_sample.csv`) dibandingkan dengan
`expected_mean_km` asumsi `outlet_type`-nya, plus flag `reclassify_review`
untuk outlet yang menyimpang signifikan. Lihat §Heterogeneous pull.

## Road-distance proxy (bukan lagi jarak lurus mentah)

`D_ij` sekarang adalah **jarak efektif**: haversine (garis lurus) dikali
`circuity factor` — `ROAD_CIRCUITY_BY_CITY` (Jakarta 1.35x, Bandung 1.25x,
Surabaya 1.30x, mencerminkan kepadatan grid jalan/toll detour tiap metro)
dikali `LAND_USE_CIRCUITY_ADJ` (office/CBD 1.10x lebih berbelok, residential
0.95x lebih langsung). Ini dipakai konsisten di `compute_huff_scores()` dan
`compute_cannibalization()` lewat `effective_distance_km()`.

**Kenapa ini penting:** di kota macet dengan grid jalan tidak beraturan
(Jakarta/Bandung/Surabaya), jarak lurus 2km bisa berarti 5 menit atau 30
menit tempuh tergantung rute. Tanpa koreksi ini, catchment terutama untuk
outlet `neighborhood` (The Harvest residential) bisa jauh lebih optimis dari
kenyataan — pelanggan sebenarnya menempuh jarak jalan yang jauh lebih
panjang dari garis lurusnya.

**Ini masih proxy, bukan solusi final** — nilai circuity factor di atas
adalah *estimasi placeholder*, bukan hasil fit terhadap data trip riil.
Upgrade berikutnya: ganti `effective_distance_km()` dengan travel-time dari
routing engine (OSRM self-hosted atau API komersial) begitu data alamat/GPS
pelanggan riil tersedia — lihat `customer_origin_sample.csv` sebagai titik
mula validasi (bandingkan `distance_km` lurus vs travel-time riil di sana).

## Batasan penting (karena data masih dummy)

- Grid demografi di-generate independen dari lokasi outlet (scatter acak per
  kota), jadi hasil whitespace di sini **ilustratif untuk menguji pipeline**,
  bukan rekomendasi lokasi nyata.
- `beta` per `outlet_type` dan multiplier tier/tipe adalah asumsi awal —
  begitu ada data transaksi & alamat pelanggan riil, sebaiknya di-*calibrate*
  (fit beta per tipe terhadap actual catchment behavior, misal via regresi
  terhadap data visit riil), bukan cuma divalidasi seperti sekarang.
- Circuity factor per kota/land-use di atas juga asumsi awal, belum di-fit
  ke data travel-time riil — lihat §Road-distance proxy.
- Model belum memperhitungkan barrier fisik spesifik (sungai, rel, area
  macet spesifik jam tertentu) — circuity factor hanya rata-rata per
  kota+land-use, bukan rute per-pasangan titik yang presisi.

## Cara pakai ulang
```bash
python3 site_scoring_huff.py
```
Ubah `ALPHA`, `BETA_BY_OUTLET_TYPE`, `TIER_MULTIPLIER`,
`INCOME_SPEND_MULTIPLIER`, atau `ROAD_CIRCUITY_BY_CITY`/
`LAND_USE_CIRCUITY_ADJ` di bagian atas file untuk skenario/kalibrasi lain.
