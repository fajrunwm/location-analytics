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
sama ("contested"). `contested_cells_pct` tinggi → sinyal untuk evaluasi
jarak antar outlet sebelum ekspansi baru di area itu.

**3. `whitespace_candidates.csv`** — grid cell dengan demand tinggi tapi
total exposure ke outlet existing rendah (`whitespace_score` = demand_norm ×
(1 - exposure_norm)). Ini kandidat area untuk lokasi baru.

**4. `outlet_catchment_validation.csv`** — median/p90 `distance_km` observed
per outlet (dari `customer_origin_sample.csv`) dibandingkan dengan
`expected_mean_km` asumsi `outlet_type`-nya, plus flag `reclassify_review`
untuk outlet yang menyimpang signifikan. Lihat §Heterogeneous pull.

## Batasan penting (karena data masih dummy)

- Grid demografi di-generate independen dari lokasi outlet (scatter acak per
  kota), jadi hasil whitespace di sini **ilustratif untuk menguji pipeline**,
  bukan rekomendasi lokasi nyata.
- `beta` per `outlet_type` dan multiplier tier/tipe adalah asumsi awal —
  begitu ada data transaksi & alamat pelanggan riil, sebaiknya di-*calibrate*
  (fit beta per tipe terhadap actual catchment behavior, misal via regresi
  terhadap data visit riil), bukan cuma divalidasi seperti sekarang.
- Model ini tidak memperhitungkan barrier fisik (macet, sungai, tol) —
  jarak yang dipakai garis lurus (as-the-crow-flies), bukan travel time.
  Upgrade lanjutan: ganti `D_ij` dengan travel-time dari routing API.

## Cara pakai ulang
```bash
python3 site_scoring_huff.py
```
Ubah `ALPHA`, `BETA`, `TIER_MULTIPLIER`, atau `INCOME_SPEND_MULTIPLIER` di
bagian atas file untuk skenario/kalibrasi lain.
