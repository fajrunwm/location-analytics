# Site scoring engine — metodologi (Huff gravity model)

Bagian "analytics engine" pada diagram arsitektur. Implementasi:
`site_scoring_huff.py`.

## Model

```
P_ij = (A_j^alpha / D_ij^beta) / sum_k (A_k^alpha / D_ik^beta)
```

- `P_ij` — probabilitas demand di grid cell *i* tertarik ke outlet *j*
- `A_j` — attractiveness outlet (luas outlet x multiplier tier: flagship 1.3x, standard 1.0x)
- `D_ij` — jarak haversine (km), floor 0.15km untuk hindari division blow-up
- `alpha = 1.0`, `beta = 1.8` — beta agak tinggi karena kategori fast-casual/bakery bersifat convenience-driven (sensitif jarak)

Kompetisi dimodelkan hanya **antar outlet brand yang sama, di kota yang sama**
— asumsinya pelanggan Marugame tidak "kompetisi" melawan pilihan The Harvest
dalam satu keputusan makan.

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

## Batasan penting (karena data masih dummy)

- Grid demografi di-generate independen dari lokasi outlet (scatter acak per
  kota), jadi hasil whitespace di sini **ilustratif untuk menguji pipeline**,
  bukan rekomendasi lokasi nyata.
- `beta=1.8` dan multiplier tier adalah asumsi awal — begitu ada data
  transaksi riil, sebaiknya di-*calibrate* (fit beta terhadap actual
  catchment behavior, misal via regresi terhadap data visit riil).
- Model ini tidak memperhitungkan barrier fisik (macet, sungai, tol) —
  jarak yang dipakai garis lurus (as-the-crow-flies), bukan travel time.
  Upgrade lanjutan: ganti `D_ij` dengan travel-time dari routing API.

## Cara pakai ulang
```bash
python3 site_scoring_huff.py
```
Ubah `ALPHA`, `BETA`, `TIER_MULTIPLIER`, atau `INCOME_SPEND_MULTIPLIER` di
bagian atas file untuk skenario/kalibrasi lain.
