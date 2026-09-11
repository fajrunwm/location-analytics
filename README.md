# Location Analytics POC — Marugame Udon & The Harvest

GeoAI location analytics untuk dua brand F&B (Marugame Udon, The Harvest).
Lokasi outlet (nama, kota, distrik, tipe lokasi, koordinat, rating & jumlah
ulasan) adalah data **REAL**, di-scrape dari Google Maps (via Claude
Desktop) dan difilter ke pulau Jawa — sisanya (demografi, POI kompetitor,
foot traffic, customer origin, dan atribut non-lokasi outlet) tetap
**sintetis**. Dirancang untuk diintegrasikan dengan existing GenAI marketing
(image generation) solution.

## Alur end-to-end

```
Data sources (lokasi real + atribut sintetis)  →  Spatial database  →
Analytics engine  →  Location insight layer  →  ┬→ Dashboard interaktif (analis) — sudah ada, DASHBOARD.html
                                                  └→ GenAI marketing (image-gen service existing)
```

Setiap tahap di bawah = satu file yang bisa langsung dijalankan. Semua
reproducible (seed=42) dan saling terhubung lewat `outlet_id`.

## 1. Data sources — `generate_dummy_data.py`

Menghasilkan 5 dataset (lihat `DATA_DICTIONARY.md` §Provenance data lokasi
untuk rincian mana yang real vs sintetis):

| File | Isi |
|---|---|
| `outlets.csv` | 72 outlet real (44 Marugame Udon, 28 The Harvest) di Jawa — nama, kota, distrik, tipe lokasi, koordinat, rating & jumlah ulasan REAL (Google Maps); luas, tier, tanggal buka tetap sintetis (`outlet_type`/`store_tier` diturunkan dari review_count real) |
| `demographics_grid.csv` | ~440 grid cell 500m (sintetis) di 6 metro — populasi, income bracket, distribusi usia |
| `competitor_pois.csv` | 90 POI kompetitor & kontekstual (sintetis) |
| `foot_traffic_transactions.csv` | ~32,400 baris — simulasi 90 hari × daypart per outlet (sintetis) |
| `customer_origin_sample.csv` | ~27,600 baris — per-customer home→outlet trip (sintetis), dipakai untuk mengukur origin dispersion (jangkauan tarik) tiap outlet |

Dokumentasi skema lengkap: `DATA_DICTIONARY.md`.
Sumber mentah (`marugame_outlets_indonesia.csv` / `harvest_outlets_indonesia.csv`,
hasil scrape Google Maps se-Indonesia) juga ada di repo — `generate_dummy_data.py`
memfilternya ke Jawa saja. Saat data POS/GIS asli dari brand tersedia, tinggal
mapping ke skema yang sama — tahap 2 & 3 di bawah tidak perlu berubah.

## 2 & 3. Analytics engine — `site_scoring_huff.py`

Huff gravity model: `P_ij = (A_j^α / D_ij^β) / Σ(A_k^α / D_ik^β)`

Menghasilkan lima output analitik dari data di tahap 1:

| File | Isi |
|---|---|
| `outlet_site_scores.csv` | Estimasi captured demand & market share per outlet |
| `cannibalization_pairs.csv` | Pasangan outlet sebrand overlap — directional (`a_demand_at_risk_pct` vs `b_demand_at_risk_pct`, jarang sama) |
| `cannibalization_network.csv` | Agregat per outlet: total demand at risk dari SELURUH tetangga overlap, bukan cuma pasangan terburuk |
| `whitespace_candidates.csv` | Grid cell demand tinggi, exposure ke outlet existing rendah |
| `outlet_catchment_validation.csv` | Dispersion jarak observed vs asumsi `outlet_type`, flag kandidat salah klasifikasi |
| `outlet_confidence_bands.csv` | Monte Carlo p10/p50/p90 & `confidence_label` per outlet — seberapa sensitif captured demand terhadap ketidakpastian asumsi (juga digabung ke `outlet_site_scores.csv`) |

Metodologi & batasan (termasuk kalibrasi beta, asumsi jarak lurus):
`SITE_SCORING_METHODOLOGY.md`.

**Temuan kunci dari run ini**: `MRG-027` (neighborhood, outlet street-front
di Jl. Raya Darmo) & `MRG-032` (destination_hub, Royal Plaza) — dua-duanya
di kecamatan Wonokromo, Surabaya — overlap cuma 6.7% demand cell, tapi
demand-at-risk directional-nya 78.6% vs 6.5%: outlet kecil kehilangan hampir
80% basisnya sendiri, outlet mall besar nyaris tidak terasa. Sinyal jelas
untuk evaluasi jarak sebelum ekspansi baru Marugame di area itu.

## 4. Location insight → GenAI marketing — `prompt_builder.py`

Mengambil insight per outlet (income bracket dominan, land use, daypart
dominan, kepadatan kompetitor, occasion flag) lalu merangkainya jadi prompt
terstruktur untuk layanan image-gen yang sudah ada — rule-based, bukan
LLM-generated, supaya konsisten & auditable.

Mapping table lengkap, guardrails (tidak ada logo kompetitor, wajah
realistis identifiable, klaim promo tidak diotorisasi), dan contoh prompt
untuk kedua brand: `PROMPT_TEMPLATE_DESIGN.md`.

## 5. Dashboard interaktif — `DASHBOARD.html`

Standalone (buka langsung di browser, tidak butuh server), menampilkan
seluruh output tahap 2–4 di atas peta CARTO Positron (light/dark) dengan
marker clustering untuk 72 outlet real:

- Filter data: search nama/ID outlet, dropdown kota, brand, karakter lokasi
- Overlay "peluang lokasi baru" (whitespace) dengan tooltip
- Panel detail per outlet: site score, confidence band, daftar saingan
  (expand/collapse)
- Perbandingan dua outlet berdampingan (klik marker kedua di peta) —
  menampilkan hubungan cannibalization directional-nya langsung dari
  `cannibalization_pairs.csv`

`ARCHITECTURE.html` (diagram teknis) dan `OVERVIEW_DECK.html` (deck 9-slide
non-teknis) adalah dua ringkasan visual lain dari pipeline yang sama —
lihat §Peta dokumen.

## Cara jalankan semuanya berurutan

```bash
python3 generate_dummy_data.py      # → outlets.csv, demographics_grid.csv, dst.
python3 site_scoring_huff.py        # → outlet_site_scores.csv, dst.
python3 prompt_builder.py           # → contoh prompt siap kirim ke image-gen
```

## Peta dokumen

| Dokumen | Menjawab pertanyaan |
|---|---|
| `DATA_DICTIONARY.md` | Apa isi tiap kolom dan mengapa dimodelkan begitu? |
| `SITE_SCORING_METHODOLOGY.md` | Bagaimana site score/whitespace dihitung, apa batasannya? |
| `PROMPT_TEMPLATE_DESIGN.md` | Bagaimana insight lokasi jadi prompt image-gen? |
| `DASHBOARD.html` | Tool interaktif — jelajahi outlet, bandingkan, cek cannibalization sendiri |
| `ARCHITECTURE.html` | Diagram teknis pipeline end-to-end, untuk audiens yang mau lihat mekanismenya |
| `OVERVIEW_DECK.html` | Ringkasan 9-slide non-teknis, untuk audiens bisnis/klien |

## Status & langkah selanjutnya

Lokasi & rating outlet sudah real (Google Maps, Jawa), tapi seluruh angka
demand/transaksi/demografi masih sintetis — **bukan** untuk keputusan bisnis
nyata. Yang masih perlu diputuskan/dikerjakan sebelum produksi:

- Sumber data riil per brand (POS, jumlah outlet lengkap dari internal
  brand — bukan cuma yang terindeks di Google Maps per 2026-09-10, demografi
  vendor/BPS) dan proses mapping ke skema yang sama
- Perluas cakupan pulau/kota kalau brand memang punya outlet signifikan di
  luar Jawa (data mentah sudah mencakup Medan, Makassar, Palembang,
  Denpasar, Balikpapan — sengaja belum dipakai)
- Kalibrasi parameter Huff model (`beta`, tier multiplier) terhadap data
  transaksi riil
- Ganti jarak garis-lurus dengan travel-time (routing API) untuk akurasi
  catchment yang lebih baik
- Siapa yang punya hak edit mapping table prompt (marketing vs data team)
  dan proses approval sebelum image hasil generate dipakai publik
- Endpoint API untuk menghubungkan `prompt_builder.py` ke layanan image-gen
  yang sudah ada (saat ini output masih berupa teks prompt, belum call API)
