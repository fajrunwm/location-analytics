# Location Analytics POC — Marugame Udon & The Harvest

GeoAI location analytics untuk dua brand F&B (Marugame Udon, The Harvest),
dibangun dengan dummy data, dirancang untuk diintegrasikan dengan existing
GenAI marketing (image generation) solution.

## Alur end-to-end

```
Data sources (dummy)  →  Spatial database  →  Analytics engine  →
Location insight layer  →  ┬→ Dashboard & API (analis)
                            └→ GenAI marketing (image-gen service existing)
```

Setiap tahap di bawah = satu file yang bisa langsung dijalankan. Semua
reproducible (seed=42) dan saling terhubung lewat `outlet_id`.

## 1. Data sources — `generate_dummy_data.py`

Menghasilkan 4 dataset sintetis (koordinat di-jitter di sekitar Jakarta/
Bandung/Surabaya asli, tapi tidak merepresentasikan lokasi/orang sungguhan):

| File | Isi |
|---|---|
| `outlets.csv` | 42 outlet (18 Marugame Udon, 24 The Harvest) — lokasi, tipe, luas, tier |
| `demographics_grid.csv` | 420 grid cell 500m — populasi, income bracket, distribusi usia |
| `competitor_pois.csv` | 90 POI kompetitor & kontekstual (mall, kantor, transit) |
| `foot_traffic_transactions.csv` | 18,900 baris — simulasi 90 hari × daypart per outlet |

Dokumentasi skema lengkap: `DATA_DICTIONARY.md`.
Saat data asli dari brand tersedia, tinggal mapping ke skema yang sama —
tahap 2 & 3 di bawah tidak perlu berubah.

## 2 & 3. Analytics engine — `site_scoring_huff.py`

Huff gravity model: `P_ij = (A_j^α / D_ij^β) / Σ(A_k^α / D_ik^β)`

Menghasilkan tiga output analitik dari data di tahap 1:

| File | Isi |
|---|---|
| `outlet_site_scores.csv` | Estimasi captured demand & market share per outlet |
| `cannibalization_pairs.csv` | Pasangan outlet sebrand yang demand-nya overlap |
| `whitespace_candidates.csv` | Grid cell demand tinggi, exposure ke outlet existing rendah |

Metodologi & batasan (termasuk kalibrasi beta, asumsi jarak lurus):
`SITE_SCORING_METHODOLOGY.md`.

**Temuan kunci dari run ini**: `MRG-004` & `MRG-017` (Surabaya) overlap 75%
demand cell — kandidat evaluasi jarak sebelum ekspansi baru di area itu.
Beberapa grid cell di Bandung & Surabaya konsisten whitespace untuk kedua
brand — kandidat lokasi baru.

## 4. Location insight → GenAI marketing — `prompt_builder.py`

Mengambil insight per outlet (income bracket dominan, land use, daypart
dominan, kepadatan kompetitor, occasion flag) lalu merangkainya jadi prompt
terstruktur untuk layanan image-gen yang sudah ada — rule-based, bukan
LLM-generated, supaya konsisten & auditable.

Mapping table lengkap, guardrails (tidak ada logo kompetitor, wajah
realistis identifiable, klaim promo tidak diotorisasi), dan contoh prompt
untuk kedua brand: `PROMPT_TEMPLATE_DESIGN.md`.

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

## Status & langkah selanjutnya

Ini POC dengan data dummy — cocok untuk uji skema, logic, dan integrasi
pipeline, **bukan** untuk keputusan bisnis nyata. Yang masih perlu
diputuskan/dikerjakan sebelum produksi:

- Sumber data riil per brand (POS, GIS outlet, demografi vendor/BPS) dan
  proses mapping ke skema yang sama
- Kalibrasi parameter Huff model (`beta`, tier multiplier) terhadap data
  transaksi riil
- Ganti jarak garis-lurus dengan travel-time (routing API) untuk akurasi
  catchment yang lebih baik
- Siapa yang punya hak edit mapping table prompt (marketing vs data team)
  dan proses approval sebelum image hasil generate dipakai publik
- Endpoint API untuk menghubungkan `prompt_builder.py` ke layanan image-gen
  yang sudah ada (saat ini output masih berupa teks prompt, belum call API)
