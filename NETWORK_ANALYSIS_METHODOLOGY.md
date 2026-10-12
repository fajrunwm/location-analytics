# Street network analysis — metodologi

Ekstensi analytics engine: selain Huff gravity model (`SITE_SCORING_METHODOLOGY.md`)
yang memakai jarak garis lurus, tahap ini mengukur **posisi setiap lokasi di dalam
jaringan jalan** — seberapa sering ia dilewati, seberapa mudah dijangkau — lalu
menguji seberapa jauh struktur itu menjelaskan aktivitas komersial nyata.

Implementasi: folder `network/`, dijalankan di Azure ML (job spec di `azureml/jobs/`).
Hasil interaktif: `NETWORK_DASHBOARD.html`.

## Pertanyaan yang dijawab

1. Di mana hotspot komersial di Jabodetabek dan 5 metro lain, dan seberapa banyak
   yang bisa diprediksi hanya dari struktur jalan + populasi?
2. Hex mana yang **posisi jaringannya kuat tetapi aktivitasnya rendah** (peluang),
   dan mana yang aktivitasnya melebihi posisi jaringan (destinasi — biasanya mall)?
3. Apakah posisi jaringan outlet Marugame Udon & The Harvest berkaitan dengan
   popularitasnya (jumlah review Google Maps)?

## Data — semuanya nyata

Berbeda dengan dataset demografi/foot traffic di repo ini (sintetis), tahap ini
hanya memakai data terbuka yang nyata:

| Sumber | Isi | Dipakai sebagai |
|---|---|---|
| OpenStreetMap (Geofabrik extract Jawa) | jaringan jalan kaki & kendaraan; stasiun KRL/MRT/LRT, halte BRT & bus | struktur jaringan, konteks transit |
| Overture Maps Places | POI dengan taksonomi kategori + confidence | **target**: jumlah POI komersial per hex (confidence ≥ 0,5) |
| WorldPop 2025 (100 m, constrained) | estimasi penduduk per sel | konteks populasi |
| `outlets.csv` | lokasi, rating & jumlah review outlet (Google Maps) | validasi tingkat outlet |

"POI komersial" = kategori teratas Overture `food_and_drink`, `shopping`,
`services_and_business`, `lodging`, `arts_and_entertainment`. Target kedua: F&B saja.

## Area

`network/areas.py` — Jakarta Pusat (iterasi cepat), DKI Jakarta, Jabodetabek,
Bandung Raya, Surabaya, Semarang, Yogyakarta, Malang. Enam metro terakhir + Jabodetabek
mencakup ke-72 outlet. Setiap kotak area diberi buffer beberapa km di luar outlet
terluar supaya centrality di lokasi outlet tidak terpotong tepi area (edge effect).

## Metrik jaringan

Dihitung per simpang jalan dengan [cityseer](https://cityseer.benchmarkurbanism.com)
(localized, berbasis jarak jaringan) dan [networkit](https://networkit.github.io)
(global):

| Metrik | Arti | Radius |
|---|---|---|
| Betweenness (shortest path) | seberapa sering simpang dilewati rute terpendek antar titik dalam radius | jalan kaki 400 m – 4,8 km; kendaraan 1 – 20 km |
| Angular betweenness (simplest path) | sama, tapi rute "paling sedikit belokan" — cara orang benar-benar memilih jalan (space syntax) | jalan kaki 400 m – 4,8 km |
| Harmonic closeness / gravity (beta) | seberapa dekat simpang ke semua simpang lain dalam radius | sama |
| Density, cycles | jumlah simpang & loop yang terjangkau | sama |
| Global betweenness & closeness | seluruh graf; eksak bila ≤ 250 ribu simpang, sampling di atasnya | – |
| Populasi terjangkau | penduduk dalam radius jarak jalan kaki | 400 m – 4,8 km |
| Aksesibilitas POI/transit | POI per kategori & halte/stasiun dalam radius jalan kaki | 400 m – 4,8 km |

Radius pendek menangkap "kehidupan jalan" lokal (10 menit jalan kaki); radius
panjang menangkap pergerakan lintas kawasan. Membandingkan keduanya adalah inti
pendekatan localized centrality.

## Dari simpang ke hex

Semua metrik diagregasi ke hex H3 resolusi 9 (~0,1 km², ~350 m antar pusat) — rata-rata
dan maksimum simpang di dalam hex. Prefiks kolom di `hex_features.parquet`:

- `walk_*`, `drive_*`, `ang_*` — struktur jaringan (prediktor)
- `ctx_*` — populasi & transit (prediktor konteks)
- `acc_*` — akses ke POI: **tidak dipakai** untuk memprediksi kepadatan POI
  (dihitung dari POI yang sama dengan target → bocor), hanya untuk profil outlet
- `y_*` — target

## Model

LightGBM, target `log(1 + jumlah POI komersial)` per hex. Tiga set fitur dibandingkan
pada fold yang sama: struktur jalan saja, populasi + transit saja, dan gabungan.
GraphSAGE (GNN pada graf ketetanggaan hex, GPU) dipakai sebagai pembanding.

**Spatial cross-validation**: fold dibentuk per blok H3 resolusi 6 (~36 km²), jadi hex
yang bertetangga tidak pernah ada di data latih dan data uji sekaligus. Tanpa ini,
autokorelasi spasial membuat skor terlihat jauh lebih bagus dari kenyataan.

Semua prediksi di dashboard berasal dari fold di mana hex tersebut **tidak** ikut
dilatih (out-of-fold).

**Gap potensi** = prediksi − aktual (skala log). Positif: struktur jaringan +
populasi "mendukung" lebih banyak aktivitas dari yang ada sekarang — kandidat
whitespace. Negatif: aktivitas melebihi posisi jaringan — biasanya destinasi
(mall, kawasan wisata) yang menarik pengunjung karena alasan lain.

## Validasi outlet

Untuk tiap outlet: fitur simpang terdekat (≤ 300 m) dan fitur hex-nya, dikorelasikan
(Spearman) dengan jumlah review Google Maps. Karena ratusan fitur diuji terhadap 72
outlet, p-value dikoreksi Benjamini–Hochberg; hanya q < 0,1 yang dianggap sinyal.

## Batasan

- **Jumlah review ≠ omzet.** Review dipengaruhi umur outlet, lokasi mall, dan
  kebiasaan pengunjung. Validasi sesungguhnya butuh data transaksi/omzet per outlet.
- **POI ≠ foot traffic.** Kepadatan POI adalah proxy aktivitas yang mapan di literatur,
  tetapi tidak sama dengan jumlah orang yang lewat.
- **Kelengkapan data** OSM dan Overture di Indonesia tidak merata — jalan kecil
  dan usaha informal bisa kurang terwakili, terutama di pinggiran.
- **Jaringan jalan kaki** dari OSM termasuk jalan arteri tanpa trotoar layak; metrik
  "jalan kaki" lebih tepat dibaca sebagai "pergerakan lokal".
- Koefisien model menjelaskan **asosiasi**, bukan sebab-akibat.

## Hasil

Run final Azure ML (`train-final`, `bootstrap-final`; 12 Okt 2026). Latih: Jabodetabek,
Bandung Raya, Surabaya, Semarang, Yogyakarta, Malang (52.760 hex, 72 outlet). Uji
out-of-sample: Medan, Palembang, Denpasar & Badung, Makassar, Balikpapan (14.537 hex,
26 outlet) — tidak pernah dipakai melatih.

**Prediksi hotspot** — target log(1 + POI komersial) per hex, R²:

| Fitur | Spatial CV (semua hex) | Spatial CV (kota padat) | Kota baru (luar Jawa) |
|---|---|---|---|
| Struktur jalan saja | 0,68 | 0,52 | 0,50 |
| Populasi + transit saja | 0,59 | 0,39 | 0,36 |
| Gabungan | 0,70 | 0,56 | 0,51 |

"Kota padat" = hex ≥ ~9.500 jiwa/km² (gradien kota-desa dihilangkan). Per kota baru
(gabungan): Makassar 0,65 · Balikpapan 0,58 · Palembang 0,53 · Medan 0,49 · Denpasar 0,23.
Di Denpasar populasi saja negatif (−0,26), struktur jalan tetap 0,38 — aktivitas di sana
digerakkan pariwisata, bukan penduduk lokal.

**Fitur terpenting** (porsi mean |SHAP|, semua radius dijumlah): angular closeness jalan kaki
34,5% · populasi terjangkau 17,2% · closeness kendaraan 10,0% · angular betweenness 7,6% ·
kepadatan simpang 5,1%. Sinyalnya lokal — radius 1,2–2,4 km paling berpengaruh; betweenness
dan closeness global (eksak, seluruh graf) hampir tidak menambah.

**GNN pembanding** (GraphSAGE di graf ketetanggaan hex, sweep di Azure; 2 trial selesai
sebelum batas waktu): terbaik R² 0,712 (kota padat 0,576) vs LightGBM 0,702 (0,559).
Konteks tetangga hampir tidak menambah di luar fitur multi-radius, jadi LightGBM yang bisa
dijelaskan tetap jadi model utama.

**Peluang & destinasi stabil** (bootstrap blok spasial 200×; label hanya bila gap ≥ 1,5×
bertahan di ≥ 90% fit out-of-bag): Jabodetabek 4.433 hex peluang / 5.678 destinasi,
Bandung 918 / 885, Surabaya 903 / 1.045, Yogyakarta 499 / 723, Semarang 371 / 495,
Malang 314 / 416.

**Outlet** (n = 98): model "mengenali" mall tanpa diberi tahu — median gap outlet mall
−1,68 (n = 47) vs ruko −0,54 (n = 50). Tidak ada fitur jaringan yang signifikan terhadap
jumlah review setelah koreksi BH dan kontrol brand × tipe lokasi (terkuat ρ = 0,25, q = 0,52):
posisi jaringan memprediksi *di mana* keramaian komersial terjadi, bukan *outlet mana* yang
paling banyak direview.

## Huff dengan jarak jaringan jalan — `network/huff_network.py`

`site_scoring_huff.py` mengubah jarak garis lurus jadi "jarak jalan" dengan faktor circuity
asumsi per kota (1,25–1,35, belum pernah di-fit). Tahap ini menggantinya dengan jarak
terpendek di graf jalan kendaraan OSM yang sama, untuk 6 metro Jawa (72 outlet), lalu
membandingkan dua varian Huff dengan parameter identik (alpha, beta per `outlet_type`,
attractiveness, kompetisi sebrand) dan demand identik (penduduk WorldPop per hex) — jadi
semua selisih murni karena cara mengukur jarak. Job Azure ML `huff-network-6metro`.

**Circuity terukur** (jarak jalan ÷ jarak lurus, pasangan hex–outlet 1–20 km):

| Metro | Asumsi lama | Median terukur | 1–3 km | 3–7 km | 7–12 km | 12–20 km |
|---|---|---|---|---|---|---|
| Jabodetabek | 1,35 | 1,30 | 1,62 | 1,41 | 1,32 | 1,27 |
| Surabaya | 1,30 | 1,30 | 1,55 | 1,37 | 1,31 | 1,25 |
| Bandung Raya | 1,25 | 1,29 | 1,47 | 1,34 | 1,28 | 1,27 |
| Yogyakarta | 1,30 | 1,29 | 1,42 | 1,32 | 1,27 | 1,26 |
| Malang | 1,30 (default) | 1,31 | 1,40 | 1,33 | 1,27 | 1,27 |
| Semarang | 1,30 (default) | 1,28 | 1,40 | 1,30 | 1,25 | 1,26 |

Rata-rata per kota ternyata sudah dekat dengan asumsi. Yang salah adalah anggapan faktornya
**konstan**: perjalanan pendek 40–60% lebih jauh dari garis lurus (sungai, rel, jalan satu
arah, blok tertutup), perjalanan jauh hanya ~26%. Faktor konstan meremehkan jarak dekat →
jangkauan outlet lingkungan (`neighborhood`, beta curam) dinilai terlalu besar.

**Dampak ke Huff:** 25 dari 72 outlet berubah captured demand ≥ 5%, 8 berubah ≥ 10%
(terbesar: Marugame MERR Surabaya −26%, Marugame Majapahit Semarang −22%, Harvest Graha
Famili −20%, Marugame Soekarno-Hatta T3 −19%; naik: Harvest Bengawan +13%, Marugame Botani
Square +12%). Ranking captured demand berubah untuk 16 dari 36 outlet Jabodetabek dan 4 dari
10 di Bandung. Populasi yang diperebutkan dua outlet sebrand (P ≥ 0,15 untuk keduanya) naik
8% dan pasangan yang berebut bertambah 119 → 135; terbesar Marugame Bandung MRG-025/026 (+38%).

**Waktu tempuh** (mobil, kecepatan diimputasi OSMnx per kelas jalan, rata-rata tertimbang
demand yang ditangkap): median 9 menit (Malang) sampai 21 menit (Jabodetabek); outlet mall
19 menit vs ruko 15 menit. Ditampilkan di popup outlet `NETWORK_DASHBOARD.html`.

**Implikasi untuk `site_scoring_huff.py`:** ganti faktor konstan dengan circuity per pita
jarak di atas (atau langsung jarak jaringan), lalu kalibrasi ulang beta per `outlet_type` —
beta lama di-set bersama jarak yang terlalu pendek untuk perjalanan dekat. Batasan: kecepatan
jalan adalah imputasi tanpa kemacetan, jadi waktu tempuh adalah batas bawah; pasangan
> 60 km jarak jalan diabaikan (bobot Huff-nya sudah ~0).
