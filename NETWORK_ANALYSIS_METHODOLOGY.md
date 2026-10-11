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

_Diisi setelah run Azure ML selesai — lihat bagian berikut._
