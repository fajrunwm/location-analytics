# Prompt template design — location insight → GenAI image generation

Tujuan: mengubah *location insight profile* (output dari analytics engine) jadi
prompt terstruktur untuk layanan GenAI image-gen yang sudah ada, sehingga
creative campaign terasa relevan secara lokal tanpa desain manual per outlet.

Prinsip desain:
- **Rule-based, bukan LLM-generated** di tahap mapping insight → directive —
  supaya konsisten, auditable, dan mudah di-tuning tim marketing tanpa
  prompt-engineering ulang tiap kali.
- **Brand visual identity tetap konstan** — insight lokasi mengubah *konteks*
  (setting, waktu, audiens, tone), bukan identitas visual brand.
- **Tidak menyertakan data personal/individual** — semua input adalah
  agregat area (demografi grid, kepadatan kompetitor), bukan data pelanggan
  individual.

---

## 1. Input: Location Insight Profile (kontrak data)

```json
{
  "outlet_id": "MRG-001",
  "brand": "Marugame Udon",
  "location_type": "mall",
  "city": "Bandung",
  "dominant_income_bracket": "upper_middle",
  "dominant_land_use": "mixed",
  "dominant_daypart": "dinner",
  "competitor_density": "medium",
  "occasion_flag": null
}
```
Field ini adalah output dari `analytics engine` + `location insight layer`
pada diagram arsitektur sebelumnya — bukan input baru yang perlu di-generate
manual.

## 2. Mapping table — insight field → creative directive

| Insight field | Nilai | Directive ke prompt |
|---|---|---|
| `brand` | Marugame Udon | Base scene: mangkuk udon hangat, bahan segar, gaya *fast-casual Jepang* |
| `brand` | The Harvest | Base scene: kue/pastry, tampilan *bakery boutique*, pencahayaan lembut |
| `location_type` | mall | Setting: interior mall modern, cahaya terang, suasana ramai-tapi-rapi |
| `location_type` | standalone_ruko | Setting: fasad toko jalan, suasana neighborhood, lebih intim |
| `dominant_daypart` | lunch/dinner | Waktu: jam makan aktif, framing "meal moment", pencahayaan hangat |
| `dominant_daypart` | breakfast/afternoon | Waktu: santai, framing "snack/treat moment" |
| `dominant_land_use` | residential | Audiens: keluarga, konteks akhir pekan/santai |
| `dominant_land_use` | office | Audiens: profesional, konteks makan siang cepat |
| `dominant_land_use` | commercial/mixed | Audiens: umum/urban, konteks jalan-jalan/belanja |
| `dominant_income_bracket` | upper_middle/high | Tone: presentasi premium, framing kualitas & pengalaman |
| `dominant_income_bracket` | low/lower_middle/middle | Tone: framing value & kehangatan, tanpa kesan murahan |
| `competitor_density` | high | Angle diferensiasi: tonjolkan keunikan produk/porsi |
| `competitor_density` | low/medium | Angle standar: fokus produk & suasana |
| `occasion_flag` | birthday/hampers (Harvest event day) | Tambahkan elemen hadiah/perayaan pada scene |

> Catatan: `income_bracket` dipakai untuk **tone presentasi**, bukan untuk
> menyasar/mengecualikan kelompok tertentu — tidak pernah dipakai untuk
> framing yang merendahkan segmen manapun.

## 3. Struktur template (assembly order)

```
[1. Brand base scene]  →  [2. Setting/location_type]  →
[3. Lighting & occasion by daypart]  →  [4. Audience framing]  →
[5. Presentation tone]  →  [6. Differentiation angle]  →
[7. Brand style guardrails]  →  [8. Negative prompt]
```

Setiap segmen adalah kalimat pendek yang di-concatenate — bukan satu blok
paragraf yang di-generate LLM, supaya tim non-teknis bisa audit/edit tiap
segmen di config tanpa menyentuh kode.

## 4. Guardrails (masuk ke setiap request)

- Tidak ada logo/merek kompetitor yang disebut atau digambar
- Tidak ada wajah/orang yang identifiable secara realistis (gunakan gaya
  ilustratif/stok, bukan foto orang riil)
- Tidak ada klaim promo/diskon yang tidak diotorisasi tim marketing
- Konsisten dengan brand guideline visual (warna, tipografi, mood) yang sudah
  ada — insight lokasi hanya mengubah konteks, bukan identitas visual

## 5. Contoh output (dari data dummy)

**Marugame Udon — outlet mall, dinner-dominant, upper-middle catchment:**
> "Mangkuk udon hangat dengan topping segar, gaya fast-casual Jepang.
> Interior mall modern dengan cahaya terang. Momen makan malam yang hangat.
> Audiens urban umum. Presentasi premium, framing kualitas & pengalaman.
> Fokus pada produk & suasana. Konsisten dengan identitas visual Marugame
> Udon (merah-putih, bersih, modern)."
> *Negative:* logo kompetitor, wajah realistis identifiable, klaim diskon.

**The Harvest — outlet standalone, "event day", residential catchment:**
> "Kue/pastry dengan tampilan bakery boutique, pencahayaan lembut. Fasad
> toko jalan, suasana neighborhood yang intim. Momen snack/treat santai
> dengan elemen hadiah & perayaan. Audiens keluarga, konteks akhir pekan.
> Framing kehangatan & value, tanpa kesan murahan. Konsisten dengan
> identitas visual The Harvest (pastel, homey, boutique)."
> *Negative:* logo kompetitor, wajah realistis identifiable, klaim diskon.

Lihat `prompt_builder.py` untuk implementasi rule-based yang generate contoh
di atas langsung dari data dummy (`outlets.csv`, `demographics_grid.csv`,
`competitor_pois.csv`, `foot_traffic_transactions.csv`).

## 6. Yang perlu diputuskan tim sebelum implementasi produksi

- Siapa yang punya hak edit mapping table (marketing vs data team)?
- Apakah `occasion_flag` di-trigger otomatis dari data (event day spike) atau
  manual oleh campaign planner?
- Bagaimana proses approval sebelum image hasil generate dipakai publik?
