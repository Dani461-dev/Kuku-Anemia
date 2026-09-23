# Instruksi Run Manual — Retrain Hb dari Dataset Baru

Semua kode sudah siap & teruji (smoke test). Jalankan berurutan dari folder
`anemia-app`:

```bash
cd "D:/Documents/Anemia Kuku/anemia-app"
```

> Pastikan terminal pakai Python yang sama dengan pipeline (Python 3.11,
>   `python --version`).

---

## ✅ Sudah selesai (tidak perlu diulang)

- Setup torch CUDA (`2.14.0+cu130`, aktif di RTX 2050)
- P0 triase (deteksi 100%)
- P1 ekstraksi fitur semua pasien →
  `experiments/hb_newdata/outputs/features_fingernails_open.csv` dan
  `features_fingernails_closed.csv` (masing-masing 5.782 baris)
- Baseline model lama di MSU-250 → `outputs/p3_predict_baseline.csv`
  (**MAE 14,93 g/L · R² 0,485**)

---

## STEP 1 — Retrain (3 varian: open / closed / mean)

```bash
python experiments/hb_newdata/p2_train.py
```

- Membuat dataset latih bersih (`outputs/train/features_{open,closed,mean}.csv`):
  drop baris fitur NaN / Hb di luar kontrak (30–200 g/L).
- Menjalankan `core/train_hb.py --protocol nested --no-balance` untuk tiap varian.
- Keluaran: `experiments/hb_newdata/models/{open,closed,mean}/`
  (joblib + model_metadata.json)
- Durasi: ±10–20 menit. Di akhir muncul **RINGKASAN CV**:

```
========= RINGKASAN CV =========
  open     n= 5782  CV MAE   .. g/L (.. g/dL)  R2 +0...
  closed   n= 5782  CV MAE   .. g/L (.. g/dL)  R2 +0...
  mean     n= 5782  CV MAE   .. g/L (.. g/dL)  R2 +0...
```

**Kriteria sukses**: CV MAE < **15,96 g/L** (model lama). Bandingkan ketiganya.

> ⚠️ Jangan pakai `--quick` / `--limit` saat run final (itu untuk tes cepat).

---

## STEP 2 — Validasi lintas-domain (MSU-250)

Jalankan untuk **setiap** model baru:

```bash
python experiments/hb_newdata/p3_eval_msu.py --model experiments/hb_newdata/models/open/elasticnet_model.joblib   --tag open
python experiments/hb_newdata/p3_eval_msu.py --model experiments/hb_newdata/models/closed/elasticnet_model.joblib --tag closed
python experiments/hb_newdata/p3_eval_msu.py --model experiments/hb_newdata/models/mean/elasticnet_model.joblib   --tag mean
```

Contoh keluaran:

```
MSU-250 (mean)   MAE  12.50 g/L (1.25 g/dL) | RMSE 16.8 | R2 +0.52 | Pearson +0.72
```

**Pembanding baseline**: `MAE 14,93 g/L · R² 0,485 · Pearson 0,698`.
Model baru dikatakan lebih baik jika **MAE < 14,93** dan/atau **R² > 0,485**.
Prediksi tersimpan di `outputs/p3_predict_{tag}.csv`.

---

## STEP 3 — Pasang model terbaik ke aplikasi (P4)

Pilih varian dengan hasil terbaik (biasanya `mean`), lalu:

```bash
# dry-run dulu (tidak mengubah apa pun)
python experiments/hb_newdata/p4_integrate.py --model-dir experiments/hb_newdata/models/mean

# jika sudah yakin, eksekusi
python experiments/hb_newdata/p4_integrate.py --model-dir experiments/hb_newdata/models/mean --apply
```

- Menyalin model ke `core/models/seg_runtime/` (dipakai otomatis oleh
  `NailHbModel`, tanpa ubah kode).
- Folder lama di-backup dulu ke `core/models/seg_runtime_backup_<timestamp>/`.

---

## ✅ Hasil tercapai (tidak perlu diulang)

| Item | Hasil |
|---|---|
| P2 retrain fitur (open / closed / mean) | CV MAE 15,60 / 15,70 / **15,40 g/L** |
| P3 lintas-domain MSU-250 | fitur baru kalah (20,67–30,44) vs baseline **14,93 g/L** |
| C4 crop kuku (11.563 foto, seg26, margin 1.6) | `outputs/crops.csv` |
| C4 CNN ResNet18 test | **MAE 14,45 g/L (1,44 g/dL)** · per-pasien **13,93** · R² 0,286 |
| Kualitas seg26 (V2 test 702 foto, mask GT) | deteksi 99,6% · Dice 0,839 · IoU 0,748 — **retrain TIDAK perlu** |

---

## FASE 2 — CNN akurasi (c6/c7)

### C6 — Training CNN v2 (koreksi warna + rebalance + TTA)

```bash
python experiments/hb_newdata/c6_train_cnn_v2.py --arch resnet18 --epochs 25 \
    --colornorm whitep98 --rebalance --tta --tag resnet18_v2 --device 0
```

- Split pasien seed 42 **sama dengan C4** → MAE bisa dibandingkan langsung.
- Opsi: `--colornorm {none,whitep98,grayworld}`, `--arch efficientnet_b0`.
- Output: `models/cnn/resnet18_v2/{best.pt,model_metadata.json,test_predictions.csv}`.
- Durasi: ±35–50 menit (RTX 2050).

### C7 — Evaluasi ensemble (CNN + model fitur)

```bash
python experiments/hb_newdata/c7_eval_ensemble.py --tta
```

- Fitur `mean` dilatih hanya di pasien **train**, CNN di-infer di val+test.
- Bobot ensemble di-tuning di **val**, dilaporkan di **test**.
- Lapor juga subgrup per-device (SAMSUNG_A05/A14/S24) & tanpa kutek/henna.

---

## FASE 3 — Tahan-lintas-domain & keputusan aplikasi (c11–c14)

Domain aplikasi = **background bebas + kartu ArUco** (bukan kertas putih).
Kesimpulan fase: **runtime baseline (fitur) dipertahankan** — bukti di bawah.

### C11 — CNN robust / mix (augmentasi kuat + data free-bg)

```bash
# varian 1: augmentasi kuat, sewa saja (MSU = holdout jujur)
python experiments/hb_newdata/c11_train_cnn_robust.py --arch resnet18 --epochs 25 \
    --colornorm whitep98 --rebalance --tta --tag resnet18_robust --device 0
# varian 2: augmentasi kuat + 200 foto MSU ikut training (50 MSU sisa = eval)
python experiments/hb_newdata/c11_train_cnn_robust.py --arch resnet18 --epochs 25 \
    --colornorm whitep98 --rebalance --tta --mix-msu --tag resnet18_mix --device 0
```

- Augmentasi kuat: ColorJitter 0,35 · grayscale 5% · GaussianBlur 10% · RandomErasing 25%
  (simulasi variasi cahaya/kamera foto bebas — analogi synthetic data HEMO-AI 2024).
- `--mix-msu`: 200 foto MSU masuk train, **50 MSU di-holdout** (disimpan di
  `model_metadata.json` → `msu_test_ids`).
- Evaluasi MSU: `python experiments/hb_newdata/c9_eval_msu_cnn.py --tags resnet18_robust resnet18_mix --tta`
  (mix hanya boleh dievaluasi di subset holdout → tambah `--exclude-mix-tag resnet18_mix`).

**Hasil (25 epoch, split identik):**

| Varian | Test sewa per-pasien | MSU (holdout) |
|---|---|---|
| `resnet18_robust` | 14,66 g/L | 28,28 g/L (50 holdout) / 29,24 (250) |
| `resnet18_mix` | 14,01 g/L | **14,65 g/L** (50 holdout) |
| pembanding: baseline fitur | — | **12,24 g/L** (50 holdout sama) |

Simpulan: **augmentasi kuat tidak menolong lintas-domain** (robust lebih buruk
dari v2 26,47); **mix data domain target bekerja** (28,28 → 14,65). Runtime
baseline tetap terbaik di foto bebas → keputusan tidak berubah.

### C12 — Rekalibrasi & koreksi bias (literatur Mannino 2018/2025, Bihar 2023)

```bash
python experiments/hb_newdata/c12_personalize.py --tag resnet18_v2
```

- 30% pasien test = kohort kalibrasi (1 CBC/pasien) → 70% sisanya dievaluasi.
- **Hasil (v2, sewa test): global offset/linear & per-device TIDAK membantu**
  (13,80 → 13,86–14,33 g/L). Bias model dalam-domain ≈ 0 → tidak ada yang dikoreksi.
- Temuan penting: prediksi CNN **menyusut ke tengah** (range 37–117 g/L vs true
  18–192) → di cutoff 120 g/L semua orang terdeteksi anemia (sens 1,00 / spec 0,00).

### C13 — Evaluasi lensa screening (semua kandidat)

```bash
python experiments/hb_newdata/c13_screening_eval.py
```

| Model | Domain | MAE | Pearson | AUC@120 | sens/spec@120 |
|---|---|---|---|---|---|
| fitur (runtime) | **MSU (free-bg)** | **14,93** | **0,698** | **0,864** | **0,72 / 0,83** ✅ |
| fitur (runtime) | sewa | 15,68 | 0,444 | 0,611 | 0,98 / 0,03 |
| CNN v2 | sewa | 13,59 | 0,618 | 0,698 | 1,00 / 0,00 |
| CNN v2 | MSU | 26,47 | 0,411 | 0,667 | 1,00 / 0,00 |

- sens/spec 1,00/0,00 di sewa = efek kohort miring (84% anemia), bukan kehebatan.
- **Di domain aplikasi, runtime fitur adalah satu-satunya kandidat yang layak
  screening** (AUC 0,864). CNN (domain kertas putih) TIDAK boleh masuk runtime.

### C14 — Nilai kohort kalibrasi lokal di free-bg (MSU)

```bash
python experiments/hb_newdata/c14_msu_calibrate.py
```

- Baseline runtime di MSU sudah **tanpa bias** (−0,00 g/L, slope 1,08) — diduga
  memang dilatih dengan data MSU.
- Kohort kalibrasi 10–125 CBC: MAE tidak membaik (14,93 → 15,3–16,9), hanya AUC
  tipis 0,864 → 0,892. Simpulan: rekalibrasi bernilai hanya untuk populasi yang
  BENAR-BENAR baru (kulit/kamera beda), tidak bisa dikuantifikasi dari data ini.

### Lever nyata untuk aplikasi (dari literatur)

| Lever | Efek (literatur) | Implementasi |
|---|---|---|
| **Personalisasi** (user input 1 CBC → kalibrasi offset per-user) | MAE 1,36 → **0,57–0,74 g/dL** (Mannino PNAS 2025) | ✅ **sudah diimplementasikan** (lihat C15) |
| Data lapangan free-bg+ArUco+Hb | retrain lokal: ±4,43 → ±2,25 g/dL (Bihar 2023) | protokol `data/full_hand/README.md` |
| Filter kualitas input (tolak kutek/kabur) | menstabilkan deteksi | runtime guard |

### C15 — Personalisasi per-user (✅ implementasi selesai)

```bash
python experiments/hb_newdata/c15_personalize_sim.py   # bukti: 15,99 -> 5,79 g/L
```

Alur aplikasi (backend menyimpan satu JSON profil per `user_id`):

```python
from core.inference import NailHbModel
from core.personalize import Personalizer

model = NailHbModel(profile_path=f"profiles/{user_id}.json")   # otomatis muat profil

# 1) pengukuran biasa (belum personalisasi)
res = model.predict(img, boxes, white_source="chart")
#    -> res["raw_hb_g_dl"], res["personalized"]=False

# 2) pasien pulang bawa hasil CBC lab -> ajarkan sekali
model.calibrate(actual_g_dl=9.8)                  # pakai raw prediksi terakhir
model.personalizer.save(f"profiles/{user_id}.json")  # simpan per-user

# 3) kunjungan berikutnya: otomatis terkoreksi (offset 1 titik / linear >=2 titik)
res = model.predict(img2, boxes, white_source="chart")
#    -> res["estimated_hb_g_dl"] terkalibrasi, res["personalized"]=True
```

- Tanpa profil → perilaku model lama **identik** (kompatibel mundur, teruji).
- `raw_hb_g_dl` selalu disertakan → hasil libur personalisasi tetap tersedia.
- Clip keluaran [0,5–25] g/dL; ≥2 titik kalibrasi → regresi linear (persis
  nutrisi personalisasi serial Sanguina).

---

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `CUDA out of memory` | Tutup aplikasi berat; atau pakai `--device cpu` di p0/p1 saja (tidak perlu untuk step 1–3) |
| `assert HB out of g/L range` | Tidak akan terjadi — step 1 sudah memfilter Hb ke [30,200] |
| `missing columns NAIL_...` | Pastikan jalankan dari folder `anemia-app` |
| Hasil CV buruk (R² mendekati 0) | Wajar jika run `--limit` kecil; jalankan tanpa `--limit` |

Setelah step 3 selesai, kabari aku hasil RINGKASAN CV + angka MSU-250-nya,
nanti aku bantu analisis dan update dokumentasi.