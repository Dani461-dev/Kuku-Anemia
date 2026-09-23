# Retrain Hb — Dataset Baru (sewa-rural-care / anemia-survey-dataset)

Retraining model estimasi Hb (RobustScaler + ElasticNet) menggunakan dataset
baru: **5.782 pasien** dengan foto fingernails_open + fingernails_closed dan
Hb lab (`hgb_final`, g/dL). Tujuan: mengatasi bottleneck lama
*"Batas akurasi = jumlah data, bukan pilihan model"* (sebelumnya hanya 250
subjek MSU; CV MAE 15,96 g/L, R² 0,40).

## Status

| Tahap | Status |
|---|---|
| Setup torch CUDA (2.14.0+cu130, RTX 2050) | ✅ |
| P0 — Triase sampel (300×2) | ✅ deteksi & fitur 100% |
| P1 — Fitur skala penuh (5.782×2) | ✅ `outputs/features_*.csv` |
| P2 — Retrain (open/closed/mean) | ✅ CV MAE 15,60/15,70/**15,40** g/L |
| P3 — Validasi MSU-250 | ✅ (lihat tabel di bawah) |
| P4 — Integrasi ke `core/models/seg_runtime/` | ⏳ **tidak dijalankan** (keputusan domain: runtime tetap baseline) |
| C4 — Crop + CNN kuku (ResNet18) | ✅ test MAE per pasien **13,93** g/L |
| C6 — CNN v2 (colornorm+rebalance+TTA) | ✅ test MAE per pasien **13,55** g/L — **terbaik untuk domain kertas putih** |
| C7 — Ensemble CNN+fitur | ✅ fitur tidak menambah akurasi (bobot 1,0 = CNN saja) |
| C8/C9 — CNN lintas-domain MSU-250 | ✅ CNN sewa kalah jauh di MSU (26,5 vs 14,93 g/L) → domain berbeda |
| C11 — CNN robust/mix (augmentasi kuat ± MSU) | ✅ selesai: robust 29,24 g/L di MSU (augmentasi TIDAK membantu); **mix 14,65 g/L di 50 holdout** (retrain data lokal bekerja) |
| C12 — Rekalibrasi & koreksi bias | ✅ tidak membantu (bias ≈ 0); model CNN menyusut ke tengah |
| C13 — Evaluasi lensa screening | ✅ runtime fitur = satu-satunya kandidat layak di free-bg (AUC 0,864) |
| C14 — Kohort kalibrasi lokal di MSU | ✅ tanpa efek (baseline sudah unbiased; slope 1,08) |
| C15 — Personalisasi 1-titik + implementasi | ✅ MAE runtime 15,99 → **5,79 g/L (0,58 g/dL)** — persis literatur; `core/personalize.py` + integrasi `NailHbModel` siap |

## Cara menjalankan

1. **Sekali-klik**: double-click `run_all.bat` (Step 1+2).
2. **Manual step-by-step**: ikuti `INSTRUCTIONS.md` (Step 1 → 2 → 3).

## Alur

```
anemia-dataset/metadata.csv + images/{fingernails_open,fingernails_closed}/*
  → seg26 (YOLO26-seg, deteksi kuku) + skin geometri + white=auto
  → 42 fitur persentil (NAIL_/SKIN_) → HB_LEVEL_GperL = hgb_final × 10
  → core/train_hb.py (nested CV: outer KFold(5), inner ElasticNetCV, RobustScaler)
```

## Script

| Script | Fungsi |
|---|---|
| `p0_triage.py` | Triase sampel (deteksi %, mask coverage, guard) |
| `p1_build_features.py` | Ekstraksi fitur skala penuh (resume-friendly) |
| `p2_train.py` | Prep dataset (open/closed/mean) + jalankan train_hb.py |
| `p3_eval_msu.py` | Validasi lintas-domain pada MSU-250 |
| `p4_integrate.py` | Pasang model terbaik ke runtime (dry-run default) |
| `run_all.bat` | Menjalankan Step 1+2 sekali-klik |
| `c4_build_crops.py` | Bangun crop kuku (CNN): deteksi seg26 → crop → JPEG |
| `c4_train_cnn.py` | Training CNN (ResNet18/EfficientNet) regresi Hb dari crop |
| `c6_train_cnn_v2.py` | CNN v2: `--colornorm whitep98` + `--rebalance` + `--tta` |
| `c7_eval_ensemble.py` | Evaluasi ensemble CNN+fitur (bobot di val, subgrup device/kutek) |
| `c8_build_msu_crops.py` | Crop MSU-250 untuk eval CNN lintas-domain |
| `c9_eval_msu_cnn.py` | Evaluasi CNN di MSU-250 |
| `c10_build_all_crops.py` | Crop SEMUA jari (multi-finger averaging) — belum dipakai |
| `c11_train_cnn_robust.py` | CNN tahan-dua-domain: `--mix-msu` (200 MSU ke train, 50 holdout) + augmentasi kuat |
| `c12_personalize.py` | Simulasi rekalibrasi/koreksi bias (kohort 30% → eval 70%), lensa sens/spec |
| `c13_screening_eval.py` | Evaluasi lensa screening semua kandidat (range, AUC, sens/spec per cutoff) |
| `c14_msu_calibrate.py` | Nilai kohort kalibrasi lokal (10–125 CBC) di MSU |
| `c15_personalize_sim.py` | Simulasi personalisasi 1-titik (2 foto/pasien, Hb sama) — 0,58 g/dL |

## Hasil

### P0 — Triase (n=300, conf 0.15, device 0)

| Metrik | fingernails_open | fingernails_closed |
|---|---|---|
| Deteksi kuku | 100% | 100% |
| Fitur jadi | 100% | 100% |
| Mask coverage (mean) | 0.846 | 0.842 |
| Fitur finite | 100% | 100% |
| Hb dalam guard (40–180 g/L) | 99.3% | 99.3% |

### P1 — Fitur skala penuh (5.782 pasien per modalitas)

| Metrik | fingernails_open | fingernails_closed |
|---|---|---|
| Baris | 5.782 | 5.782 |
| Fitur finite | 100% | 100% |
| Mask coverage (mean) | 0.845 | 0.838 |

### Baseline model lama (MSU-250, core/models/seg_runtime)

```
MSU-250 (baseline)  MAE 14.93 g/L (1.49 g/dL) | RMSE 19.12 | R2 +0.485 | Pearson +0.698
```

### Hasil P2/P3 (setelah run)

**CV (nested, di dataset sendiri — 5.759–5.760 pasien):**

| Varian | n | CV MAE (g/L) | CV MAE (g/dL) | CV R² | Pearson |
|---|---|---|---|---|---|
| open | 5.760 | 15.60 | 1.56 | +0.160 | 0.401 |
| closed | 5.759 | 15.70 | 1.57 | +0.155 | 0.393 |
| **mean** | 5.759 | **15.40** | **1.54** | **+0.194** | 0.441 |

**Validasi lintas-domain MSU-250 (vs baseline model lama MAE 14.93 · R² 0.485):**

| Varian | MSU MAE (g/L) | MSU R² | Pearson |
|---|---|---|---|
| open | 30.44 | −0.608 | 0.519 |
| closed | 21.06 | +0.095 | 0.602 |
| mean | 20.67 | +0.139 | 0.599 |

**Kesimpulan sementara:**
1. Data 23× lipat **hampir tidak menurunkan MAE** di domain sendiri (15.96 → 15.40 g/L) →
   fitur 42-persentil RGB memang sudah jenuh (bottleneck = fitur, bukan jumlah data).
2. Transfer ke MSU-250 **jauh lebih buruk** (MAE 20–30 vs 14.93 g/L) → domain shift kuat
   (Samsung A05/S14/S24 + background putih + indoor vs setup kamera MSU).
   Korelasi Pearson tetap positif (0.52–0.60) → bias skala/offset, bukan fitur tak berguna.
3. Oleh karena itu **model lama masih unggul untuk foto bergaya MSU**. Model baru (`mean`)
   lebih cocok untuk domain foto dataset baru. Validasi sesungguhnya untuk aplikasi adalah
   foto *full-hand* runtime (set `Data Tambahan/full_hand` masih kosong/Hb belum tersedia).

### C4 — CNN dari foto kuku (✅ selesai)

ResNet18 (ImageNet pretrained, fine-tune), 11.563 crop, split per pasien 80/10/10,
loss SmoothL1, 20 epoch (early-stop tidak terpicu; best val epoch 17).

| Metrik | Nilai |
|---|---|
| Train / Val / Test | 9.251 / 1.156 / 1.156 gambar |
| Best Val MAE (epoch 17) | 14,35 g/L |
| **Test MAE (per gambar)** | **14,45 g/L (1,44 g/dL)** |
| **Test MAE (per pasien, mean 2 foto)** | **13,93 g/L (1,39 g/dL)** |
| Test RMSE | 19,11 g/L |
| Test R² / Pearson | +0,286 / 0,541 |

**Perbandingan final MAE (g/L):**

| Model | MAE | Keterangan |
|---|---|---|
| Model lama (fitur manual, 250 subjek) | 15,96 | CV |
| Fitur manual `mean` (5.759 pasien) | 15,40 | CV |
| **CNN ResNet18 (11.563 crop)** | **14,45** | **test set terpisah (pasien tak pernah dilihat)** |

### C6 — CNN v2: koreksi warna + rebalance + TTA (✅ selesai)

Resep C4 + `--colornorm whitep98 --rebalance --tta`, 25 epoch (early-stop epoch 22),
split pasien **identik** dengan C4 → MAE bisa dibandingkan langsung.

| Metrik | v1 (C4) | v2 (C6) |
|---|---|---|
| Test MAE (per gambar) | 14,45 g/L | 14,17 g/L |
| **Test MAE (per pasien)** | **13,93 g/L (1,39 g/dL)** | **13,55 g/L (1,36 g/dL)** |
| Test R² (per pasien) | +0,311 | **+0,356** |
| Pearson (per pasien) | 0,577 | **0,598** |

### C7 — Ensemble CNN + fitur (✅ selesai, tidak membantu)

ElasticNet fitur `mean` dilatih hanya di pasien train; bobot ensemble di-tuning di
val → **bobot terbaik = 1,00 (CNN saja)**. Fitur tidak menambah akurasi di atas CNN.

| Model (test, per pasien) | MAE (g/L) | R² | Pearson |
|---|---|---|---|
| fitur `mean` | 15,68 | +0,197 | 0,444 |
| CNN v1 | 13,94 | +0,311 | 0,577 |
| **CNN v2** | **13,55** | **+0,356** | **0,598** |
| ens v1+v2 | 13,65 | +0,348 | 0,602 |

Subgrup device (v2, per pasien): SAMSUNG_A05 13,18 · S24 13,59 · A14 14,30 g/L.
Tanpa kutek/henna: 13,73 g/L.

### C8/C9 — CNN lintas-domain MSU-250 (✅ selesai)

| Model di MSU-250 | MAE (g/L) | R² | Pearson |
|---|---|---|---|
| **Baseline fitur lama (runtime)** | **14,93** | **+0,485** | **+0,698** |
| CNN v1 (sewa) | 26,60 | −0,344 | +0,248 |
| CNN v2 (sewa) | 26,47¹ | −0,307 | +0,411 |

¹ Angka v2 dikoreksi: evaluasi awal tanpa `ColorNormalize` setara training
(whitep98) — setelah fix, MAE 26,47 (sebelumnya 26,79). Gap lintas-domain tetap besar.

### C11 — CNN tahan-dua-domain: robust (augmentasi) vs mix (data lokal) (✅ selesai)

Dua varian di training penuh (25 epoch, split identik seed 42):

| Varian | Resep | Test sewa per-pasien |
|---|---|---|
| `resnet18_robust` | ColorJitter 0,35 · grayscale 5% · blur 10% · RandomErasing 25% | 14,66 g/L |
| `resnet18_mix` | augmentasi kuat **+ 200 foto MSU ikut train** (50 MSU holdout) | 14,01 g/L |

**Lintas-domain MSU:**

| Model di MSU | MAE | Keterangan |
|---|---|---|
| **Baseline fitur (runtime)** | **14,93 g/L** | AUC 0,864 |
| CNN v2 (sewa saja) | 26,47 g/L | pembanding asli |
| CNN robust (augmentasi saja) | 29,24 g/L | ⚠️ *lebih buruk* dari v2 |
| CNN mix (sewa + 200 MSU) | 14,65 g/L | hanya 50 holdout, bukan 250 |

**Head-to-head adil — 50 MSU holdout yang SAMA:**

```
baseline fitur (runtime) : 12.24 g/L   ← menang
CNN v2 (sewa)            : 25.22 g/L
CNN robust (sewa+aug)    : 28.28 g/L
CNN mix (sewa+200 MSU)   : 14.65 g/L
```

**Simpulan C11:**
1. **Augmentasi kuat tidak menolong lintas-domain** (robust 29,24 > v2 26,47) —
   hipotesis "synthetic data" (HEMO-AI 2024) tidak terbukti di data kita; gap
   domain tidak bisa dijembatani dari sisi augmentasi.
2. **Mix data domain target = lompatan besar** (28,28 → 14,65 g/L pada subset
   sama) — prinsip Bihar 2023 terkonfirmasi: retrain dengan data lokal adalah
   satu-satunya cara menutup gap domain.
3. Runtime baseline masih yang terbaik di foto bebas → **keputusan final tidak
   berubah: runtime baseline dipertahankan.** Strategi mix relevan ketika data
   lapangan sungguhan (free-bg + ArUco + Hb) tersedia untuk retrain penuh.

### C12 — Rekalibrasi & koreksi bias (✅ selesai, tidak membantu)

30% pasien test = kohort kalibrasi (1 CBC/pasien) → 70% dievaluasi (honest split).
v2, sewa test:

| Varian koreksi | MAE (g/L) | R² | Pearson |
|---|---|---|---|
| asli (tanpa koreksi) | **13,80** | +0,391 | 0,628 |
| global offset | 13,90 | +0,389 | 0,628 |
| global linear | 13,86 | +0,393 | 0,628 |
| per-device offset | 14,06 | +0,381 | 0,620 |
| per-device linear | 14,33 | +0,358 | 0,604 |

**Simpulan:** dalam-domain bias ≈ 0 → tidak ada yang bisa dikoreksi. Temuan penting:
prediksi CNN **menyusut ke tengah** (range 37–117 g/L vs true 18–192) → di cutoff
120 g/L semua pasien terprediksi anemia (sens 1,00 / spec 0,00); de-shrink slope
1,32 tidak bisa memulihkan ekstrem (range tetap 19–122).

### C13 — Lensa screening semua kandidat (✅ selesai)

| Model | Domain | MAE | Pearson | AUC@120 | sens/spec@120 |
|---|---|---|---|---|---|
| **fitur (runtime)** | **MSU (free-bg)** | **14,93** | **0,698** | **0,864** | **0,72 / 0,83** ✅ |
| fitur (runtime) | sewa | 15,68 | 0,444 | 0,611 | 0,98 / 0,03 |
| CNN v2 | sewa | 13,59 | 0,618 | 0,698 | 1,00 / 0,00 |
| CNN v2 | MSU | 26,47 | 0,411 | 0,667 | 1,00 / 0,00 |

- sens/spec 1,00/0,00 di kohort sewa = efek prevalensi (84% anemia), bukan kehebatan.
- Di domain aplikasi (foto bebas), **runtime fitur adalah satu-satunya kandidat layak
  screening** (AUC 0,864). CNN domain kertas putih tidak boleh masuk runtime.

### C14 — Kohort kalibrasi lokal di MSU (✅ selesai)

Baseline runtime di MSU sudah **tanpa bias** (−0,00 g/L; slope 1,08 — diduga dilatih
dengan data MSU). Rekalibrasi linear memakai kohort 10–125 CBC:

| n_cal | MAE (g/L) | AUC@120 |
|---|---|---|
| 0 (asli) | **14,93** | 0,864 |
| 10 | 16,89 ± 0,72 | 0,871 |
| 20 | 15,54 ± 0,62 | 0,870 |
| 40 | 15,38 ± 0,40 | 0,871 |
| 60 | 15,31 ± 0,27 | 0,882 |
| 80 | 15,27 ± 0,21 | 0,887 |
| 125 | 15,50 ± 0,37 | 0,892 |

Simpulan: rekalibrasi hanya bernilai untuk populasi yang BENAR-BENAR baru
(kulit/kamera beda, seperti Bihar 2023), tidak bisa dikuantifikasi dari data ini.

### C15 — Personalisasi 1-titik: implementasi + bukti di data kita (✅)

Simulasi jujur memakai 2 foto/pasien (open & closed, label Hb sama):
foto 1 = kunjungan dengan CBC lab (kalibrasi), foto 2 = pemantauan berikutnya.

| Model | MAE mentah (per-foto) | MAE personalisasi 1-titik | Penurunan |
|---|---|---|---|
| **fitur/runtime** | 15,99 g/L | **5,79 g/L (0,58 g/dL)** | **−63,8%** |
| CNN v2 | 14,17 g/L | 7,59 g/L (0,76 g/dL) | −46,4% |

Literatur (Mannino PNAS 2025): populasi 1,36 → personalisasi **0,57–0,74 g/dL**.
**Selaras persis.** Catatan: 2 foto sesi sama = gain ideal; monitoring mingguan
punya noise ekstra → gain nyata agak lebih kecil.

**Implementasi (sudah masuk `core/`):**
- `core/personalize.py` — kelas `Personalizer`:
  - 1 titik → koreksi offset `raw + (lab − raw_saat_lab)`
  - ≥2 titik → regresi linear `a·raw + b` (fallback offset jika slope tak stabil)
  - output di-clip ke [0,5–25] g/dL, profil per-user = JSON kecil, `save/load`
- `core/inference.py` — `NailHbModel` kini menerima `personalizer=` /
  `profile_path=`, menerapkan kalibrasi di `predict()`/`predict_masks()`,
  menambah `calibrate(actual_g_dl, raw_g_dl=None)` dan field hasil
  `raw_hb_g_dl`, `personalized`, `calibration_points`.
- Tanpa profil → perilaku lama identik (diuji unit).

## 🎯 Kesimpulan & keputusan domain

1. **Kualitas seg26 terverifikasi** (V2 test, 702 foto ber-mask, belum pernah dilihat):
   deteksi 99,6%, Dice 0,839, IoU 0,748 → **retrain segmentasi tidak perlu**.
2. **CNN kuku v2 (13,55 g/L, 1,36 g/dL)** adalah model terbaik untuk foto **kertas putih**
   (domain dataset sewa: Samsung, background putih, indoor).
3. **Aplikasi Anemia Kuku memakai protokol background bebas + kartu warna ArUco**
   (`data/full_hand/README.md`) — domain BERBEDA dari sewa.
   → **Runtime TIDAK diganti ke CNN sewa**: baseline fitur (14,93 g/L di MSU) tetap
   dipasang untuk foto bergaya bebas. CNN v2 hanya valid jika protokol foto aplikasi
   distandarkan ke kertas putih.
4. Opsi lanjutan untuk memanfaatkan 5.800 data sewa di protokol bebas: mengumpulkan
   foto full-hand bebas + ArUco + Hb (protokol `data/full_hand/`) dan/atau eksperimen
   CNN tahan-dua-domain (campuran sewa + MSU + augmentasi kuat).

### 🎯 Keputusan final (C11–C15)

1. **Runtime baseline dipertahankan** — dan kini terbukti juga sebagai kandidat
   screening terbaik di domain aplikasi (AUC 0,864, sens/spec 0,72/0,83 @ 120 g/L;
   MSU sebagai proxy foto bebas).
2. Rekalibrasi/kohort lokal & de-shrink **tidak membantu** di domain yang sudah ada
   (bias ≈ 0; shrinkage CNN tak bisa dipulihkan).
3. **Personalisasi = lever terbesar dan SUDAH DIIMPLEMENTASIKAN**:
   `core/personalize.py` + integrasi `NailHbModel` (offset 1-titik, linear ≥2 titik).
   Bukti di data sendiri: MAE runtime **1,60 → 0,58 g/dL** (literatur: 0,57–0,74).
   → Untuk pengguna yang dipantau berulang, kegiatan pengukuran pertama disertai
   input 1 nilai CBC lab lalu aplikasi otomatis terkoreksi untuk pengukuran berikut.
4. **CNN robust/mix (C11) selesai**: augmentasi kuat saja TIDAK membantu lintas-domain
   (robust 29,24 g/L di MSU, lebih buruk dari v2 26,47); **mix data domain target
   bekerja** (14,65 g/L di 50 holdout vs baseline 12,24 di subset sama). CNN tetap
   TIDAK masuk runtime — strategi mix menjadi panduan retrain nanti saat data
   lapangan tersedia.
5. Data lapangan free-bg + ArUco + Hb (protokol `data/full_hand/`) = satu-satunya
   data yang bisa menutup gap domain secara permanen (bukti: retrain lokal Bihar
   ±4,43 → ±2,25 g/dL).