# Anemia Kuku (Anevia) — Estimasi Hb dari Foto Kuku

Estimasi kadar hemoglobin (Hb, g/dL) dari **foto kuku di atas latar putih/terang**.
Segmentasi kuku memakai YOLO26-seg; estimasi Hb memakai ensemble CNN (ResNet18)
yang digabung dengan metadata pasien, plus kalibrasi opsional per pengguna.

> **Ini alat skrining kasar, bukan pengukuran lab.** Galat rata-rata sekitar
> 1,2 g/dL (lihat [Performa](#performa)) — tidak cukup untuk membedakan anemia
> dari normal pada satu pasien.

Alur lengkapnya — dari data training sampai angka yang keluar di runtime — ada di
[Pipeline utama](#pipeline-utama-experimentshb_white_bg).

## Model yang dipakai (fiks)

**Ensemble CNN ResNet18 + stacker metadata**, kelas `WhiteBgHb` di
[`experiments/hb_white_bg/w5_infer.py`](experiments/hb_white_bg/w5_infer.py).
Bobot: 10 checkpoint (model A dan C, masing-masing 5-fold) + `stacker_meta.joblib`.

| Aspek | Nilai |
|---|---|
| Masukan | foto kuku latar putih/terang (+ opsional umur, jenis kelamin, status hamil) |
| Keluaran | `hb_g_dl` (+ `raw_hb_g_dl`, `status`: `ok` / `retake`) |
| MAE out-of-fold (5.257 pasien) | **1,19 g/dL** dengan metadata · 1,30 g/dL foto kuku saja |
| r · R² | 0,71 · 0,51 (dengan metadata) |
| Uji eksternal MSU-250 (250 subjek) | MAE **25,6 g/L** · r **0,58** (ensemble produksi A+C) |
| Bila latar/lampu bergeser | MAE 1,46 g/dL, r 0,55 (foto kuku saja) |

Jalur 42 fitur di `core/` **bukan model produksi**; ia dipertahankan sebagai pembanding (lihat
[Pipeline 42 fitur](#pipeline-42-fitur-core--pembanding-bukan-model-produksi)).

| Jalur | Lokasi | Domain | Status |
|---|---|---|---|
| **CNN latar putih** | `experiments/hb_white_bg/` | foto kuku, latar terang | **model produksi** |
| 42 fitur persentil (SVR) | `core/` | latar bebas + kartu ArUco | pembanding; akurasi lebih rendah (MAE 1,51 g/dL, r 0,45) |

> **Untuk integrasi backend:** panggil `WhiteBgHb` dari `w5_infer.py`. Jangan memakai
> `core.inference.NailHbModel` untuk estimasi final — ia memuat model 42 fitur secara default.

---

## Struktur

```text
anemia-app/
├── core/                       # pipeline pembanding 42 fitur + util bersama
│   ├── seg_detector.py         #   wrapper YOLO26-seg (dipakai kedua jalur)
│   └── pipeline.py             #   load_rgb dan util (dipakai kedua jalur)
├── experiments/
│   ├── yolo26_seg/             # training segmentasi kuku (YOLO26-seg) — AKTIF
│   ├── hb_white_bg/            # MODEL PRODUKSI: CNN Hb (w0 → w9); detail di README-nya
│   └── hb_models/              # benchmark model alternatif
├── archive/                    # kode mati — TIDAK dipakai, tidak dihapus
├── data/                       # semua data; lihat data/README.md
├── docs/                       # protokol & status pipeline
├── models/                     # bobot legacy (arsip)
├── weights/                    # catatan bobot base YOLO
├── tests/
└── requirements.txt
```

## Data

```text
data/
├── 10_train_anemia/     # training: 5.782 subjek, 11.564 foto kuku, Hb lab (hgb_final)
├── 20_test_msu250/      # test eksternal: 250 subjek (tidak pernah masuk training CNN)
├── 30_seg_training/     # NailSegmentationDatasetV2 untuk training segmentasi
├── 40_inference_samples/# 3 foto tangan penuh, uji runtime (tanpa label Hb)
└── 90_unused/           # gagal QC / di luar scope (tidak dihapus)
```

Rincian provenance: [`docs/DATA_REGISTRY.md`](docs/DATA_REGISTRY.md). Folder data
besar (`10_train_anemia/images/`, `30_seg_training/`, `90_unused/`) tidak ada di
git; `20_test_msu250/` ter-version karena itu satu-satunya test eksternal.

Dari 5.782 subjek, **5.277 subjek** lolos QC latar terang + deteksi kuku
(48.183 crop kuku); **5.257** di antaranya masuk training/validasi silang
(out-of-fold).

---

## Pipeline utama (`experiments/hb_white_bg/`)

Ada dua fase: **training (offline, sekali jalan)** dan **inferensi (runtime)**.
Runtime membangun crop dengan fungsi yang sama dengan training
(`w0_build_crops.py`), jadi preprocessing-nya identik.

### Fase training (offline)

```text
data/10_train_anemia       5.782 subjek · 11.564 foto kuku · Hb lab (hgb_final)
   → w0_build_crops.py     QC latar: 4 sudut gambar; lolos bila bg_gray ≥ 130 dan bg_p90 ≥ 170
                           → YOLO26-seg → crop margin 1,6 → JPEG q90 → manifest
   hasil                   5.277 subjek lolos QC · 48.183 crop · 5.257 masuk validasi silang
   → w4_robust_train.py    GroupKFold per patient_uuid (5 fold) → tiap fold satu model
                           A: --norm p98 · C: --norm skin --robust · ResNet18 160 px
   → 10 checkpoint         a_p98, a_p98_f1..f4, c_skin_rob, c_skin_rob_f1..f4
   → w8_final_stack.py     prediksi OOF dirata-ratakan per pasien, lalu stacker
                           (HistGB + Ridge) atas nail, umur, jenis kelamin, hamil
                           → models/stacker_meta.joblib

segmentasi (terpisah)      experiments/yolo26_seg pada NailSegmentationDatasetV2
                           → bobot seg26/weights/best.pt (mask mAP50 0,965, ~10 ms/foto)

data/20_test_msu250        TIDAK PERNAH masuk training — hanya uji eksternal
```

Resep training lengkap, angka per fold, dan pitfall:
[`experiments/hb_white_bg/README.md`](experiments/hb_white_bg/README.md).

### Fase inferensi (runtime)

```text
foto tangan/kuku (latar putih/terang)
   → QC latar              4 sudut gambar; gagal → status "retake"
   → YOLO26-seg            conf 0,15; mask + box tiap kuku (maks 6)
   → crop kuku             margin 1,6 → JPEG q90 (identik dengan training)
                           kuku terdeteksi < 2 → status "retake"
   → 2 jenis CNN ResNet18  5 model per jenis, TTA flip, rata-rata antar kuku
        A  normalisasi p98 (acuan latar)          ← paling akurat di latar putih
        C  normalisasi kulit + augmentasi shift   ← tahan perubahan lampu/latar
   → rata-rata A dan C     nail_only_g_dl (Hb dari foto kuku saja)
                           model_disagreement_g_dl = |A − C| (tanda ketidakpastian)
   → stacker metadata      bila ada umur + jenis kelamin: 0,5 HistGB + 0,5 Ridge
                           → raw_hb_g_dl
   → kalibrasi pengguna    bila ada profil: offset dari hasil lab sebelumnya (opsional)
                           → hb_g_dl
   → keluaran              status · n_nails · nail_only_g_dl · per_model_g_dl ·
                           model_disagreement_g_dl · raw_hb_g_dl · hb_g_dl · basis
```

**Kenapa dua jenis model.** A memakai latar putih sebagai acuan warna sehingga
paling akurat di latar putih, tapi rapuh begitu lampu/latar bergeser (r 0,69 →
0,39). C menormalkan warna kuku terhadap kulit jari yang sama dan dilatih dengan
augmentasi latar/pencahayaan, jadi stabil (r tetap 0,64 saat bergeser). Rata-rata
keduanya hampir sama akurat dengan A di latar putih dan jauh lebih stabil.

**Penolakan foto.** Jika latar tidak terang (`bg_gray < 130` atau `bg_p90 < 170`)
atau kuku yang terdeteksi kurang dari 2, hasilnya `status: "retake"` dan
pengguna diminta memotret ulang di atas kertas putih. Tidak ada angka Hb yang
dikeluarkan untuk foto seperti itu.

### Skrip

| Skrip | Fungsi |
|---|---|
| `w0_build_crops.py` | QC latar + YOLO26-seg → crop kuku + manifest |
| `w1_cv_train.py` | pustaka training: `make_model` (resnet18/34/50, efficientnet_b0, mobilenet_v3_small), loss, rebalance Hb |
| `w1f_featmodel.py` | model fitur tangan (ridge/HistGB/RF); pembanding, tidak dipakai |
| `w4_robust_train.py` | training CNN GroupKFold per pasien, evaluasi **clean** dan **shift**; flag eksperimen `--arch`, `--lr`, `--wd`, `--tail-weight`, `--seed` |
| `w7_stack.py` | uji stacking metadata di holdout satu fold |
| `w8_final_stack.py` | OOF 5-fold penuh + melatih dan menyimpan stacker |
| `w9_personalize.py` | kalibrasi per pengguna (offset konservatif; selftest: `python w9_personalize.py`) |
| `w5_infer.py` | inferensi ujung ke ujung (QC → crop → ensemble → stacker → kalibrasi) |
| `run_ablation.sh` | fold 0 baseline: `a_p98`, `c_skin_rob`, `b_skin`, `d_p98_rob` |
| `run_5fold.sh` | fold 1–4 kedua jenis model (training final) |
| `run_stage1.sh` | ablasi resolusi 224 + EfficientNet-B0 → `outputs/log_s1_*.txt`, `log_s2_*.txt` |
| `run_stage3.sh` | ablasi seed/arch/LR/weight-decay/tail-weight → `outputs/log_s3_*.txt` |
| `run_final.sh` | 1 model dari hampir seluruh data (`final_*`); **hanya eksperimen**, lebih buruk di data eksternal |

Bobot: `experiments/hb_white_bg/models/` — `a_p98`, `a_p98_f1..f4`, `c_skin_rob`,
`c_skin_rob_f1..f4` (10 checkpoint, dipakai `w5_infer.py`) dan
`stacker_meta.joblib`. Checkpoint percobaan lain di folder itu (termasuk
`final_a_p98`, `final_c_skin_rob`, `s1_*`, `s3_*`) tidak dipakai.
Bobot segmentasi: `experiments/yolo26_seg/runs/seg26/weights/best.pt`.

### Menjalankan

```bash
# Inferensi satu foto (umur & jenis kelamin opsional; tanpa itu hanya foto kuku)
python experiments/hb_white_bg/w5_infer.py foto.jpg --gender female --age 30
# tambah --pregnant bila hamil

# Kalibrasi per pengguna: simpan hasil lab (g/dL) untuk foto ini ke profil
python experiments/hb_white_bg/w5_infer.py foto.jpg --gender female --age 30 \
    --profile profiles/u1.json --lab 9.5
# kunjungan berikutnya memakai profil yang sama → hb_g_dl otomatis terkalibrasi
python experiments/hb_white_bg/w5_infer.py foto2.jpg --gender female --age 30 \
    --profile profiles/u1.json
```

Keluaran penting: `hb_g_dl` (final), `raw_hb_g_dl` (sebelum kalibrasi),
`nail_only_g_dl`, `per_model_g_dl` (Hb per jenis model), `model_disagreement_g_dl`
(selisih A vs C; makin besar makin tidak yakin), `n_nails`, `basis`, dan `status`
(`ok` atau `retake`).

### Melatih ulang

```bash
cd anemia-app

# 0. Crop kuku + manifest (butuh bobot seg26), ±7 menit
python experiments/hb_white_bg/w0_build_crops.py --source anemia
python experiments/hb_white_bg/w0_build_crops.py --source external

# 1. Training 5-fold, dua jenis model (±17 mnt per fold, GPU 4 GB cukup)
for k in 0 1 2 3 4; do
  python experiments/hb_white_bg/w4_robust_train.py --norm p98 \
      --folds 5 --only-fold $k --epochs 12 --imgsz 160 --tag a_p98$([ $k -gt 0 ] && echo _f$k)
  python experiments/hb_white_bg/w4_robust_train.py --norm skin --robust \
      --folds 5 --only-fold $k --epochs 12 --imgsz 160 --tag c_skin_rob$([ $k -gt 0 ] && echo _f$k)
done

# 2. OOF penuh + stacker metadata → models/stacker_meta.joblib
python experiments/hb_white_bg/w8_final_stack.py
```

Setara dengan skrip: fold 1–4 = `run_5fold.sh`; fold 0 juga dihasilkan
`run_ablation.sh` (skrip itu sekaligus melatih varian `b_skin` dan `d_p98_rob`
yang tidak dipakai). Training dan inferensi memakai Python sistem 3.11 (torch +
CUDA). `.venv/` di repo ini adalah venv Linux dan tidak bisa dipakai di Windows.

---

## Performa

Semua angka di tingkat **pasien**, prediksi **out-of-fold** (GroupKFold per pasien,
tidak ada pasien yang sama di training dan validasi), 5.257 pasien di
`10_train_anemia`, latar putih. Sumber angka: `outputs/final_5fold_report.json`
(dihasilkan `w8_final_stack.py`).

| Varian | MAE | RMSE | r | R² |
|---|---|---|---|---|
| Prediktor konstan (median) | 1,61 g/dL | 2,20 | — | −0,03 |
| Metadata saja | 1,59 | 2,10 | 0,26 | 0,07 |
| Model A saja | 1,27 | 1,64 | 0,69 | 0,43 |
| Model C saja | 1,35 | 1,75 | 0,64 | 0,35 |
| A + C | 1,30 | 1,68 | 0,69 | 0,41 |
| **A + C + metadata (dipakai)** | **1,18–1,19** | **1,53** | **0,71** | **0,51** |

Baris "Metadata saja" dihitung ulang untuk README ini: HistGB atas umur/jenis
kelamin/kehamilan tanpa citra, CV 5-fold acak pada 5.257 pasien, dari
`outputs/manifest_anemia.csv` → MAE 1,59 · RMSE 2,10 · r 0,26 · R² 0,07.
(Nilai lama yang beredar, r 0,31 · R² 0,10, tidak punya artefak tersimpan.)

**Tahan pergeseran domain** (crop yang sama, latar diganti + white balance/lampu
digeser secara deterministik; fold 0, 1.052 pasien; `outputs/oof_*.csv` kolom
`pred_shift`):

| | Latar putih (MAE, r) | Digeser (MAE, r) |
|---|---|---|
| A saja | 1,27 g/dL, 0,69 | 1,60 g/dL, 0,39 |
| C saja | 1,34 g/dL, 0,65 | 1,38 g/dL, 0,64 |
| A + C | 1,29 g/dL, 0,69 | 1,44 g/dL, 0,61 |

**Eksternal — `20_test_msu250` (250 subjek).** Prediksi dirata-ratakan per pasien
dari **ensemble yang sama dengan inferensi** (5 checkpoint A + 5 checkpoint C,
lihat `MEMBERS` di `w5_infer.py`); angka dihitung dari `outputs/external_*.csv`:

| | MAE | r |
|---|---|---|
| A ens (5 fold) | 24,4 g/L | 0,61 |
| C ens (5 fold) | 26,7 g/L | 0,51 |
| **A + C ens (dipakai)** | **25,6 g/L** | **0,58** |

Jangan baca MAE-nya sendirian: rata-rata Hb sumber ini 26 g/L lebih tinggi dari
data training dan tidak punya demografi, jadi sebagian besar galat adalah
pergeseran populasi. Yang bermakna adalah r (0,58).

Dua catatan: (1) angka lama di README, "MAE 25,4 g/L, r 0,58", berasal dari
satu checkpoint A (fold 0) saja; (2) model tunggal dari `run_final.sh` justru
lebih buruk di data ini — `final_a_p98` 27,8 g/L (r 0,44), `final_c_skin_rob`
26,5 g/L (r 0,43) — jadi **jangan** menggantikan ensemble 5-fold dengan model
"final" tunggal.

**Yang dicoba dan tidak membantu** (jangan diulang tanpa alasan baru):
resolusi 224 (A sedikit lebih baik di latar putih tapi jauh lebih rapuh saat
bergeser), EfficientNet-B0 (lebih buruk dari ResNet18), de-shrink/isotonic
(sudah tercakup stacker), median/min/max antar-kuku, memisah kuku open/closed,
skin tone sebagai fitur stacker, serta membuang foto "bermasalah" (galat tidak
terkonsentrasi di HP, kutek, atau sumber label tertentu).

**Upaya menaikkan akurasi CNN lebih jauh** (fold 0, 1.052 pasien; model A MAE 1,273 g/dL;
selisih diuji dengan *bootstrap* berpasangan, SE MAE tunggal ±0,033 g/dL).
Ablasi stage 3 dijalankan `run_stage3.sh`, log di `outputs/log_s3_*.txt`,
tahap awal di `run_stage1.sh`:

| Percobaan | MAE bersih | Catatan |
|---|---|---|
| Ensemble 3 seed model A | 1,260 | −0,013 g/dL, tidak bermakna |
| *Weight decay* 1e-2 | 1,253 | −0,019 g/dL (bermakna, tapi kecil); r sama |
| ResNet34 | 1,280 | tidak lebih baik; sangat rapuh saat bergeser (1,76) |
| *Learning rate* 2e-4 / 5e-5 (18 epoch) | 1,264 / 1,285 | dalam noise |
| Bobot ekor ×2 | 1,275 | tidak mengurangi prediksi mengerut |
| MobileNetV3-small | 1,338 | lebih buruk |
| Resolusi 224 (model A) | 1,258 | sedikit lebih baik di latar putih, lebih rapuh saat bergeser |
| Ensemble 3 seed + ResNet34 + MobileNet + C | 1,272–1,278 | tidak lebih baik dari A+C |
| CNN + prediksi 42 fitur (stacking) | 1,189 | sama dengan CNN + metadata saja (1,189) |

Semua varian berada dalam ±0,03 g/dL dari baseline: CNN sudah jenuh pada data ini. Ensemble
3 seed model A memang sedikit lebih baik di latar putih (1,260 vs 1,291 untuk A+C, selisih
bermakna), tetapi lebih rapuh saat lampu/latar bergeser (1,56 vs 1,44 g/dL), jadi ensemble A+C
dipertahankan.

---

## Kalibrasi per pengguna (`w9_personalize.py`)

Kebijakannya sengaja konservatif: **hanya offset**, dari rata-rata berbobot-recency
maksimal 3 hasil lab terakhir, dibatasi ±3 g/dL, hasil lab lebih tua dari 90 hari
tidak dipakai. Regresi slope dari 2–3 titik tidak dipakai (tidak stabil).
`core/personalize.py` yang lama (offset 1 titik + regresi linear ≥2 titik) tidak
dipakai di jalur ini.

Dasar pengukuran: galat model konsisten per orang (korelasi galat foto terbuka vs
mengepal 0,94), tetapi ~84% darinya adalah efek mengerut ke tengah yang bergantung
pada Hb asli; bias pribadi sejati hanya ±0,66 g/dL (SD). Model hanya merespons
sekitar 30% dari perubahan Hb yang nyata.

**Belum tervalidasi pada data serial nyata** — dataset hanya punya satu
pengukuran per pasien. Simulasi berbasis angka di atas memperkirakan offset masih
membantu selama Hb tidak berubah jauh sejak kalibrasi (≈1,3 → 0,8 g/dL untuk
perubahan 1 g/dL), tetapi itu perkiraan; klaim "MAE 0,6–0,7 g/dL" dari literatur
**belum terbukti** untuk model ini.

---

## Pipeline 42 fitur (`core/`) — pembanding, bukan model produksi

```text
foto → YOLO26-seg → mask kuku + kotak kulit (geometri) → jari tengah → white reference (auto)
     → 42 fitur persentil RGB (kuku 21 + kulit 21) → Pipeline sklearn → Hb (g/L → g/dL) → guard kewajaran
```

> Model ini lebih lemah pada data yang sama (MAE 1,51 vs 1,19 g/dL; r 0,45 vs 0,71).
> Estimasi Hb produksi memakai CNN (di atas).

Dilatih dari **gabungan Anemia-survey (11.519 foto, 5.760 pasien) + MSU-250** —
konsekuensinya **MSU-250 di jalur ini adalah data latih, bukan data uji eksternal**.
Bandingkan dengan jalur CNN, di mana MSU-250 murni data uji.

| File | Fungsi |
|---|---|
| `core/build_features_survey.py` | 42 fitur untuk foto survei lewat jalur runtime seg26 yang sama persis (resume-friendly) |
| `core/model_features.py` | `NailSkinRatios`: fitur turunan kuku/kulit di dalam Pipeline, sehingga `inference.py` tetap hanya memanggil `.predict` |
| `core/train_hb_combined.py` | benchmark 7 jenis model (GroupKFold per pasien), sapuan bobot MSU, simpan model terbaik |
| `core/models/seg_survey/` | model aktif (`elasticnet_model.joblib` + `model_metadata.json`; nama file dipertahankan agar kompatibel) |

`core/inference.py` memilih model dengan urutan: env `ANEVIA_HB_MODEL_DIR` → `seg_survey` →
`seg_runtime` → model kanonik. `core/guard.py` menghitung rentang fitur "wajar" dari
MSU **dan** survei.

```bash
python core/build_features_survey.py --device 0      # ±10 menit, GPU
python core/train_hb_combined.py                      # benchmark + latih + simpan ke core/models/seg_survey/
python core/run_seg_pipeline.py --input-dir <folder foto> --white auto --gender female
```

**Model terpilih: SVR (RBF) + fitur rasio kuku/kulit, bobot MSU = 10** (rata-rata MAE
per-foto di kedua sumber; `core/outputs/combined_benchmark.csv`):

| Model (bobot MSU 5) | MAE survei | MAE MSU | rata-rata |
|---|---|---|---|
| ElasticNet (model lama) | 15,93 g/L | 18,16 | 17,04 |
| Ridge + rasio | 15,96 | 17,49 | 16,73 |
| HistGB + rasio | 15,54 | 16,54 | 16,04 |
| RandomForest + rasio | 15,54 | 16,48 | 16,01 |
| **SVR + rasio** | **15,48** | **16,37** | **15,93** |

Dengan bobot MSU 10: survei 15,54 g/L, MSU 16,16 g/L (rata-rata 15,85). Bobot 1 / 3 / 30 lebih buruk.

**Hasil model terpilih** (out-of-fold): survei per foto 15,54 g/L (r 0,38) · survei
per pasien 15,10 g/L = **1,51 g/dL** (R² 0,18) · MSU-250 16,16 g/L (r 0,61).

**Baca dengan hati-hati.** Pada survei, model 42 fitur hanya sedikit lebih baik dari
menebak rata-rata (R² 0,18). Pada 5.257 pasien yang sama, CNN latar putih mencapai
MAE 1,30 g/dL dan r 0,69, melawan 1,51 g/dL dan r 0,45. Fitur persentil warna
menyimpan sinyal Hb jauh lebih sedikit daripada citra kuku utuh; kurva belajar
(R² 0,08 → 0,16 → 0,17 dari 250 → 4.760 pasien) mendatar, jadi penghambatnya
adalah fiturnya, bukan jumlah data. Keunggulan jalur 42 fitur hanya di
**populasi survei**: model ElasticNet MSU-saja meleset MAE 39 g/L dengan bias
−28 g/L dan r 0,13 di sana.

Pembatasan metode: tidak ada filter latar putih pada foto survei saat ekstraksi fitur
(berbeda dari jalur CNN), dan `ElasticNetCV` internal memakai KFold biasa sehingga dua foto
satu pasien bisa terpisah di lipatan dalam (hanya memengaruhi pemilihan hiperparameter).

---

## Keterbatasan

1. **Bukan pengukuran lab.** MAE ≈ 1,2 g/dL dan R² ≈ 0,5. Target MAE < 1 g/dL dari
   foto kuku saja **tidak realistis**: SD Hb di data ±2,2 g/dL dan sinyal kuku
   hanya cukup untuk r ≈ 0,7.
2. **Prediksi mengerut ke tengah.** Hb sangat rendah (< 7 g/dL) diprediksi ±1,8
   g/dL terlalu tinggi, dan Hb > 13 diprediksi ±1,9 g/dL terlalu rendah.
3. **Tidak ada label "anemia" yang dikeluarkan.** Di data ini 83% pasien anemia
   dan prediksi mengerut, sehingga pada cutoff WHO (wanita < 12, pria < 13 g/dL)
   sensitivitas 0,99 tetapi spesifisitas hanya 0,07 — hampir semua ditandai anemia.
4. **Hanya latar putih/terang.** Foto latar bebas ditolak, bukan diprediksi.
   Pergeseran lampu/kamera kecil masih tertangani model C, tetapi hanya diuji
   pada simulasi, bukan foto aplikasi nyata.
5. **Belum ada validasi lapangan.** Belum ada foto dari aplikasi asli dengan Hb lab.
   Tiga foto tangan penuh di `40_inference_samples/` tidak berlabel, jadi akurasi
   di kondisi nyata belum terukur. Inilah langkah berikutnya yang paling
   bernilai: kumpulkan foto latar putih dari aplikasi bersama hasil lab.
6. **Stacker butuh umur, jenis kelamin, status hamil** dari pengguna. Tanpa itu
   hasil hanya dari foto kuku (MAE ≈ 1,30 g/dL).
7. **Populasi.** Data training berasal dari satu kohort (83% anemia, tiga model
   HP Samsung); generalisasi ke populasi/HP lain belum diuji.
8. **Eksternal MSU tidak bisa dipersonalisasi** (tanpa demografi) dan berbeda
   populasi; baca berpasangan dengan r.
9. **`core/inference.py` memuat model 42 fitur secara default** (`seg_survey`), bukan CNN.
   Backend harus memanggil `WhiteBgHb` dari `w5_infer.py` agar memakai model produksi.
10. **`pytest` tidak terpasang**; `tests/` belum dijalankan di lingkungan ini.
    `w9_personalize.py` punya selftest sendiri (`python w9_personalize.py`).

---

## Referensi

- `experiments/hb_white_bg/README.md` — detail teknis model produksi (resep training, filter latar, pitfall)
- `docs/DATA_REGISTRY.md` — provenance setiap dataset
- `docs/PROTOKOL_NOTEBOOK.md` — kesesuaian `core/` dengan notebook asli
- `docs/PROTOKOL_FULLHAND.md` — protokol pengumpulan dataset full-hand
- `docs/STRATEGI_WHITE_BG_HB.md` — strategi awal pipeline CNN (angka lamanya
  sebagian sudah digantikan README ini)
- Dataset latih: Hugging Face `sewa-rural-care/anemia-survey-dataset`;
  dataset uji eksternal: [biophotonics-msu/photo-haemoglobin](https://github.com/biophotonics-msu/photo-haemoglobin) (MIT)

**Dokumen lama — jangan dipakai sebagai status pipeline saat ini:**

- `docs/PIPELINE_STATUS.md` (per 18 Sep 2026) — masih menyebut jalur 42 fitur
  sebagai "arsitektur final" dan `seg_runtime` sebagai model aktif; statusnya
  sudah digantikan CNN + `seg_survey`.
- `docs/ALUR_PROGRAM_LAPORAN.md` — alur untuk laporan, masih memakai jalur
  42 fitur ElasticNet dari `20_test_msu250` sebagai satu-satunya alur.
- `docs/NAIL_ANALYSIS_PIPELINE.md` — desain lama berbasis MediaPipe Hands.
- `docs/Anevia_Fusion_Calculation.md` — rencana fusion mata + kuku; model mata
  belum ada di repo ini.
