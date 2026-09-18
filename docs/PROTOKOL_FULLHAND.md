# PROTOKOL FULL-HAND — Pengumpulan Dataset & Retrain Model Hb (T3)

Untuk menaikkan akurasi model Hb pada **runtime aplikasi** (foto tangan penuh), perlu
dikumpulkan dataset **foto full-hand + Hb lab**. Sumber foto tidak lagi harus
ber-box: region kuku/skin dideteksi otomatis oleh **YOLO26-seg**
(`experiments/yolo26_seg`, mAP mask 0.965; metadata tetap **lean** — hanya id & Hb).

---

## Mengapa perlu?

- Model Hb `seg_runtime` dilatih dari foto **MSU 3-jari terpotong** (punya Hb lab).
- Aplikasi runtime memotret **tangan penuh** → foto full-hand + lab membuat model
  belajar fitur pada domain runtime sebenarnya → MAE turun di aplikasi.

---

## Protokol pengumpulan

| Aspek | Standar |
|---|---|
| Jumlah subjek | Minimal **50**, ideal **100+** (makin banyak makin stabil) |
| Hb lab | **Wajib**: darah vena / lab standar **di hari yang sama** dengan foto. |
| Posisi foto | **Tangan penuh** terlihat jelas (5 jari + kuku tajam); 1 foto ↔ 1 subjek. |
| Kartu warna | **Opsional** — pipeline memakai `white=auto` (deteksi putih) bila kartu tak ada; jika kartu dipakai, konsisten antar foto. |
| Pencahayaan | Merata, tanpa kilau langsung di kuku (kuku tampak jelas). |
| Kondisi kuku | Bersih (tanpa kutek/acrylic/plester). Jika ada, catat di metadata. |
| Format | JPG/PNG, ≥640×480; nama `{PATIENT_ID}.jpg`. |
| Etika | Informed consent, anonim ID (tanpa wajah identitas), IRB jika perlu. |

---

## Format metadata lean (`data/full_hand_metadata.csv`)

Template: `data/full_hand_template.csv`

| Kolom | Tipe | Keterangan |
|---|---|---|
| `PATIENT_ID` | int | ID unik = nama file foto `{PATIENT_ID}.jpg` |
| `Hb_LAB_GperL` | float | Hasil lab Hb (g/L), rentang 30–200. **Wajib** |
| `GENDER` | str | `female` / `male` *(opsional)* |
| `PREGNANT` | bool | `0/1` *(opsional — ambang WHO ibu hamil)* |
| `MEASUREMENT_DATE` | str | Tanggal tes Hb *(opsional)* |
| `N_IMAGE` | int | Banyak foto per subjek (default 1) |

> Tidak ada kolom bounding box — region dideteksi otomatis (YOLO-seg).

---

## Alur setelah data ada

```bash
# 1) Foto  -> data/full_hand/{PID}.jpg
# 2) Isi metadata lean -> data/full_hand_metadata.csv (copy dari template)
# 3) Retrain model Hb (jalur YOLO-seg), membandingkan 2 strategi:
#      A) full-hand saja   B) MSU + full-hand (gabung)
python3 core/train_fullhand.py --conf 0.15 --device 0
```

`train_fullhand.py` otomatis:
- ekstraksi fitur runtime (YOLO-seg mask + skin geometri + `white=auto`),
- nested CV untuk kedua strategi (A dan B),
- menyimpan model CV-MAE terbaik → `core/models/fullhand_model.joblib` (+ metadata),
- alternatif kombinasi ke `core/models/fullhand_msu_combined.joblib`.

Deploy: gunakan model terbaik via env `ANEVIA_HB_MODEL_DIR` (atau salin ke
`core/models/` dan biarkan `inference._model_paths()` memilih).

---

## Evaluasi

| Metrik | Target |
|---|---|
| CV MAE (full-hand / gabung) | **Lebih kecil dari 15.96 g/L** = ada peningkatan vs `seg_runtime` |
| MAE per pasien (individu) | < 20 g/L untuk mayoritas |
| Bandingkan A vs B di laporan | bukti apakah data MSU membantu atau menghambat |

---

## Keterbatasan etika & disclaimer

- Informed consent wajib; anonim; data hanya untuk riset/kompetisi, bukan
  diagnosa klinis.
- Output tetap **AI-estimated Hb**, dikonfirmasi lewat tes klinis.