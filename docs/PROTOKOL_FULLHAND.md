# PROTOKOL FULL-HAND — Pengumpulan Dataset Runtime

Untuk menutup **gap domain** antara pelatihan (dataset MSU, tangan terpotong) dan
runtime aplikasi (tangan penuh, background bebas), perlu dikumpulkan subset foto
tangan penuh dengan Hb lab — dipakai untuk retrain/rekalibrasi model.

---

## Mengapa perlu?

- Fitur model saat ini dilatih dari crop layout MSU (tangan 3 jari terpotong).
- Di aplikasi, MediaPipe menghasilkan crop dengan **pose/ukuran/posisi berbeda** → ada risiko pergeseran fitur.
- Data full-hand + Hb lab memungkinkan model belajar fitur yang **setara dengan runtime nyata**.

---

## Protokol pengumpulan

| Aspek | Standar |
|---|---|
| Jumlah subjek | Minimal 50, ideal 100 (makin banyak makin stabil) |
| Hb lab | **Wajib**: ambil darah vena (standard lab) **sama hari** dengan pengambilan foto. |
| Kartu warna | Cetak `core/chart_output/chart.png` (A4). Masukkan dalam frame setiap foto. |
| Posisi foto | Tangan **terbuka penuh**, telapak ke atas atau netral; semua kuku terlihat. Pastikan **kartu warna** terlihat penuh. |
| Pencahayaan | Hindari kilau langsung di kuku; cahaya sekitar merata. |
| Kondisi kuku | Bersih (tanpa kutek/acrylic/plester). Jika ada, catat di metadata. |
| Format foto | JPG/PNG, resolusi cukup (≥640×480). |
| Banyak foto | 1 foto per subjek cukup untuk baseline; ideal: 2 (kiri+kanan) sebagai augmentasi. |
| Split | Bagi pasien (bukan foto) menjadi train/val/test, hindari leakage. |

---

## Format metadata

Kolom (lihat template: `data/full_hand_template.csv`):

| Kolom | Tipe | Keterangan |
|---|---|---|
| `PATIENT_ID` | int/str | ID unik, harus cocok dengan nama foto |
| `Hb_LAB_GperL` | float | Hasil lab Hb dalam g/L (40–200) |
| `NAIL_BOUNDING_BOXES` | JSON | `[[top,left,bottom,right], ...]` 3 jari — **opsional** (bisa diisi otomatis lalu QA) |
| `SKIN_BOUNDING_BOXES` | JSON | `[[top,left,bottom,right], ...]` 3 jari — **opsional** |
| `GENDER` | str | `female` atau `male` |
| `PREGNANT` | bool/0-1 | `0/1` — threshold WHO berbeda untuk ibu hamil |
| `MEASUREMENT_DATE` | str | Tanggal tes Hb |
| `N_IMAGE` | int | Banyak foto per subjek (default 1) |

> Box opsional: jika dikosongkan, `core/train_fullhand.py` akan menjalankan MediaPipe
> otomatis → box perlu **QA manual** sebelum training (inspeksi visual via
> `core/validate_mediapipe.py`).

---

## Alur kerja (setelah data ada)

```bash
# 1) Taruh foto di data/full_hand/{PATIENT_ID}.jpg
# 2) Isi metadata di data/full_hand_metadata.csv (copy dari template)

# 3) Validasi deteksi + box + mask (opsional tapi disarankan):
python3 core/validate_mediapipe.py \
    --input-dir data/full_hand \
    --out-csv data/output/fullhand_detection_check.csv

# 4) Retrain model:
python3 core/train_fullhand.py --features-csv <setelah build_dataset selesai>
# (atau jalankan pipeline penuh yang otomatis: build + train)
```

---

## Evaluasi

| Metrik | Target |
|---|---|
| CV MAE (full-hand) | ≤ 1.8 g/dL (menurut gap terhadap MSU) |
| Prediksi vs lab selisih individual | < 2 g/dL untuk mayoritas pasien |
| Domain-shift flag (validate_mediapipe) | fitur dalam rentang training ≥ 80% |

Bandingkan metrik MSU vs full-hand di laporan → bukti penutupan gap.

---

## Keterbatasan etika

- **Informed consent** wajib untuk pengambilan foto + data kesehatan.
- Anonimisasi ID, tidak ada wajah identitas.
- IRB jika diperlukan institusi/kompetisi.
- Data hanya untuk riset/kompetisi, tidak untuk diagnosa klinis.