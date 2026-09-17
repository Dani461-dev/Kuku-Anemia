# data/full_hand/

Folder untuk **dataset full-hand** (T3) — foto tangan penuh dari pengguna nyata.

## Cara pakai
1. Cetak kartu warna: `core/chart_output/chart.png` (A4).
2. Foto SUBJEK dengan:
   - tangan **penuh** terlihat jelas,
   - **kartu warna ArUco** dalam frame (untuk white-normalisasi),
   - background**bebas** (tidak harus putih),
   - dalam waktu dekat dengan pengambilan darah lab (Hb bisa berubah).
3. Simpan foto di sini dengan nama `{PATIENT_ID}.jpg` (mis. `1001.jpg`).
4. Isi metadata di `data/full_hand_metadata.csv` (template: `data/full_hand_template.csv`).
5. Jalankan `core/train_fullhand.py` untuk retrain model full-hand.

## Catatan
- Dataset ini adalah cara menutup gap domain model (fitur MSU layout terpotong vs runtime full-hand).
- Box kuku/kulit opsional di metadata: jika kosong, akan dihasilkan otomatis oleh MediaPipe lalu diperiksa manual (QA).