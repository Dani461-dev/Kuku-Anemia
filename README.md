# Kuku-Anemia — Estimasi Hb dari Foto Kuku (Anevia)

Repo wadah untuk proyek **Anevia**: skrining anemia kasar dari foto kuku —
segmentasi kuku dengan YOLO26-seg, estimasi kadar hemoglobin (Hb, g/dL) dengan
ensemble CNN ResNet18, plus pipeline pembanding berbasis 42 fitur persentil RGB.

> **Alat skrining, bukan pengukuran lab.** Galat rata-rata ±1,2 g/dL pada data
> validasi — tidak cukup untuk membedakan anemia dari normal pada satu pasien.

Kode inti ada di **`anemia-app/`**. Folder itu adalah **repo git terpisah di
dalam repo ini**, jadi isinya (termasuk README teknis dan kode pipeline) tidak
ikut tampil di halaman GitHub ini — buka `anemia-app/README.md` di mesin lokal
untuk dokumentasi lengkap.

## Isi repositori

| Path | Isi |
|---|---|
| `anemia-app/` | **Kode inti (repo git terpisah).** Pipeline produksi CNN `experiments/hb_white_bg/`, training segmentasi `experiments/yolo26_seg/`, pipeline pembanding 42 fitur `core/`, data `data/`, protokol `docs/` |
| `import_anemia_dataset.py` | Impor `sewa-rural-care/anemia-survey-dataset` dari Hugging Face **tanpa download penuh** (±648 GB): pyarrow column-pruning + remote range-read via `HfFileSystem`, jadi total transfer hanya beberapa GB. Resume-friendly per shard |
| `import_hf_dataset.py` | Versi sederhana: `load_dataset` langsung (butuh login + akses dataset) |
| `hugging face.ipynb` | Notebook eksplorasi yang sama (instalasi, inspeksi kolom, opsi streaming) |
| `hf_cache/` | `common.py` (pembersih kolom/nilai kosong, slugify nama kolom) dan `download_and_restore.py` (unduh per shard lalu kembalikan ke struktur folder per pasien) |
| `laporan/` | `build_laporan.js` — skrip Node.js yang menyusun tabel/bagian laporan Word (`cd laporan && npm i docx && node build_laporan.js`); hasilnya `laporan_dataset_kuku_v3.docx` |
| `weights/yolo26n.pt` | Bobot base YOLO26-nano untuk training segmentasi kuku |
| `_meta_tmp/`, `__pycache__/` | Artefak: status resume impor per shard dan cache bytecode |

## Akses dataset

Dataset latih **`sewa-rural-care/anemia-survey-dataset`** di Hugging Face
bersifat *gated* (butuh persetujuan akses):

1. Minta akses di <https://huggingface.co/datasets/sewa-rural-care/anemia-survey-dataset>
   (isi Data Access Agreement, tunggu approval).
2. Buat token **Read** di <https://huggingface.co/settings/tokens>.
3. Login sekali di terminal: `hf auth login`.
4. Jalankan `python import_anemia_dataset.py` (kolom + foto kuku/mata/tongue,
   resume bila terputus).

Dataset uji eksternal: [biophotonics-msu/photo-haemoglobin](https://github.com/biophotonics-msu/photo-haemoglobin) (MIT).

## Menjalankan pipeline

Semua perintah ada di `anemia-app/README.md`. Ringkasnya:

```bash
# Inferensi Hb dari satu foto (latar putih/terang)
python anemia-app/experiments/hb_white_bg/w5_infer.py foto.jpg --gender female --age 30

# Pipeline pembanding 42 fitur (bukan model produksi); --input-dir punya default
python anemia-app/core/run_seg_pipeline.py --white auto --gender female
```

Training dan inferensi memakai Python 3.11 dengan torch + CUDA; `anemia-app/.venv`
adalah venv Linux sehingga tidak bisa dipakai di Windows.

## Catatan repo

- `anemia-app/` ter-commit sebagai *gitlink* (repo di dalam repo) tanpa
  `.gitmodules`, dan commit di dalamnya belum di-push — jadi di GitHub folder
  itu muncul sebagai tautan yang tidak bisa dibuka, bukan berisi file.
- `anemia-app-backup-20261005.bundle` (backup git ±600 MB) sengaja di-*ignore*.
- Folder data besar (`anemia-app/data/*/images/`) tidak ikut digit karena
  ukurannya; provenansinya tercatat di `anemia-app/docs/DATA_REGISTRY.md`.
- `_meta_tmp/` dan `__pycache__/` adalah hasil eksekusi, bukan sumber.
