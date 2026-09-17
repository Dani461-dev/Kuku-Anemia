# models/

Penjelasan isi folder model.

## Struktur

```text
models/
├── legacy_74features/   # ARSIP — skema 74 fitur (percobaan lama, JANGAN dipakai)
└── (canonical model berada di core/models/, bukan di sini)
```

## Di mana model yang dipakai?

Model terlatih yang dipakai aplikasi ada di **`core/models/`**:

| File | Keterangan |
|---|---|
| `core/models/elasticnet_model.joblib` | Model KANONIK = **improved** (`--protocol nested`, masked, semua 250 data). MAE 1.60 g/dL. |
| `core/models/model_metadata.json` | Fitur order, metrik CV, ambang WHO, satuan. |
| `core/models/notebook_baseline/` | Reproduksi baseline asli `Usage Notes.ipynb` (untuk perbandingan laporan). |
| `core/models/improved_all250/` | Duplikat kanonik (cadangan). |

Alasan `legacy_74features/` diarsipkan: model lama mati (`n_nonzero_coef=0`) dan
fitur HSV/LAB terbukti memperburuk akurasi. Lihat `legacy_74features/README.md`.

Lihat juga [`docs/PROTOKOL_NOTEBOOK.md`](../docs/PROTOKOL_NOTEBOOK.md) untuk
perbandingan baseline vs improved.