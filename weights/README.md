# weights/

Base-weight untuk training/deteksi YOLO (opsional).

| File | Dipakai oleh | Status |
|---|---|---|
| `yolo11n.pt` | `experiments/yolo/train.py` (default `--model`) | sebagai base YOLO; **belum ada hasil fine-tune** |

> Catatan: `yolo26n.pt` lama dihapus (tidak pernah direferensikan).

Hasil fine-tune (jika nanti dijalankan) akan berada di:
`experiments/yolo/runs/detector/weights/best.pt`
dan dikonsumsi oleh `core/detectors.YOLODetector`.

Detektor YOLO **belum dilatih** — pipeline utama memakai box ground truth dari
`data/metadata.csv`. Lihat `experiments/yolo/train.py` dan `docs/PROTOKOL_NOTEBOOK.md`.