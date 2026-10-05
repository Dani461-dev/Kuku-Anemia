"""
Import dataset `sewa-rural-care/anemia-survey-dataset` dari Hugging Face.

Sebelum menjalankan script ini, lakukan setup akses satu kali:
  1. Login akun di https://huggingface.co dan minta akses di
     https://huggingface.co/datasets/sewa-rural-care/anemia-survey-dataset
     (isi form Data Access Agreement & Submit).
  2. Buat token Read di https://huggingface.co/settings/tokens
  3. Autentikasi terminal:  hf auth login  (paste token)

Kalau skip, akan muncul error 401 Unauthorized / GatedRepoError.

Jalankan:  python3 import_hf_dataset.py

Error "gated dataset ... ask for access" berarti salah satu dari:
  a) Belum login HF: ketik  hf auth login   (atau huggingface-cli login di Windows)
     lalu paste token. Token dibuat di https://huggingface.co/settings/tokens
     (pilih "Read").
  b) Belum di-approve untuk dataset ini: cek
     https://huggingface.co/datasets/sewa-rural-care/anemia-survey-dataset
     (isi form Data Access Agreement & Submit, tunggu approval pemilik).
"""

from datasets import load_dataset

SPLIT = "train"
STREAMING = False


def check_auth():
    from huggingface_hub import whoami

    try:
        user = whoami()
        print(f"Terautentikasi sebagai: {user['name']}")
    except Exception:
        raise SystemExit(
            "Belum login ke Hugging Face.\n"
            "Jalankan:\n"
            "  hf auth login\n"
            "lalu paste token Read (buat di https://huggingface.co/settings/tokens).\n"
        ) from None


def main():
    check_auth()

    if STREAMING:
        ds = load_dataset(
            "sewa-rural-care/anemia-survey-dataset", split=SPLIT, streaming=True, token=True
        )
        row = next(iter(ds))
        print(row["patient_uuid"], row["hemoglobin_category"])
        return

    try:
        ds = load_dataset("sewa-rural-care/anemia-survey-dataset", split=SPLIT, token=True)
    except Exception as exc:
        msg = str(exc)
        if "gated" in msg.lower() or "access" in msg.lower():
            raise SystemExit(
                "Dataset masih gated / belum di-approve.\n"
                "1) Pastikan sudah login: hf auth login\n"
                "2) Minta akses di https://huggingface.co/datasets/sewa-rural-care/anemia-survey-dataset\n"
                "   (isi form Data Access Agreement, tunggu approval dari pemilik).\n"
            ) from None
        raise
    print(ds)

    import pandas as pd

    df = ds.to_pandas()
    print("Kolom:", df.columns.tolist())
    print("Distribusi hemoglobin_category:")
    print(df["hemoglobin_category"].value_counts())


if __name__ == "__main__":
    main()