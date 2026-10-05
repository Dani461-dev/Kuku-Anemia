#!/usr/bin/env python3
# =============================================================================
#  download_and_restore.py
#
#  Downloads the anemia-survey-dataset from Hugging Face and converts it back
#  into the original per-participant folder structure — one shard at a time
#  by default (8 participants, ~1.1 GB), the same unit of work used
#  throughout this pipeline's own convert/upload scripts. This keeps peak
#  disk usage minimal; raise --batch-size if you have disk/bandwidth to
#  spare and want fewer, larger downloads instead.
#
#  Requires common.py and parquet_to_folders.py in the same folder (all three
#  are published together under scripts/ in the dataset repo).
#
#  USAGE:
#    # Recommended default: one shard (8 participants, ~1.1 GB) at a time —
#    # deletes each shard's file after converting it, so peak disk usage
#    # never exceeds a single shard
#    python3 download_and_restore.py --out ./restored
#
#    # Just try it on the first couple of shards first
#    python3 download_and_restore.py --out ./restored --max-batches 2
#
#    # Fewer, larger downloads instead (less overhead, more disk needed)
#    python3 download_and_restore.py --out ./restored --batch-size 25
#
#    # Keep the downloaded shard files instead of deleting them after conversion
#    python3 download_and_restore.py --out ./restored --keep-shards
# =============================================================================

import argparse
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from parquet_to_folders import restore_patient

REPO_ID = "sewa-rural-care/anemia-survey-dataset"


def list_shard_files() -> list:
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    return sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))


def process_batch(shard_files: list, shard_dir: Path, out_root: Path, keep_shards: bool) -> int:
    n_participants = 0
    for remote_path in shard_files:
        local_path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset",
            filename=remote_path, local_dir=str(shard_dir),
        )
        table = pq.read_table(local_path)
        df = table.to_pandas()
        for _, row in df.iterrows():
            restore_patient(row.to_dict(), out_root)
            n_participants += 1

        if not keep_shards:
            Path(local_path).unlink()

    return n_participants


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Output directory for restored participant folders")
    ap.add_argument("--batch-size", type=int, default=1,
                     help="Shards per batch (~8 participants/shard). Default 1 shard "
                          "(~8 participants, ~1.1 GB) — matches the pipeline's own "
                          "per-shard unit of work and keeps disk usage minimal. Raise "
                          "this for fewer, larger downloads if you have the disk/bandwidth.")
    ap.add_argument("--max-batches", type=int, default=0,
                     help="Stop after N batches (0 = process everything). Useful for trying it out first.")
    ap.add_argument("--keep-shards", action="store_true",
                     help="Don't delete downloaded shard files after converting them.")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    shard_dir = Path("./_shards_tmp")
    shard_dir.mkdir(exist_ok=True)

    print("Listing all shard files in the dataset ...")
    all_shards = list_shard_files()
    total_shards = len(all_shards)
    print(f"Found {total_shards} shard files (~{total_shards * 8} participants)")

    batches = [all_shards[i:i + args.batch_size] for i in range(0, total_shards, args.batch_size)]
    if args.max_batches > 0:
        batches = batches[: args.max_batches]
        print(f"--max-batches applied: processing {len(batches)} of {len(batches)} batches")

    total_participants = 0
    for i, batch in enumerate(batches, 1):
        print(f"\nBatch {i}/{len(batches)}: downloading + converting {len(batch)} shard(s) ...")
        n = process_batch(batch, shard_dir, out_root, args.keep_shards)
        total_participants += n
        print(f"Batch {i}/{len(batches)} done — {n} participants restored "
              f"(running total: {total_participants})")

    if not args.keep_shards:
        shutil.rmtree(shard_dir, ignore_errors=True)

    print(f"\nDone. Restored {total_participants} participants to {out_root}/")


if __name__ == "__main__":
    main()
