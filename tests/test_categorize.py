#!/usr/bin/env python3
"""Sanity tests untuk core/categorize.py (ambang WHO)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.categorize import categorize_hb, threshold_g_dl  # noqa: E402


def main() -> int:
    # ── threshold ─────────────────────────────────────────────
    assert threshold_g_dl("female") == 12.0, "wanita -> 12.0"
    assert threshold_g_dl("male") == 13.0, "pria -> 13.0"
    assert threshold_g_dl("pria") == 13.0, "alias pria"
    assert threshold_g_dl("m") == 13.0, "alias m"
    assert threshold_g_dl("") == 12.0, "default -> wanita"

    # ── categorisasi ───────────────────────────────────────────
    assert categorize_hb(11.9, "female") == "Rendah (Anemia)"
    assert categorize_hb(12.0, "female") == "Normal"   # threshold = batas normal
    assert categorize_hb(12.9, "male") == "Rendah (Anemia)"
    assert categorize_hb(13.0, "male") == "Normal"
    assert categorize_hb(10.0, "pria") == "Rendah (Anemia)"  # alias gender

    # ── edge cases ─────────────────────────────────────────────
    assert categorize_hb(None, "female") == "Tidak dapat diklasifikasikan"
    assert categorize_hb(float("nan"), "male") == "Tidak dapat diklasifikasikan"

    print("OK: categorisasi & threshold WHO lolos")
    return 0


if __name__ == "__main__":
    sys.exit(main())