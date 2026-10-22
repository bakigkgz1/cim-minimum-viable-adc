# Paste in Colab (Drive already mounted). Prints the locked C100 histogram dumps.
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/NeuroSim15v_paper3")
OUT = DRIVE / "full_rerun_v1"
files = [
    "diag_cliff_summary.csv",
    "diag_pred_hist_PR64_ADC5.csv",
    "diag_pred_hist_PR128_ADC5.csv",
]

# Keep a copy next to the full_rerun campaign (does not overwrite if already there).
OUT.mkdir(parents=True, exist_ok=True)
for name in files:
    src = DRIVE / name
    dst = OUT / name
    if src.exists() and not dst.exists():
        dst.write_bytes(src.read_bytes())
        print(f"copied -> {dst}")

for name in files:
    p = DRIVE / name
    print(f"\n==== {name} bytes={p.stat().st_size} ====")
    print(p.read_text())
