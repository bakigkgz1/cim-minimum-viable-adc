# Paste in Colab (Drive already mounted). Does NOT re-run inference.
# Original 2026-08-25 dumps lived at Drive ROOT, not full_rerun_v1/.
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/NeuroSim15v_paper3")
NEEDLES = [
    "diag_cliff_summary.csv",
    "diag_pred_hist_PR64_ADC5.csv",
    "diag_pred_hist_PR128_ADC5.csv",
    "fig6_cliff_collapse_REAL.png",
    "diag_cliff_preds.py",
]

print("DRIVE exists:", DRIVE.exists(), DRIVE)
print("DRIVE is_dir:", DRIVE.is_dir() if DRIVE.exists() else False)
print()

print("=== exact names under Drive root + full_rerun_v1 ===")
for name in NEEDLES:
    for base in (DRIVE, DRIVE / "full_rerun_v1"):
        p = base / name
        print(f"  {p.exists()!s:5}  {p}")

print("\n=== recursive glob under NeuroSim15v_paper3 (diag_* / fig6*) ===")
hits = []
if DRIVE.exists():
    for pat in ("**/diag_*", "**/fig6*", "**/*cliff*", "**/*pred_hist*"):
        hits.extend(DRIVE.glob(pat))
seen = set()
for p in sorted(hits, key=lambda x: str(x)):
    if p in seen or not p.is_file():
        continue
    seen.add(p)
    print(f"  {p.stat().st_size:8d}  {p.relative_to(DRIVE)}")

if not seen:
    print("  (no matches — files are gone from this Drive folder)")
    print("\nNext: run COLAB_RESTORE_C100_HIST.py (~8–12 min).")
else:
    print("\n=== dump matching CSVs ===")
    for p in sorted(seen):
        if p.suffix.lower() == ".csv":
            print(f"\n==== {p} ====")
            print(p.read_text())
