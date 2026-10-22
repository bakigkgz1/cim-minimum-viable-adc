# === AUDIT: were nested-quote cells actually OK? ===
# Paste in Colab. Checks Drive scripts + CSVs from prior locks.
from pathlib import Path
import csv

DRIVE = Path("/content/drive/MyDrive/NeuroSim15v_paper3")

SCRIPTS = [
    "adc_clip_calib_pr72_88.py",
    "vgg8_clip_adc6.py",
    "adc_clip_grid_v2.py",
    "adc_clip_vs_pr.py",
    "blok5_multiseed.py",
]
CSVS = [
    "adc_clip_calib_pr72_88.csv",
    "vgg8_clip_adc6.csv",
    "adc_clip_grid_v2.csv",
    "adc_clip_vs_pr.csv",
    "pr128_adc6_lock.csv",
    "pr128_adc7_hw.csv",
    "blok2d_energy_breakdown.csv",
    "blok2d_single_batch_hw.csv",
    "blok5_multiseed.csv",
]

print("=== SCRIPTS (must be real Python, not mostly comments) ===")
for name in SCRIPTS:
    p = DRIVE / name
    if not p.exists():
        print(f"MISSING  {name}")
        continue
    lines = p.read_text(errors="ignore").splitlines()
    code = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    comments = [ln for ln in lines if ln.lstrip().startswith("#")]
    has_import = any(ln.lstrip().startswith("import ") or ln.lstrip().startswith("from ") for ln in code)
    has_def = any(ln.lstrip().startswith("def ") for ln in code)
    verdict = "OK" if has_import and len(code) > 20 else "SUSPECT"
    print(f"{verdict:8} {name}: lines={len(lines)} code={len(code)} comment={len(comments)} import={has_import} def={has_def}")
    if verdict != "OK":
        print("  head:", lines[:8])

print("\n=== CSVS (spot-check key numbers) ===")
EXPECT = {
    "adc_clip_calib_pr72_88.csv": [("pr", "72"), ("acc", "91.7969")],
    "vgg8_clip_adc6.csv": [("adc", "6"), ("acc", "91.7969")],
    "blok2d_energy_breakdown.csv": [("tag", "PR64_ADC5"), ("tops_w", "28.0011")],
    "pr128_adc7_hw.csv": [("adc", "7"), ("tops_w", "20.0431")],
    "pr128_adc6_lock.csv": [("adc", "6")],
}

for name in CSVS:
    p = DRIVE / name
    if not p.exists():
        print(f"MISSING  {name}")
        continue
    with p.open() as f:
        rows = list(csv.DictReader(f))
    print(f"OK-FILE {name}: nrows={len(rows)} cols={list(rows[0].keys())[:8] if rows else []}")
    if name in EXPECT and rows:
        # show first row keys of interest
        r0 = rows[0]
        print(f"         sample: { {k:r0.get(k) for k in list(r0)[:6]} }")

print("\n=== CROSS-CHECK: energy TOPS/W must match Blok2d ===")
e = DRIVE / "blok2d_energy_breakdown.csv"
b = DRIVE / "blok2d_single_batch_hw.csv"
if e.exists() and b.exists():
    with e.open() as f: er = {row["tag"]: row for row in csv.DictReader(f)}
    with b.open() as f:
        br = {}
        for row in csv.DictReader(f):
            tag = row.get("tag") or f"PR{row.get('pr')}_ADC{row.get('adc')}"
            br[tag] = row
    for tag in ["PR64_ADC5", "PR128_ADC5", "PR64_ADC8", "PR128_ADC8"]:
        if tag in er:
            tw = er[tag].get("tops_w")
            print(f"  {tag}: energy_csv tops_w={tw}")
else:
    print("  (one of the CSVs missing — skip)")

print("\nDone. SUSPECT scripts => that cell paste was bad; re-run with flat .py.")
print("OK scripts + matching CSVs => those locks are trustworthy.")
