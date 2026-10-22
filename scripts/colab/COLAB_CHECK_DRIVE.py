# === Bağlantı kontrolü (Drive + runtime) — yeniden koşum YOK ===
from pathlib import Path
import os, time

print("cwd:", os.getcwd())
print("time:", time.strftime("%Y-%m-%d %H:%M:%S"))

mnt = Path("/content/drive")
print("\n=== mount ===")
print("/content/drive exists:", mnt.exists())
print("/content/drive/MyDrive exists:", (mnt / "MyDrive").exists())

if not (mnt / "MyDrive").exists():
    print("\nDrive bağlı DEĞİL. Çalıştır:")
    print("  from google.colab import drive")
    print("  drive.mount('/content/drive')")
else:
    # cheap I/O probe — hang here => stale mount
    t0 = time.time()
    listing = list((mnt / "MyDrive").iterdir())
    dt = time.time() - t0
    print(f"MyDrive listdir: {len(listing)} entries in {dt:.2f}s")
    if dt > 15:
        print("UYARI: Drive I/O yavaş / kopuk mount olabilir — Runtime > Disconnect, sonra yeniden mount.")

DRIVE = Path("/content/drive/MyDrive/NeuroSim15v_paper3")
OUT = DRIVE / "full_rerun_v1"
print("\n=== paper3 folder ===")
print("DRIVE exists:", DRIVE.exists(), DRIVE)
print("full_rerun_v1 exists:", OUT.exists())

print("\n=== known files (connection probe) ===")
probes = [
    DRIVE / "blok2d_Param_BASE.cpp",
    DRIVE / "models" / "resnet18_cifar100.pth",
    OUT / "P2b_clip_by_layer.csv",
    OUT / "P4_c100_full_acc.csv",
    OUT / "P5b_vgg8_full.csv",
]
ok = 0
for p in probes:
    ex = p.exists()
    sz = p.stat().st_size if ex else None
    print(f"  {ex!s:5}  {sz}  {p.name}")
    ok += int(ex)

print("\n=== NeuroSim runtime ===")
root = Path("/content/NeuroSim15v")
print("NeuroSim15v exists:", root.exists())
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch:", type(e).__name__, e)

print("\n=== verdict ===")
if not (mnt / "MyDrive").exists():
    print("KOPUK: Drive mount yok.")
elif not DRIVE.exists():
    print("MOUNT VAR ama NeuroSim15v_paper3 yok — yanlış hesap / klasör.")
elif ok == 0:
    print("KOPUK veya boş klasör: bilinen CSV/ckpt hiçbiri görünmüyor.")
elif ok < len(probes):
    print(f"KISMİ: {ok}/{len(probes)} probe OK. Drive bağlı; eksik dosyalar gerçekten yok olabilir.")
else:
    print("BAĞLI: Drive + full_rerun_v1 okunuyor. Histogram CSV ayrı aranabilir.")
