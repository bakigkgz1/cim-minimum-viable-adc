# Paste in Colab (Drive already mounted). Prints + downloads the FULL
# P2b dump (all hooked layers, nclip/nelem filled). Not a new run.
from pathlib import Path
from google.colab import files

p = Path("/content/drive/MyDrive/NeuroSim15v_paper3/full_rerun_v1/P2b_clip_by_layer.csv")
print("exists", p.exists(), "bytes", p.stat().st_size if p.exists() else None)
if p.exists():
    text = p.read_text()
    print("lines", text.count("\n") + (0 if text.endswith("\n") else 1))
    print(text)
    files.download(str(p))
