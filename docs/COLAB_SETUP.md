# Colab ordered cells

Runtime: **T4 GPU**. Empty notebook. One cell at a time; paste output here.

## Cell 1 — Drive + clone

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone --depth 1 -b 2DInferenceV1.5-dev https://github.com/neurosim/NeuroSim.git /content/NeuroSim15v
%cd /content/NeuroSim15v
DRIVE='/content/drive/MyDrive/NeuroSim15v_paper3'
!mkdir -p {DRIVE}/results {DRIVE}/models {DRIVE}/scripts
!nvidia-smi -L
print('Next: Cell 1b — upload Paper3_NeuroSim15v.zip')
```

## Cell 1b — upload paper3 zip

```python
from google.colab import files
import shutil
from pathlib import Path
uploaded = files.upload()  # Paper3_NeuroSim15v.zip from Desktop
!unzip -q -o Paper3_NeuroSim15v.zip -d /tmp/p3
src = Path('/tmp/p3/Paper3_NeuroSim15v')
dest = Path('/content/NeuroSim15v/paper3_resnet')
dest.mkdir(parents=True, exist_ok=True)
for name in ['scripts','results','FEASIBILITY.md','COLAB_SETUP.md','RUN_ORDER.md','README.md']:
    p = src/name
    if p.is_dir():
        if (dest/name).exists(): shutil.rmtree(dest/name)
        shutil.copytree(p, dest/name)
    elif p.is_file():
        shutil.copy2(p, dest/name)
!ls -la /content/NeuroSim15v/paper3_resnet/scripts/train_resnet18_cifar10.py
!cp -r /content/NeuroSim15v/paper3_resnet/. /content/drive/MyDrive/NeuroSim15v_paper3/
```

## Cell 2 — torch + quantization

```python
%cd /content/NeuroSim15v
!pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cu121
%cd pytorch-quantization
!pip -q install -e .
%cd /content/NeuroSim15v
import torch; print(torch.__version__, torch.cuda.is_available())
```

## Cell 3 — train checkpoint

```python
%cd /content/NeuroSim15v
!python -u paper3_resnet/scripts/train_resnet18_cifar10.py --epochs 100 --batch_size 128 --min_acc 90
!cp models/resnet18_cifar10.pth /content/drive/MyDrive/NeuroSim15v_paper3/models/
!ls -lh models/resnet18_cifar10.pth
```

## Cell 4 — Gate0 A (hardware=0)

```python
%cd /content/NeuroSim15v
!python -u inference.py --dataset cifar10 --model resnet18 --hardware 0 --ppa 0 \
  --num_batches 2 --batch_size 128 --adc_precision 8 \
  --sub_array "[128,128]" --parallel_read 128 \
  --mem_type resistive --mem_states_file mem_states.csv \
  --data_path ./datasets/ --model_path ./models/
```

## Cell 5 — Gate0 B (CIM PR=128)

```python
!python -u inference.py --dataset cifar10 --model resnet18 --hardware 1 --ppa 0 \
  --num_batches 2 --batch_size 64 --adc_precision 8 \
  --sub_array "[128,128]" --parallel_read 128 \
  --mem_type resistive --mem_states_file mem_states.csv \
  --data_path ./datasets/ --model_path ./models/
```

## Cell 6 — Gate0 C (CIM PR=32)

```python
!python -u inference.py --dataset cifar10 --model resnet18 --hardware 1 --ppa 0 \
  --num_batches 2 --batch_size 64 --adc_precision 8 \
  --sub_array "[128,128]" --parallel_read 32 \
  --mem_type resistive --mem_states_file mem_states.csv \
  --data_path ./datasets/ --model_path ./models/
```
