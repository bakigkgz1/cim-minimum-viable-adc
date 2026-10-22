# Plumbing: read_noise is not on the accuracy path

`macro.py` `simulate_array`:

```python
if self._cim_args.hardware:
    out = self.ADC_output(input, weight, weight_min, j)  # read_noise lives here
else:
    out = torch.matmul(input.to(torch.float32), weight.to(torch.float32))
out = self._adc_quant(out)
```

`read_noise` is applied only inside `ADC_output` (conductance `torch.normal`).  
`stuck_at_fault_sample` runs **before** that branch, on the digital weight tensor.

full_rerun_v1 Acc uses `hardware=0` → identical Acc at σ=0.1 and 0.2 is a no-op, not a robustness finding.

Stuck-at is a real accuracy-path effect, but the **magnitude is fault-map dependent**.
Report **median + min–max** over 10 maps (not mean±std):
PR64/ADC5 **67.97 (40.23–79.30)**; PR128/ADC8 **73.14 (53.32–80.66)**; cliff **9.57 (8.01–9.77)**.
Seed 1234's 73.83% was one draw.

To measure read_noise: `hardware=1` (see `COLAB_P8_HW1_READNOISE.py`).
