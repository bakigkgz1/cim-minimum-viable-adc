# GitHub issue draft — paste into neurosim/NeuroSim (branch 2DInferenceV1.5-dev)

**Repo:** https://github.com/neurosim/NeuroSim.git  
**Branch / commit:** `2DInferenceV1.5-dev` @ `9825ef4`  
**Title:** `[2DInferenceV1.5-dev] Param.cpp hardcodes levelOutput=128 independently of Python --adc_precision`

---

## Observation

In `NeuroSIM/Param.cpp` (around line 260) the multilevel sense-amp output range is hardcoded:

```cpp
levelOutput = 128;  // # of levels of the multilevelSenseAmp output, should be in 2^N forms; e.g. 32 levels --> 5-bit ADC
```

The comment already states the intended rule (`2^N` levels ↔ N-bit ADC). The assignment does not follow it: `levelOutput` is always 128 (ADC = 7 costing), regardless of the Python wrapper flag `--adc_precision`.

`NeuroSIM/main.cpp` currently receives from `pytorch_quantization/utils/hook.py`:

```
./NeuroSIM/main NetWork_<model>.csv  weight_precision  input_precision  sub_array  parallel_read  ...
```

ADC precision is not on that argv list, so the C++ PPA path cannot match the Python accuracy path.

## Impact (measured, ResNet18, PR=128, 2DInferenceV1.5-dev)

We re-ran hardware costing with `levelOutput = 2^ADC` before `make`. At PR=128, matched-LO Pipelined TOPS/W / area:

| ADC | levelOutput | TOPS/W | Chip area (µm²) |
|-----|-------------|--------|-----------------|
| 5   | 32          | 14.97  | 3.10×10⁸        |
| 6   | 64          | 26.98  | 3.41×10⁸        |
| 7   | 128 (stock) | 20.04  | 3.94×10⁸        |
| 8   | 256         | 13.41  | 4.79×10⁸        |
| 12  | 4096        | 1.17   | 3.04×10⁹        |

Matched ADC=12 vs matched ADC=5 is ~9.8× area and ~12.8× worse TOPS/W. With stock LO=128 (P6b, sha 446ce3087d02f7eb, PR=128), ADC=6/8/12 all report 20.04 TOPS/W and 3.94×10⁸ µm² (same printed Pipelined point as matched ADC=7); ADC=5 keeps that area/FPS but 11.51 TOPS/W. Unmatched costing therefore makes ADC=12 look identical to ADC=8 and hides the matched ADC=6 peak (26.98 TOPS/W). Component energy also changes sign with PR at matched LO (ADC dynamic energy ×2.47 when PR doubles at ADC=5, ×0.53 at ADC=8).

## Suggestion

Keep the comment’s `2^N` rule in code:

1. Pass ADC bits from Python (`hook.py` / `trace_command.sh`) into `NeuroSIM/main`.
2. In `Param.cpp` / `main.cpp`, set `levelOutput = 1 << adcBits` (or read an explicit `--levelOutput`).
3. Until that lands, document that hardware sweeps must patch `levelOutput` and rebuild, or rankings across ADC widths are not comparable.

Happy to send a small PR if you want the argv wired through.

---

After the issue is opened, put the number here: **#____**  
Then cite it in the manuscript (Methods / Limitations) as the upstream report of the default.
