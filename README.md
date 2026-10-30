# Unit-Column Clipping Limits the ADC Resolution in Offset-Encoded Analog Compute-in-Memory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23573121.svg)](https://doi.org/10.5281/zenodo.23573121)

Code, measurement files and analysis scripts for the manuscript

> B. Gökgöz, *Unit-Column Clipping Limits the ADC Resolution in Offset-Encoded Analog Compute-in-Memory* (under review).

Version 1.3.0 adds the runs S8–S13 behind the current manuscript: clipping split into data columns and offset terms, exact (digital) offset removal, 256-row arrays, the unit-column/offset-row ablation, layer-resolved feedback with exact offset, hardware costs of the digital-offset design, two independently trained ResNet18 checkpoints, VGG8 on CIFAR-100 as a held-out fourth pair, code noise with exact offset, static and dynamic converter errors, and a differential (W⁺/W⁻) mapping with its first-order cost. Earlier versions (v1.0–v1.2) accompanied previous drafts that calibrated the clipping budget with the analog unit column; their files are kept and are still used in the Supplementary Material.

Every number in the manuscript comes from the files in `results/`. The scripts in `scripts/colab/` produced them; `analysis/unit_column/run_all.py` rebuilds Figs. 1–5 and Tables 2–4 and S3–S4 from them, and `analysis/unit_column/s12.py` rebuilds Table 5 and Tables S17–S18.

## What the study shows

NeuroSim shifts inputs and weights to unsigned values and removes the shift with an offset row and a **unit column** whose cells all conduct. The unit column's count is the number of active inputs, which bounds every data-column partial sum of the same read.

- **The unit column clips first.** Near the accuracy boundary the offset terms, under 1% of conversions, clip 35–8500× more often per conversion than a data column. Converting only the unit column exactly removes the loss; converting only the offset row (zero after a ReLU) changes no prediction.
- **Feedback.** With the analog unit column, clipping raises later layers' mean partial sums by up to 51%; with exact offset terms every layer clips its open-loop share (< 0.2% shift).
- **One bit saved.** With exact offset terms, 5 bits serve PR = 128 (128-row arrays) and 6 bits PR = 256 (256-row arrays) on CIFAR-10, over three calibration seeds and three trained ResNet18 checkpoints; VGG8/CIFAR-100 needs 6 instead of 7 bits at PR = 128.
- **Cost.** A popcount (NeuroSim adder-tree model: 0.27 pJ, 343 µm² at PR = 128) adds ≈ 0.2% chip energy; efficiency rises 1.22–1.35× (flash-type MLSA) and 1.08–1.13× (SAR).
- **Converter errors.** With static column offsets, code-transition errors and thermal noise of 0.075 LSB per source the saved bit holds; at 0.10 LSB every design fails, the analog unit column hardest (−61 pp against −13 to −17 pp).
- **Differential baseline.** W⁺/W⁻ arrays without a unit column need 4 bits at PR = 128 and 5 at PR = 256, but convert twice the columns; first-order estimate: at most 0.81–0.90 of the efficiency of offset encoding with exact offset terms.
- **Residual budget.** With the offset exact, tolerable data-column clipping depends on network and task (conclusively safe up to 1.9×10⁻³ for VGG8 and 5.8×10⁻⁴ for ResNet18 on CIFAR-10; ResNet18 on CIFAR-100 loses accuracy from 1.3×10⁻⁴). With the analog unit column, a 10⁻⁵ budget on all conversions classifies every conclusive point, including the held-out pair.

## Simulator version

| Item | Value |
|---|---|
| Repository | https://github.com/neurosim/NeuroSim |
| Branch | `2DInferenceV1.5-dev` |
| Commit | `9825ef40bf14d12a72c99d8e32ff8c499aeddf24` |

NeuroSim is third-party code and is **not** redistributed here. Each Colab driver clones it at the commit above and patches it at run time (converter clamp `[0, 2^b − 1]`, exact-offset hook, converter-error model, differential `simulate_array`, `levelOutput = 2^b`, VGG8 on CIFAR-100).

## Datapath conventions

- **Accuracy path:** `--hardware 0`; exact integer partition sums pass a hook in front of every converter (`_adc_quant`). The hook converts with `Q(y) = min(max(y, 0), 2^b − 1)` and can pass the offset row/unit column through unaltered (`exempt` = 1 both, 2 offset row, 3 unit column).
- **Open loop:** one unclipped pass on 512 training images (four batches of 128), histograms split into `signal` (data columns) and `offset` (offset row + unit column) rows in `S*_openloop.csv`.
- **Hardware path:** `--hardware 1 --ppa 1`, `levelOutput = 2^b`, MLSA or SAR (`SARADC`), with the offset terms converted exactly for the digital-offset rows (`S9_hw.csv`, tag `S9_digoff_*`).

## Repository layout

```
results/full_rerun_v1/   measurement CSVs of all campaigns; S*_preds/ per-image predictions for paired tests
logs/                    NeuroSIM hardware logs
patches/                 Param.cpp changes (levelOutput rule, per-corner diffs)
scripts/colab/           Colab drivers that produced every CSV (COLAB_S8 … COLAB_S13 for this version)
scripts/train/           checkpoint training scripts
analysis/unit_column/    analysis for the current manuscript (run_all.py)
analysis/popcount/       NeuroSim adder-tree cost probe for the popcount
analysis/*.py            analysis of earlier versions (Supplementary Material S1–S2)
docs/                    setup notes, upstream issue draft
models/                  checkpoints, if included in this release
```

## Where each result comes from

| Manuscript item | Files | Driver |
|---|---|---|
| Table 2, Fig. 4a (clipping of data columns vs offset terms) | `S8_openloop.csv`, `S9_openloop.csv`, `S10_openloop.csv`, `S11_openloop.csv` | `COLAB_S8_EXTENSION.py` … `COLAB_S11_FINAL.py` |
| Fig. 2, Sections 4.3 (exact offset, ablation, 256-row arrays, seeds) | `S8_runs.csv`, `S9_runs.csv`, `S8_preds/`, `S9_preds/` | `COLAB_S8_EXTENSION.py`, `COLAB_S9_DIGITAL_OFFSET.py` |
| Independent checkpoints, VGG8/CIFAR-100, denser CIFAR-100 points, code noise with exact offset | `S10_*`, `S11_*`, `S10_train_log.csv` | `COLAB_S10_STRENGTHEN.py`, `COLAB_S11_FINAL.py` |
| Fig. 3 (feedback) | `S7_layers.csv`, `S8_layers.csv` | `COLAB_S7_REVIEW2_RUNS.py`, `COLAB_S8_EXTENSION.py` |
| Fig. 4b, Table 3 (budgets, minimum resolution) | all of the above + `S6_*`, `S7_*` | — |
| Table 4, Fig. 5 (hardware) | `S8_hw.csv`, `S9_hw.csv` | `COLAB_S8_EXTENSION.py`, `COLAB_S9_DIGITAL_OFFSET.py` |
| Popcount cost (Section 4.6) | — | `analysis/popcount/popcount.cpp` |
| Table 5, Section 4.8, Tables S17–S18 (converter errors, differential mapping and its cost) | `S12_runs.csv`, `S12_openloop.csv`, `S12_hist.csv`, `S12_preds/`, `S13_hw.csv` | `COLAB_S12_ADC_DIFF.py`, `COLAB_S13_DIFF_COST.py`; analysis `analysis/unit_column/s12.py` |
| Supplementary S1–S2 (calibration with the analog unit column, closed forms) | `S6_*`, `S7_*`, `S5_partial_sum_hist.csv` | `COLAB_S6_REVIEW_RUNS.py`, `COLAB_S7_REVIEW2_RUNS.py` |
| Supplementary S4–S9 (noise, coarse LSB, cost models, earlier campaign) | `S6_runs.csv`, `S8_runs.csv` (phase F), `S10_runs.csv` (phase N), `P*`, `S1*`–`S3*` | see v1.2.0 README in the release history |

## Reproducing

1. **Accuracy and open-loop runs.** In Colab with a T4 GPU, run the drivers in `scripts/colab/` (each is one cell, resume-safe; run times are stated in their headers). They write to `MyDrive/NeuroSim15v_paper3/full_rerun_v1/`.
2. **Analysis.** From the repository root:
   `python analysis/unit_column/run_all.py` (environment variables `P3_RESULTS` and `P3_OUT` override `results/full_rerun_v1` and `figures`). It needs `numpy` and `matplotlib` and writes the figures, `tables.json` (Tables 2–4, S3–S4) and `points.json` (all paired statistics).

## Citation

See `CITATION.cff`. Please cite the article once it is published; until then, cite this repository (concept DOI above).

## License

Code: MIT (`LICENSE`). Measurement data in `results/` and `logs/`: CC BY 4.0. NeuroSim keeps its own license.
