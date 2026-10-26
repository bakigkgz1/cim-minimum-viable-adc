# A Clipping-Tail Rule for Sizing the ADC in Parallel-Read Analog Compute-in-Memory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23573122.svg)](https://doi.org/10.5281/zenodo.23573122)

Code, measurement files and analysis scripts for the manuscript

> B. Gökgöz, *A Clipping-Tail Rule for Sizing the ADC in Parallel-Read Analog Compute-in-Memory Accelerators* (under review).

Version 1.1.0 adds the open-loop histogram (S5) and full-scale calibration (S4) runs and the sizing-rule analysis. Version 1.0.0 accompanied an earlier version of the manuscript with the title *Minimum-Viable ADC Resolution for Subarray Parallel Read in Analog Compute-in-Memory*.

Every number in the manuscript comes from the CSV files in `results/`. The scripts in `scripts/colab/` produced them, and the scripts in `analysis/` rebuild the data figures and the architecture table from them.

## What the study does

ResNet18 (CIFAR-10, CIFAR-100) and a VGG8 control are evaluated on DNN+NeuroSim V1.5 over a joint sweep of subarray parallel read (PR ∈ {16 … 128} rows) and ADC resolution (*b* ∈ {5, 6, 7, 8, 12} bits). The main findings are:

- **Sizing rule.** The minimum ADC resolution for a read width PR is the smallest *b* whose open-loop clipped fraction, the share of partial sums at or above full scale 2^*b* in one unclipped pass, stays below τ = 2×10⁻⁵. The rule classifies all 33 measured operating points correctly, including nine VGG8 and CIFAR-100 points that played no part in choosing τ.
- **Cascade.** Above the budget, clipped outputs raise the inputs of later layers; the measured clipped fraction exceeds the open-loop value 12× at PR = 80 and about 1000× at PR = 96, and accuracy collapses.
- **Distribution models.** A per-layer two-moment beta-binomial form never under-sizes the ADC; the independent-bit binomial model under-sizes it in all 16 configurations. The full-range rule ⌈log₂(PR+1)⌉ over-provisions by 1–2 bits (8 instead of 6 at PR = 128).
- **Exact LSB.** Scaling the full scale so that one LSB spans 2 or 4 cells removes clipping but collapses accuracy (S4); code noise above about 0.11 LSB rms does the same (S1b).
- **Costing.** NeuroSim's converter cost must be matched to the swept resolution (`levelOutput = 2^b`); with matched flash-type, measured SAR and analytic costs, 8-bit converters are dominated.

## Simulator version

| Item | Value |
|---|---|
| Repository | https://github.com/neurosim/NeuroSim |
| Branch | `2DInferenceV1.5-dev` |
| Commit | `9825ef40bf14d12a72c99d8e32ff8c499aeddf24` |
| Converter model | multilevel sense amplifier (`SARADC = false`), 8 columns per converter |

NeuroSim is third-party code and is **not** redistributed here. Clone it at the commit above and apply the patches in `patches/`.

## The `levelOutput` correction

`NeuroSIM/Param.cpp` (line 260 at the commit above) sets `levelOutput = 128` at build time. The Python flag `--adc_precision` never reaches the C++ estimator; only PR is passed on its command line. Every resolution is therefore costed as a 7-bit converter unless the line is rewritten and NeuroSIM is rebuilt.

All hardware numbers in the paper use `levelOutput = 2^b`, set by `set_lo()` in `scripts/colab/COLAB_FULL_RERUN.py`. `patches/` holds the one-line change and, where available, the exact per-resolution diffs that were built. `docs/NEUROSIM_LEVELOUTPUT_ISSUE.md` is the upstream report.

## Accuracy path vs hardware path

Accuracy runs use NeuroSim's digital mode (`--hardware 0`). Each layer is still split into 1-bit input cycles, 1-bit weight cells and PR-row partitions, and every partition sum passes the ADC clamp `[0, 2^b]`; the sum itself is an exact integer. Hardware runs use `--hardware 1 --ppa 1`.

Conductance read noise lives only in the `--hardware 1` conductance model. The read-noise rows of `P8_variation.csv` are therefore no-ops and are not reported as a robustness result (see `docs/PLUMBING_READ_NOISE.md`). Stuck-at faults act on both paths.

## Repository layout

```
results/full_rerun_v1/   measurement CSVs: main campaign (25–26 Aug 2026), S1–S3 (22 Oct 2026), S1b, S4, S5 (Oct 2026)
logs/                    NeuroSIM hardware logs for each matched-levelOutput corner
patches/                 Param.cpp changes (one-line levelOutput rule, per-corner diffs)
scripts/colab/           Colab drivers that produced every CSV
scripts/train/           checkpoint training and early grid scripts
analysis/                figure and architecture-model scripts (read results/ only)
docs/                    setup notes, upstream issue draft, read-noise plumbing note
models/                  checkpoints, if included in this release
```

## Where each result comes from

Numbering follows the current manuscript (v1.1.0). Supplementary items are prefixed S.

| Manuscript item | File(s) | Script |
|---|---|---|
| Table II, Fig. 2b | `P3_c10_full_acc.csv` | `COLAB_FULL_RERUN.py` (P3), `COLAB_PRE_SUBMIT.py` |
| Table III, Fig. 2a, Fig. 4 | `P2_clip_grid.csv` | `COLAB_FULL_RERUN.py` (P2) |
| Fig. 3 | `diag_pred_hist_*.csv`, `diag_cliff_summary.csv` | `COLAB_DUMP_HIST.py` |
| Table IV, Fig. S2 | `P4_c100_full_acc.csv` | `COLAB_FULL_RERUN.py` (P4) |
| Table V, Fig. S3 | `P5_vgg8.csv`, `P5b_vgg8_full.csv` | `COLAB_FULL_RERUN.py` (P5), `COLAB_OPTIONAL_LOCKS.py` |
| Table VI, Fig. 5 (sizing rule) | `S5_partial_sum_hist.csv`, `P2_clip_grid.csv` | `COLAB_S5_HISTOGRAMS.py`, `analysis/sizing_rule.py` |
| Table VII, Fig. 6, Fig. S6 | `P6_hw_energy.csv` | `COLAB_FULL_RERUN.py` (P6) |
| Table VIII | `P6b_unmatched_demo.csv` | `COLAB_P6B_UNMATCHED.py` |
| Table IX | `P6_hw_energy.csv` | `analysis/arch_model.py` |
| Table X (measured SAR) | `S2_sar_hw.csv`, `P6_hw_energy.csv` | `COLAB_STRENGTHEN.py` (S2) |
| Table XI (conversion noise) | `S1_enob_noise.csv`, `S1b_noise_threshold.csv` | `COLAB_STRENGTHEN.py` (S1), `COLAB_S1B_NOISE.py` |
| Table XII (full-scale scaling) | `S4_calibration.csv`, `S4_layer_scale.csv` | `COLAB_S4_CALIBRATION.py` |
| Table S1, Fig. S1 (stuck-at) | `P8b_stuck_multiseed.csv` | `COLAB_OPTIONAL_LOCKS.py`, `COLAB_PRE_SUBMIT.py` |
| Table S2 (seeds) | `P7_multiseed.csv` | `COLAB_FULL_RERUN.py` (P7) |
| Table S3, Fig. S4 (per-layer allocation) | `S3_mixed_precision.csv`, `P2b_clip_by_layer.csv` | `COLAB_STRENGTHEN.py` (S3) |
| Fig. S5 (per-layer clipping) | `P2b_clip_by_layer.csv` | `COLAB_LAYER_CLIP.py`, `COLAB_PRE_SUBMIT.py` |

`analysis/make_figures.py` writes the data figures under their working file names (for example `fig2.png` = Fig. 2, `fig3.png` = Fig. 6); `analysis/sizing_rule.py` prints Table VI and the 33-point validation and writes Fig. 5.

The S1–S3 runs used a newer Colab runtime (Python 3.13, CUDA 13.0) with the same NeuroSim commit and checkpoints. On that runtime NeuroSim's `pytorch-quantization/setup.py` no longer installs, so `COLAB_STRENGTHEN.py` compiles its CUDA extension with `torch.utils.cpp_extension.load` instead.

The 12-bit hardware point (1.17 TOPS/W, 3.04×10⁹ µm²) comes from an earlier matched sweep with the same configuration and is not in `P6_hw_energy.csv`.

## Reproducing

1. **Set up Colab.** Use a T4 GPU and follow `docs/COLAB_SETUP.md`: clone NeuroSim at the commit above, run `pip install -e pytorch-quantization`, and put the checkpoints in `models/`.
2. **Copy the drivers.** Put `scripts/colab/*.py` in `MyDrive/NeuroSim15v_paper3/`.
3. **Run the main campaign.** `%run .../COLAB_FULL_RERUN.py` covers phases P1–P8 (~12–20 h on a T4, resume-safe), then `COLAB_PRE_SUBMIT.py`.
4. **Rebuild the outputs.**
   - `python analysis/make_figures.py results/full_rerun_v1 figures`
   - `python analysis/arch_model.py results/full_rerun_v1`
   - `python analysis/sizing_rule.py results/full_rerun_v1 figures`

The figure script needs only `matplotlib` and `numpy`.

## Citation

See `CITATION.cff`. Please cite the article once it is published; until then, cite this repository.

## License

Code: MIT (`LICENSE`). Measurement data in `results/` and `logs/`: CC BY 4.0. NeuroSim keeps its own license.
