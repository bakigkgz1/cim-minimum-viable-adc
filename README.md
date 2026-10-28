# Calibrating a Clipping-Tail Criterion for ADC Sizing in Parallel-Read Analog Compute-in-Memory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23573121.svg)](https://doi.org/10.5281/zenodo.23573121)

Code, measurement files and analysis scripts for the manuscript

> B. Gökgöz, *Calibrating a Clipping-Tail Criterion for ADC Sizing in Parallel-Read Analog Compute-in-Memory Accelerators* (under review).

Version 1.2.0 adds the reviewer-driven runs S6 and S7: a textbook 2^b-code converter, 41 operating points (three network–dataset pairs, three quantization seeds) with per-image predictions for paired (McNemar) tests, open-loop histograms, layer-resolved feedback and single-layer clipping, conductance read noise, coarse-LSB runs with three rounding schemes and symmetric code noise. Earlier versions accompanied previous drafts of the manuscript.

Every number in the manuscript comes from the CSV files in `results/`. The scripts in `scripts/colab/` produced them, and the scripts in `analysis/` rebuild the data figures and the architecture table from them.

## What the study does

ResNet18 (CIFAR-10, CIFAR-100) and a VGG8 control are evaluated on DNN+NeuroSim V1.5 over a joint sweep of subarray parallel read (PR ∈ {16 … 128} rows) and ADC resolution (*b* ∈ {5, 6, 7, 8, 12} bits). The main findings are:

- **Measured budget.** Every point whose loss is conclusively within 0.5 pp (95% paired CI) has an open-loop clipped fraction ≤ 7.4×10⁻⁶, every conclusively lossy point ≥ 1.4×10⁻⁵; τ = 10⁻⁵, set on ResNet18/CIFAR-10, classifies all 36 conclusive points of 41 and follows seed-to-seed changes.
- **Feedback.** Clipping raises the next layer's mean partial sum by 8–11% and compounds with depth; the measured altered fraction reaches up to 1600× the open-loop value.
- **Closed forms.** A per-layer beta-binomial (mean, variance) never under-sizes; the independent-bit binomial under-sizes all 23 configurations. Full-range sizing ⌈log₂(PR+1)⌉ asks for 8 bits at PR = 128 where 6 suffice.
- **One alteration budget.** Code noise is tolerated at 0.11 LSB rms (code-change probability 5.5×10⁻⁶) but not 0.14 (3.6×10⁻⁴); a 2- or 4-cell LSB collapses accuracy with any rounding; read noise up to 3% does not move the boundary.
- **Cost.** With `levelOutput = 2^b`, six bits at PR = 128 raises efficiency 1.35× (flash-type MLSA) and 1.10× (SAR) over practical full-range sizing (7 bits).

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
results/full_rerun_v1/   measurement CSVs: main campaign (Aug 2026), S1–S5 and S6 (Oct 2026); S6_preds/ per-image predictions
logs/                    NeuroSIM hardware logs for each matched-levelOutput corner
patches/                 Param.cpp changes (one-line levelOutput rule, per-corner diffs)
scripts/colab/           Colab drivers that produced every CSV
scripts/train/           checkpoint training and early grid scripts
analysis/                figure and architecture-model scripts (read results/ only)
docs/                    setup notes, upstream issue draft, read-noise plumbing note
models/                  checkpoints, if included in this release
```

## Where each result comes from

Numbering follows the current manuscript (v1.2.0). Supplementary items are prefixed S.

| Manuscript item | File(s) | Script |
|---|---|---|
| Fig. 2 (accuracy vs PR) | `S6_runs.csv`, `S6_preds/*.npy` | `COLAB_S6_REVIEW_RUNS.py`, `analysis/s6_review_analysis.py` |
| Table II, Figs. 3–4 (41-point validation, seeds, feedback) | `S6_*`, `S7_runs.csv`, `S7_openloop.csv`, `S7_layers.csv`, `S7_preds/*.npy` | `COLAB_S7_REVIEW2_RUNS.py`, `analysis/s7_review2_analysis.py` |
| Table III, Fig. 5 (sizing chart, closed forms) | `S6_openloop.csv`, `S6_hist.csv`, `S5_partial_sum_hist.csv` | `COLAB_S5_HISTOGRAMS.py`, `COLAB_S6_REVIEW_RUNS.py`, `analysis/s6_review_analysis.py` |
| Table IV, Fig. 6, Fig. S2 (hardware) | `P6_hw_energy.csv`, `S2_sar_hw.csv` | `COLAB_FULL_RERUN.py` (P6), `COLAB_STRENGTHEN.py` (S2) |
| Table V (analytic cost models) | `P6_hw_energy.csv` | `analysis/arch_model.py` |
| Table VI (measured SAR) | `S2_sar_hw.csv` | `COLAB_STRENGTHEN.py` (S2) |
| Table VII (code noise) | `S6_runs.csv` (phase D) | `COLAB_S6_REVIEW_RUNS.py` |
| Table VIII (range scaling) | `S6_runs.csv` (phase C) | `COLAB_S6_REVIEW_RUNS.py` |
| Section 4.7 (read noise) | `S7_runs.csv` (phase N) | `COLAB_S7_REVIEW2_RUNS.py` |
| Tables S1–S3, Fig. S1 (earlier campaign, simulator clamp) | `P2_clip_grid.csv`, `P3_c10_full_acc.csv`, `P4_c100_full_acc.csv`, `P5*_vgg8*.csv`, `diag_pred_hist_*.csv` | `COLAB_FULL_RERUN.py`, `COLAB_DUMP_HIST.py` |
| Table S4 (unmatched levelOutput) | `P6b_unmatched_demo.csv` | `COLAB_P6B_UNMATCHED.py` |
| Figs. S3–S4, Table S5 (per-layer clipping, allocation) | `P2b_clip_by_layer.csv`, `S3_mixed_precision.csv` | `COLAB_LAYER_CLIP.py`, `COLAB_STRENGTHEN.py` (S3) |
| Tables S6–S7, Fig. S5 (stuck-at, seeds) | `P8b_stuck_multiseed.csv`, `P7_multiseed.csv` | `COLAB_OPTIONAL_LOCKS.py`, `COLAB_FULL_RERUN.py` (P7) |
| Table S8 (fine noise sweep) | `S1_enob_noise.csv`, `S1b_noise_threshold.csv` | `COLAB_STRENGTHEN.py` (S1), `COLAB_S1B_NOISE.py` |

`analysis/s6_review_analysis.py` writes Figs. 2, 5 and 6 and prints Table III; `analysis/s7_review2_analysis.py` prints Table II and writes Figs. 3–4; `analysis/sizing_rule.py` reproduces the earlier S5-only analysis.

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
   - `python analysis/s6_review_analysis.py results/full_rerun_v1 figures`
   - `python analysis/s7_review2_analysis.py results/full_rerun_v1 figures`

The figure script needs only `matplotlib` and `numpy`.

## Citation

See `CITATION.cff`. Please cite the article once it is published; until then, cite this repository.

## License

Code: MIT (`LICENSE`). Measurement data in `results/` and `logs/`: CC BY 4.0. NeuroSim keeps its own license.
