# Minimum-Viable ADC Resolution for Subarray Parallel Read in Analog Compute-in-Memory

Code, measurement files and analysis scripts for the manuscript

> B. Gökgöz, *Minimum-Viable ADC Resolution for Subarray Parallel Read in Analog Compute-in-Memory: Accuracy Limits and Architecture-Aware Converter Costing* (under review).

Every number in the manuscript comes from the CSV files in `results/`. The scripts in `scripts/colab/` produced them, and the scripts in `analysis/` rebuild the data figures and the architecture table from them.

## What the study does

ResNet18 (CIFAR-10, CIFAR-100) and a VGG8 control are evaluated on DNN+NeuroSim V1.5 over a joint sweep of subarray parallel read (PR ∈ {16 … 128} rows) and ADC resolution (*b* ∈ {5, 6, 7, 8, 12} bits). The main findings are:

- With a 5-bit ADC, accuracy collapses by PR = 96. At PR = 128 a 6-bit ADC restores floating-point accuracy.
- The collapse follows the fraction of partial sums clipped at the ADC full scale.
- NeuroSim's converter cost must be matched to the swept resolution (`levelOutput = 2^b`) before ADC options can be compared.

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
results/full_rerun_v1/   measurement CSVs (one campaign, 25–26 Aug 2026) + later S1–S3 runs
logs/                    NeuroSIM hardware logs for each matched-levelOutput corner
patches/                 Param.cpp changes (one-line levelOutput rule, per-corner diffs)
scripts/colab/           Colab drivers that produced every CSV
scripts/train/           checkpoint training and early grid scripts
analysis/                figure and architecture-model scripts (read results/ only)
docs/                    setup notes, upstream issue draft, read-noise plumbing note
models/                  checkpoints, if included in this release
```

## Where each result comes from

| Manuscript item | File(s) | Script |
|---|---|---|
| Table I, Fig. 2b | `P3_c10_full_acc.csv` | `COLAB_FULL_RERUN.py` (P3), `COLAB_PRE_SUBMIT.py` |
| Table II, Fig. 2a, Fig. 4 | `P2_clip_grid.csv` | `COLAB_FULL_RERUN.py` (P2) |
| Fig. 3 | `diag_pred_hist_*.csv`, `diag_cliff_summary.csv` | `COLAB_DUMP_HIST.py` |
| Fig. 5 | `P2b_clip_by_layer.csv` | `COLAB_LAYER_CLIP.py`, `COLAB_PRE_SUBMIT.py` |
| Table III, Fig. 6 | `P4_c100_full_acc.csv` | `COLAB_FULL_RERUN.py` (P4) |
| Table IV, Fig. 7 | `P5_vgg8.csv`, `P5b_vgg8_full.csv` | `COLAB_FULL_RERUN.py` (P5), `COLAB_OPTIONAL_LOCKS.py` |
| Tables V, VII, Figs. 8–9 | `P6_hw_energy.csv` | `COLAB_FULL_RERUN.py` (P6) |
| Table VI | `P6b_unmatched_demo.csv` | `COLAB_P6B_UNMATCHED.py` |
| Table VIII | `P6_hw_energy.csv` | `analysis/arch_model.py` |
| Table IX, Fig. 10 | `P8b_stuck_multiseed.csv` | `COLAB_OPTIONAL_LOCKS.py`, `COLAB_PRE_SUBMIT.py` |
| Table X | `P7_multiseed.csv` | `COLAB_FULL_RERUN.py` (P7) |
| ENOB, measured SAR, mixed precision | `S1_enob_noise.csv`, `S2_sar_hw.csv`, `S3_mixed_precision.csv` | `COLAB_STRENGTHEN.py` |

The 12-bit hardware point (1.17 TOPS/W, 3.04×10⁹ µm²) comes from an earlier matched sweep with the same configuration and is not in `P6_hw_energy.csv`.

## Reproducing

1. **Set up Colab.** Use a T4 GPU and follow `docs/COLAB_SETUP.md`: clone NeuroSim at the commit above, run `pip install -e pytorch-quantization`, and put the checkpoints in `models/`.
2. **Copy the drivers.** Put `scripts/colab/*.py` in `MyDrive/NeuroSim15v_paper3/`.
3. **Run the main campaign.** `%run .../COLAB_FULL_RERUN.py` covers phases P1–P8 (~12–20 h on a T4, resume-safe), then `COLAB_PRE_SUBMIT.py`.
4. **Rebuild the outputs.**
   - `python analysis/make_figures.py results/full_rerun_v1 figures`
   - `python analysis/arch_model.py results/full_rerun_v1`

The figure script needs only `matplotlib` and `numpy`.

## Citation

See `CITATION.cff`. Please cite the article once it is published; until then, cite this repository.

## License

Code: MIT (`LICENSE`). Measurement data in `results/` and `logs/`: CC BY 4.0. NeuroSim keeps its own license.
