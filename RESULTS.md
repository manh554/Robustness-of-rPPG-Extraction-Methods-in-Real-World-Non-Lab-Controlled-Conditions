# Results and Analysis

This document summarizes what distinguishes this project from the base toolbox: a **cross-dataset
robustness study** comparing training-free and supervised rPPG methods on data recorded outside
the lab. The full write-up is in the final report; this is the short version.

> **Plots live on a separate branch.** To keep the main branch clean, all figures (summary chart,
> recovered waveforms, power spectra, Bland-Altman plots) are on the `plots` branch. If you want to
> see them: `git checkout plots` (or browse that branch on the repo).

## Key finding

On a self-collected test set recorded with a consumer webcam and a MAX30102 contact sensor,
**training-free CHROM is the most robust method (MAE 10.83 bpm, Pearson 0.64)**, while the
supervised models trained on UBFC-rPPG **degrade sharply under domain shift** (MAE around
30 bpm, with negative HR correlation). The supervised failure is a *frequency-calibration /
peak-selection* problem, not a failure to detect the pulse.

## Data

- **UBFC-rPPG** (public, lab): 38 usable subjects, 640x480 @ 30 fps, CMS50E finger-oximeter
  reference. Used for supervised training and an in-domain test.
- **Self-collected** (ours): laptop webcam @ 30 fps + MAX30102 fingertip PPG via Arduino.
  18 recorded, **11 retained** after automatic quality control (stuck samples, counter resets,
  valid finger-contact duration). Held out as an independent cross-dataset test.

All splits are subject-level (person-disjoint) to prevent identity leakage.

## Quantitative comparison

Self-collected test set (11 recordings). Lower is better for MAE / RMSE / MAPE; higher is better
for Pearson / SNR / MACC.

| Method   | Type          | MAE   | RMSE  | MAPE | Pearson | SNR    | MACC |
|----------|---------------|-------|-------|------|---------|--------|------|
| DeepPhys | supervised    | 31.30 | 37.94 | 33.2 | -0.83   | -10.04 | 0.21 |
| TS-CAN   | supervised    | 29.73 | 37.17 | 31.4 | -0.31   | -8.74  | 0.23 |
| LGI      | training-free | 27.81 | 33.54 | 30.1 | -0.21   | -8.42  | 0.25 |
| POS      | training-free | 12.82 | 19.23 | 16.0 | 0.38    | -6.03  | 0.34 |
| **CHROM**| training-free | **10.83** | **16.57** | **12.3** | **0.64** | -6.05 | 0.33 |

Among training-free methods, CHROM matches its reference behavior best; POS is slightly higher
(more sensitive to the region of interest); LGI is the weakest.

## Failure-mode analysis

The supervised errors do not come from a missing pulse. The recovered waveforms are quasi-periodic,
but their **dominant frequency is biased low**, so heart rate is systematically underestimated
(e.g. 81 -> 39 bpm, 116 -> 43 bpm). The power spectra confirm this is a **peak-selection** issue:
on favorable recordings the predicted spectrum has a single peak at the true rate, while on failing
recordings it becomes broadband and the peak-picker locks onto a spurious low-frequency component.

Bland-Altman analysis shows the same pattern: training-free CHROM (in-domain, UBFC-rPPG) has a
small bias (~4 bpm) with points near zero, while supervised DeepPhys on the self-collected data
shows a large bias (~24 bpm), much wider limits of agreement, and predictions collapsing toward a
narrow low range. (See the `plots` branch for these figures.)

## Takeaways

- On consumer hardware, **training-free methods generalize more reliably** without any adaptation;
  CHROM is the strongest baseline here.
- The supervised models still capture pulse information but need **domain adaptation** (e.g.
  fine-tuning on a few self-collected subjects) to fix the low-frequency bias.

## Limitations

- The self-collected set is **small (n = 11) and noisy**; absolute errors depend on the frequency
  band and the region of interest.
- The self-collected HR reference is derived by **FFT over the contact-PPG signal**, not an ECG
  gold standard.
- The supervised results are unusually poor (MAE ~30 bpm, negative correlation); part of this may
  reflect the peak-selection / post-processing stage rather than domain shift alone, and warrants
  a check before drawing strong conclusions.

