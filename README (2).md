# Robustness of rPPG Extraction Methods in Real-World (Non-Lab-Controlled) Conditions

Evaluating how well unsupervised rPPG extraction methods (CHROM, POS, LGI, ...) hold up outside
controlled lab settings. The pipeline estimates heart rate from face video and compares methods
across data sources (public lab data vs. self-collected) and across environments, to measure their
robustness in realistic conditions.

Adapted from the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox): same evaluation
metrics, but restructured to use a Python config instead of YAML, with a quality-control (QC)
step for self-collected data.

## Disclaimer and Scope

- Covers only the unsupervised (non-deep-learning) part of rPPG.
- For supervised / neural methods (DeepPhys, TS-CAN, PhysNet, EfficientPhys, PhysFormer, ...),
  use the original repo: https://github.com/ubicomplab/rPPG-Toolbox.
- No YAML. The workflow is set in the `CONFIG` dict at the top of `main.py`.
- Only the UBFC-rPPG data format is supported. Self-collected data must be arranged the same way
  (one folder per subject, with a video and a UBFC-style `ground_truth.txt`).
- Research use only. Not a medical device; do not use for diagnosis.

## Repository Structure

```
.
├── collect/            Data collection (Arduino + MAX3010x sensor), see below
├── dataset/            rgb_preprocess.py: face-crop, QC, manifest generation
├── evaluation/         post_process.py: toolbox metrics
├── methods/            Unsupervised algorithms (CHROM, POS, LGI, ...)
├── tools/              plots.py: waveforms and spectra
├── main.py             Entry point: preprocess | run
└── predictor.py        Runs a method and evaluates
```

## Collecting Your Own Data

The `collect/` folder records synchronized face video + contact PPG using an Arduino board and a
MAX3010x (MAX30102) pulse-oximeter sensor:

- `max30102_stream.ino` — flashed to the Arduino; streams the sensor's IR values over serial
  (115200 baud) as CSV.
- `collecting.py` — runs on the PC; opens the webcam and serial port, waits for a finger on the
  sensor, records for a set duration, then aligns the PPG to the video frames.

Output is written to one subject folder (default `raw_custom/subject01`): `video.avi`, `ppg.csv`,
`frames.csv`, `meta.txt`, and a UBFC-style `ground_truth.txt` (waveform, HR, timestamps). This is
the UBFC-format folder the pipeline expects.

```bash
python collect/collecting.py --out raw_custom/subject01 --duration 30 --fps 30
```

Note: the HR in `ground_truth.txt` is estimated by a rolling FFT over the contact-PPG signal, not
measured by a clinical device. Treat it as a reference, not a gold standard.

## Configuring the Workflow

Edit the `CONFIG` dict at the top of `main.py` to set mode, paths, face backend, QC, method, and
sampling rate. Set it once, then run `python main.py`. Flags can override it per run.

## Usage

Two stages, dispatched by `--mode`.

### 1. Preprocess (face crop, QC, manifest)

```bash
python main.py --mode preprocess \
    --root ../ubfc \            # raw data folder (UBFC format)
    --format ubfc \
    --out ../cache_rgb \        # preprocessed .npz + manifest go here
    --size 72 \                 # face-crop size (px)
    --face_backend auto \       # auto | mediapipe | haar | center
    --qc on \                   # input quality control
    --source_tag ubfc           # "source" label in the manifest
```

### 2. Run and evaluate

```bash
python main.py --mode run \
    --method all \              # one method name, or "all"
    --manifest ../cache_rgb/manifest_rgb.csv \
    --fs 30 \                   # video fps
    --outdir ../runs/unsup
```

Default paths use `../`, so `ubfc/`, `cache_rgb/`, and `runs/` are expected next to the project
folder. Adjust to your layout.

## Self-Collected Data Workflow

1. Record with `collect/collecting.py` (produces a UBFC-format subject folder).
2. Point `root` in `main.py`'s `CONFIG` at that folder, and give it a distinct `source_tag`
   (e.g. `own`) so it appears separately in the report. The folder name must match.
3. Run preprocess, then run.

Manifest columns `path`, `source`, `subject`, `env` drive the report breakdowns. If `source` /
`subject` / `env` are blank they default to `na` / `all`, and per-source / per-environment splits
become meaningless — confirm `rgb_preprocess.py` fills them.

Ethics: self-collected physiological data involves human subjects. Get informed consent, respect
privacy, and do not commit raw recordings to a public repo.

## Evaluation and Outputs

Metrics (from `evaluation/post_process.py`): MAE, RMSE, MAPE, Pearson, SNR, MACC. HR is estimated
by FFT; the BVP is treated as a raw waveform (`diff_flag=False`).

Each report gives: overall pooled (micro-average), overall 1:1 balanced (each source weighted
equally), per-source, and per-environment.

Files in `--outdir`: `per_recording_<method>.csv`, `report_<method>.json`,
`waveforms_<method>.png`, `spectra_<method>.png`.

## Acknowledgement

Built on and modified from the open-source rPPG-Toolbox by the UbiComp Lab. The unsupervised
methods and the evaluation code come from their work; thanks to the original authors.

- rPPG-Toolbox (NeurIPS 2023): https://github.com/ubicomplab/rPPG-Toolbox
- Paper: https://arxiv.org/abs/2210.00716

## Citation

```bibtex
@article{liu2022rppg,
  title={rPPG-Toolbox: Deep Remote PPG Toolbox},
  author={Liu, Xin and Narayanswamy, Girish and Paruchuri, Akshay and Zhang, Xiaoyu and Tang, Jiankai and Zhang, Yuzhe and Wang, Yuntao and Sengupta, Soumyadip and Patel, Shwetak and McDuff, Daniel},
  journal={arXiv preprint arXiv:2210.00716},
  year={2022}
}
```

Also cite each unsupervised method you use (e.g. POS: Wang et al., 2016; CHROM: de Haan et al.,
2013; LGI: Pilz et al., 2018).

## License

rPPG-Toolbox is released under the Responsible AI License (RAIL):
https://www.licenses.ai/source-code-license. As a derivative work, this repo is likely bound by
those terms — review them and state your license below. (Not legal advice.)

License of this repository: TODO
