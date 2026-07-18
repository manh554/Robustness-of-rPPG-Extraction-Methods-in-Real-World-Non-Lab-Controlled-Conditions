"""
qc.py -- dataset Quality Control: scan every recording and reject bad ones
(noisy / corrupted ground-truth) BEFORE preprocessing.

Calibrated on real MAX30102 captures: the strong discriminators are MECHANICAL
(serial duplication, counter resets, finger-on duration, % no-finger), not the
pulse SNR. So QC checks the RAW ppg.csv when present; if only the aligned label
(ground_truth.txt / ppg.txt) exists, it falls back to a lenient pulse check
(this lets trusted datasets like UBFC pass).

Verdict per recording: KEEP / MARGINAL / REJECT (+ reason). REJECT recordings are
skipped by preprocessing; MARGINAL are kept but flagged (likely "worst subjects").
"""
import os
import glob
import numpy as np
from scipy import signal as sp


class QCThresholds:
    max_dup = 5.0          # raw rows / unique samples; >5 = serial read problem
    max_resets = 0         # counter resets; >0 = timing broken
    min_finger_s = 20.0    # finger-on seconds; below = too short for rPPG
    max_nofinger = 0.40    # fraction of samples with no finger
    ir_finger = 1000       # IR above this = finger present
    min_prominence = 2.0   # FFT peak / median in HR band; below = no clear pulse
    marginal_snr = -3.0    # harmonic SNR (dB) below this = flag MARGINAL (kept)


def _pulse_quality(ppg, fs):
    ppg = np.asarray(ppg, float)
    if len(ppg) < fs * 3 or np.std(ppg) < 1e-9:
        return dict(hr=0.0, prominence=0.0, snr_db=-99.0)
    d = np.diff(ppg); d = d / (np.std(d) + 1e-7)
    b, a = sp.butter(1, [0.6 / (fs / 2), 3.3 / (fs / 2)], btype="bandpass")
    df = sp.filtfilt(b, a, d)
    n = 1
    while n < len(df):
        n *= 2
    f, p = sp.periodogram(df, fs=fs, nfft=n)
    m = (f >= 0.6) & (f <= 3.3)
    fb, pb = f[m], p[m]
    peak = pb.max(); hr = fb[np.argmax(pb)] * 60
    prom = float(peak / (np.median(pb) + 1e-12))
    f1 = hr / 60; f2 = 2 * f1
    bw = lambda c, w=6 / 60: (fb >= c - w) & (fb <= c + w)
    sig = pb[bw(f1)].sum() + pb[bw(f2)].sum(); rest = pb[~bw(f1) & ~bw(f2)].sum()
    snr = 10 * np.log10(sig / rest) if rest > 0 else 0.0
    return dict(hr=float(hr), prominence=prom, snr_db=float(snr))


def _read_label(path):
    if os.path.basename(path) == "ground_truth.txt":
        with open(path) as f:
            rows = [r.split() for r in f.read().strip().split("\n")]
        return np.asarray(rows[0], dtype=np.float64)
    return np.loadtxt(path).astype(np.float64)


def assess_recording(folder, fps_label=30, raw_fs=100, th=QCThresholds):
    metrics = {}
    raw = [c for c in sorted(glob.glob(os.path.join(folder, "*.csv")))
           if "ppg" in os.path.basename(c).lower()
           and os.path.basename(c).lower() not in ("frames.csv", "qc_report.csv")]
    if raw:
        try:
            import csv
            si, ir = [], []
            with open(raw[0]) as f:
                for r in csv.DictReader(f):
                    si.append(int(float(r["sample_idx"]))); ir.append(float(r["ir"]))
            si = np.asarray(si); ir = np.asarray(ir)
            dup = len(si) / max(1, len(np.unique(si)))
            resets = int((np.diff(si) < 0).sum())
            u, idx = np.unique(si, return_index=True); ir_u = ir[idx]
            on = ir_u > th.ir_finger
            pct_nf = float(1 - on.mean()); dur = float(on.sum() / raw_fs)
            pq = _pulse_quality(ir_u[on] if on.sum() > raw_fs * 5 else ir_u, raw_fs)
            metrics = dict(source="raw_ppg", dup=round(dup, 1), resets=resets,
                           pct_nofinger=round(pct_nf, 2), finger_s=round(dur, 1), **pq)
            if dup > th.max_dup:
                return "REJECT", f"serial duplication {dup:.0f}x", metrics
            if resets > th.max_resets:
                return "REJECT", f"counter reset x{resets}", metrics
            if dur < th.min_finger_s:
                return "REJECT", f"finger only {dur:.0f}s", metrics
            if pct_nf > th.max_nofinger:
                return "REJECT", f"no-finger {pct_nf*100:.0f}%", metrics
            if pq["prominence"] < th.min_prominence:
                return "REJECT", f"no clear pulse (prom {pq['prominence']:.1f})", metrics
            if pq["snr_db"] < th.marginal_snr:
                return "MARGINAL", f"low SNR {pq['snr_db']:.1f}dB", metrics
            return "KEEP", "ok", metrics
        except Exception as e:
            metrics["raw_error"] = str(e)
    label = (glob.glob(os.path.join(folder, "ground_truth.txt"))
             or glob.glob(os.path.join(folder, "ppg.txt")))
    if not label:
        return "REJECT", "no label file", metrics
    pq = _pulse_quality(_read_label(label[0]), fps_label)
    metrics = dict(metrics, source="label", **pq)
    if pq["prominence"] < th.min_prominence or pq["snr_db"] <= -90:
        return "REJECT", f"flat / no pulse (prom {pq['prominence']:.1f})", metrics
    if pq["snr_db"] < th.marginal_snr:
        return "MARGINAL", f"low SNR {pq['snr_db']:.1f}dB", metrics
    return "KEEP", "ok", metrics


def qc_scan(clips, out_csv=None, fps_label=30, th=QCThresholds, verbose=True):
    import csv as _csv
    kept, rows = set(), []
    counts = {"KEEP": 0, "MARGINAL": 0, "REJECT": 0}
    for rec, vid, label_path in clips:
        folder = os.path.dirname(vid)
        verdict, reason, m = assess_recording(folder, fps_label, th=th)
        counts[verdict] += 1
        if verdict != "REJECT":
            kept.add(rec)
        rows.append(dict(recording=rec, verdict=verdict, reason=reason, **m))
        if verbose:
            tag = {"KEEP": "  ok ", "MARGINAL": " ~mid", "REJECT": "REJECT"}[verdict]
            print(f"  [QC {tag}] {rec:<22} {reason}")
    if out_csv:
        keys = sorted({k for r in rows for k in r})
        head = ["recording", "verdict", "reason"] + [k for k in keys if k not in ("recording", "verdict", "reason")]
        with open(out_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=head)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    if verbose:
        print(f"  [QC summary] KEEP={counts['KEEP']}  MARGINAL={counts['MARGINAL']}  "
              f"REJECT={counts['REJECT']}  (kept {len(kept)}/{len(clips)})")
    return kept
