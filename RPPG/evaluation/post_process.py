"""
metrics_toolbox.py -- evaluation metrics that match ubicomplab/rPPG-Toolbox EXACTLY.

Faithful port of evaluation/post_process.py + evaluation/metrics.py (MIT License):
  - per-VIDEO aggregation (chunks of one recording concatenated, then HR computed)
  - detrend (cumsum if diff label) + 1st-order Butterworth bandpass
  - HR by FFT or Peak
  - SNR (harmonic power ratio, dB) and MACC (max cross-correlation over lags)
  - MAE / RMSE / MAPE / Pearson with the toolbox's standard-error formulas

Use this instead of the quick per-chunk metrics when you need numbers that line
up with published rPPG-Toolbox results.
"""
from copy import deepcopy
import numpy as np
import scipy.signal
from scipy.sparse import spdiags


def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _detrend(input_signal, lambda_value):
    n = input_signal.shape[0]
    H = np.identity(n)
    ones = np.ones(n)
    minus_twos = -2 * np.ones(n)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (n - 2), n).toarray()
    return np.dot((H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))),
                  input_signal)


def power2db(mag):
    return 10 * np.log10(mag)


# Default band = the toolbox's SHIPPED default (0.6-3.3 Hz). Set PAPER_BAND=True
# to use 0.75-2.5 Hz, the NeurIPS-2023 paper recommendation noted in their code.
PAPER_BAND = False
LOW_HZ, HIGH_HZ = (0.75, 2.5) if PAPER_BAND else (0.6, 3.3)


def _calculate_fft_hr(ppg_signal, fs=30, low_pass=LOW_HZ, high_pass=HIGH_HZ):
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask)
    mask_pxx = np.take(pxx_ppg, fmask)
    return np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60


def _calculate_peak_hr(ppg_signal, fs):
    peaks, _ = scipy.signal.find_peaks(ppg_signal)
    return 60 / (np.mean(np.diff(peaks)) / fs)


def _compute_macc(pred_signal, gt_signal):
    pred = np.squeeze(deepcopy(pred_signal))
    gt = np.squeeze(deepcopy(gt_signal))
    m = min(len(pred), len(gt))
    pred, gt = pred[:m], gt[:m]
    tlcc = []
    for lag in np.arange(0, len(pred) - 1, 1):
        tlcc.append(np.abs(np.corrcoef(pred, np.roll(gt, lag))[0][1]))
    return max(tlcc)


def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=LOW_HZ, high_pass=HIGH_HZ):
    f1 = hr_label / 60
    f2 = 2 * f1
    dev = 6 / 60
    sig = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(sig.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(sig, fs=fs, nfft=N, detrend=False)
    i1 = np.argwhere((f_ppg >= (f1 - dev)) & (f_ppg <= (f1 + dev)))
    i2 = np.argwhere((f_ppg >= (f2 - dev)) & (f_ppg <= (f2 + dev)))
    irem = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass)
                       & ~((f_ppg >= (f1 - dev)) & (f_ppg <= (f1 + dev)))
                       & ~((f_ppg >= (f2 - dev)) & (f_ppg <= (f2 + dev))))
    pxx = np.squeeze(pxx_ppg)
    p1, p2, prem = np.sum(pxx[i1]), np.sum(pxx[i2]), np.sum(pxx[irem])
    return power2db((p1 + p2) / prem) if prem != 0 else 0


def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True,
                               use_bandpass=True, hr_method="FFT"):
    """Video-level HR (pred + label), SNR, MACC. Matches the toolbox 1:1.

    diff_flag=True when the model/label are the 1st derivative of PPG
    (i.e. DiffNormalized label) -> cumsum to integrate before detrending.
    """
    if diff_flag:
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    if use_bandpass:
        b, a = scipy.signal.butter(1, [LOW_HZ / fs * 2, HIGH_HZ / fs * 2], btype="bandpass")
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
    macc = _compute_macc(predictions, labels)
    if hr_method == "FFT":
        hr_pred = _calculate_fft_hr(predictions, fs=fs)
        hr_label = _calculate_fft_hr(labels, fs=fs)
    else:
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    snr = _calculate_SNR(predictions, hr_label, fs=fs)
    return hr_label, hr_pred, snr, macc


def calculate_metrics(video_preds, video_labels, fs=30, diff_flag=True,
                      hr_method="FFT"):
    """Aggregate the toolbox metrics over many videos.

    video_preds/video_labels: dict {video_id: 1D np.array of the full-video
    predicted / ground-truth signal (chunks already concatenated in order)}.
    Returns a dict of metric -> (value, standard_error).
    """
    gt_hr, pred_hr, snr_all, macc_all = [], [], [], []
    for vid in video_preds:
        p = np.asarray(video_preds[vid], dtype=np.float64)
        l = np.asarray(video_labels[vid], dtype=np.float64)
        hl, hp, snr, macc = calculate_metric_per_video(
            p, l, fs=fs, diff_flag=diff_flag, hr_method=hr_method)
        gt_hr.append(hl); pred_hr.append(hp); snr_all.append(snr); macc_all.append(macc)

    gt_hr = np.array(gt_hr); pred_hr = np.array(pred_hr)
    snr_all = np.array(snr_all); macc_all = np.array(macc_all)
    n = len(gt_hr)

    out = {}
    err = pred_hr - gt_hr
    out["MAE"] = (np.mean(np.abs(err)), np.std(np.abs(err)) / np.sqrt(n))
    sq = err ** 2
    out["RMSE"] = (np.sqrt(np.mean(sq)), np.sqrt(np.std(sq) / np.sqrt(n)))
    out["MAPE"] = (np.mean(np.abs(err / gt_hr)) * 100,
                   np.std(np.abs(err / gt_hr)) / np.sqrt(n) * 100)
    r = np.corrcoef(pred_hr, gt_hr)[0][1] if n > 1 else float("nan")
    out["Pearson"] = (r, np.sqrt((1 - r ** 2) / (n - 2)) if n > 2 else float("nan"))
    out["SNR"] = (np.mean(snr_all), np.std(snr_all) / np.sqrt(n))
    out["MACC"] = (np.mean(macc_all), np.std(macc_all) / np.sqrt(n))
    out["_n_videos"] = n
    out["_per_video"] = {"gt_hr": gt_hr.tolist(), "pred_hr": pred_hr.tolist(),
                         "snr": snr_all.tolist(), "macc": macc_all.tolist()}
    return out


def format_toolbox(metrics, hr_method="FFT"):
    lines = [f"  ({hr_method}, n_videos={metrics['_n_videos']})"]
    for k in ["MAE", "RMSE", "MAPE", "Pearson", "SNR", "MACC"]:
        v, se = metrics[k]
        unit = " bpm" if k in ("MAE", "RMSE") else ("%" if k == "MAPE" else
               (" dB" if k == "SNR" else ""))
        lines.append(f"  {k:<8} = {v:.3f} +/- {se:.3f}{unit}")
    return "\n".join(lines)


# ---- quick per-chunk stats (lightweight, for per-epoch validation logging) ----
def compute_metrics(pred_hrs, gt_hrs, sources=None):
    pred = np.asarray(pred_hrs, float); gt = np.asarray(gt_hrs, float)
    diff = pred - gt
    def block(p, g, d):
        n = len(d)
        return {"n": n, "MAE": float(np.mean(np.abs(d))),
                "RMSE": float(np.sqrt(np.mean(d ** 2))),
                "MAPE": float(np.mean(np.abs(d / g)) * 100),
                "Pearson": float(np.corrcoef(p, g)[0, 1]) if n > 1 else float("nan"),
                "MAE_SE": float(np.std(np.abs(d)) / np.sqrt(n)) if n else float("nan")}
    out = {"overall": block(pred, gt, diff)}
    if sources is not None:
        s = np.asarray(sources)
        for src in sorted(set(s)):
            m = s == src
            if m.sum():
                out[src] = block(pred[m], gt[m], diff[m])
    return out


def format_metrics(m):
    return "\n".join(
        f"  [{k:<10}] n={v['n']:<4} MAE={v['MAE']:.2f}+/-{v['MAE_SE']:.2f}  "
        f"RMSE={v['RMSE']:.2f}  MAPE={v['MAPE']:.1f}%  Pearson={v['Pearson']:.3f}"
        for k, v in m.items())
