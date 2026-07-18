"""Plots for unsupervised results: waveform grid + FFT spectra (best vs worst)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal


def _bandpass(sig, fs, lo=0.6, hi=3.3):
    b, a = scipy.signal.butter(1, [lo / fs * 2, hi / fs * 2], btype="bandpass")
    return scipy.signal.filtfilt(b, a, np.double(sig))


def waveform_grid(records, fs, path, n_best=3, n_worst=3):
    if not records:
        return
    ranked = sorted(records, key=lambda r: r["macc"])
    picks = ranked if len(ranked) <= n_best + n_worst else ranked[:n_worst] + ranked[-n_best:]
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 1.7 * len(picks)), squeeze=False)
    for ax, r in zip(axes[:, 0], picks):
        pred = _bandpass(r["pred"], fs); gt = _bandpass(r["gt"], fs)
        t = np.arange(len(pred)) / fs
        ax.plot(t, (gt - gt.mean()) / (gt.std() + 1e-8), lw=1.3, label="GT")
        ax.plot(t, (pred - pred.mean()) / (pred.std() + 1e-8), lw=1.0, alpha=0.85, label="pred")
        good = "GOOD" if (r["macc"] >= 0.7 and r["mae"] <= 5) else \
               ("POOR" if (r["macc"] < 0.5 or r["mae"] > 10) else "mid")
        ax.set_title(f"{r['subject']} ({r['env']}) [{r['source']}]  MAE={r['mae']:.1f}  "
                     f"MACC={r['macc']:.2f}  SNR={r['snr']:.1f}dB -> {good}", fontsize=9, loc="left")
        ax.set_yticks([])
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Per-recording waveforms (worst -> best by MACC)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); plt.savefig(path, dpi=120); plt.close()


def spectra(records, fs, path, n_show=3, lo=0.6, hi=3.3):
    if not records:
        return
    ranked = sorted(records, key=lambda r: r["macc"])
    picks = ([("WORST", r) for r in ranked[:n_show]] +
             [("BEST", r) for r in ranked[-n_show:]]) if len(ranked) > 2 * n_show else \
            [("WORST" if i < len(ranked) / 2 else "BEST", r) for i, r in enumerate(ranked)]
    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 1.6 * len(picks)), squeeze=False)
    for ax, (tag, r) in zip(axes[:, 0], picks):
        s = _bandpass(r["pred"], fs)
        n = 1
        while n < len(s):
            n *= 2
        f, p = scipy.signal.periodogram(s, fs=fs, nfft=n, detrend=False)
        m = (f >= lo) & (f <= hi)
        ax.plot(f[m] * 60, p[m], lw=1.1)
        ax.axvline(r["hr_gt"], color="g", ls="--", lw=1, label=f"GT {r['hr_gt']:.0f}")
        ax.axvline(r["hr_pred"], color="r", ls=":", lw=1, label=f"pred {r['hr_pred']:.0f}")
        ax.set_title(f"[{tag}] {r['subject']} ({r['env']})  MACC={r['macc']:.2f}",
                     fontsize=9, loc="left")
        ax.set_yticks([]); ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("frequency (bpm)")
    fig.suptitle("Predicted-signal spectra: best vs worst (harmonic-error check)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); plt.savefig(path, dpi=120); plt.close()
