"""
predictor.py -- run an unsupervised method (CHROM/POS/LGI) and evaluate with the
SAME toolbox metrics as the supervised repo.

Reports, for each method:
  - OVERALL (pooled / micro-average over all recordings)
  - OVERALL (1:1 BALANCED / macro-average: each data SOURCE weighted equally)
  - per-source breakdown   (e.g. ubfc vs own)
  - per-environment breakdown (rest / lowlight / exercise ...)
Plus per-recording CSV, waveform grid, and best/worst spectra.

The 1:1 balanced overall answers "tỉ lệ đo từng phần dữ liệu là 1:1": each part
contributes equally regardless of how many recordings it has.
"""
import os
import csv
import json
import argparse
import numpy as np

from methods import METHODS
import evaluation.post_process as mtb
import tools.plots as plots


def load_manifest(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"path": r["path"], "source": r.get("source", "na"),
                         "subject": r.get("subject", "na"), "env": r.get("env", "all")})
    return rows


def aggregate(records):
    if not records:
        return None
    hp = np.array([r["hr_pred"] for r in records]); hg = np.array([r["hr_gt"] for r in records])
    snr = np.array([r["snr"] for r in records]); macc = np.array([r["macc"] for r in records])
    n = len(records); err = hp - hg
    r = np.corrcoef(hp, hg)[0, 1] if n > 1 else float("nan")
    return {"n": n,
            "MAE": [float(np.mean(np.abs(err))), float(np.std(np.abs(err)) / np.sqrt(n))],
            "RMSE": [float(np.sqrt(np.mean(err ** 2))), float(np.sqrt(np.std(err ** 2) / np.sqrt(n)))],
            "MAPE": [float(np.mean(np.abs(err / hg)) * 100), float(np.std(np.abs(err / hg)) / np.sqrt(n) * 100)],
            "Pearson": [float(r), float(np.sqrt((1 - r ** 2) / (n - 2))) if n > 2 else float("nan")],
            "SNR": [float(np.mean(snr)), float(np.std(snr) / np.sqrt(n))],
            "MACC": [float(np.mean(macc)), float(np.std(macc) / np.sqrt(n))]}


def balanced_overall(records, key="source"):
    """1:1 macro-average: aggregate per group, then average the per-group metric
    values with equal weight (each data part counts the same)."""
    groups = {}
    for r in records:
        groups.setdefault(r[key], []).append(r)
    per_group = {g: aggregate(rs) for g, rs in groups.items()}
    metrics = ["MAE", "RMSE", "MAPE", "Pearson", "SNR", "MACC"]
    out = {"n_groups": len(per_group), "groups": list(per_group)}
    for m in metrics:
        vals = [v[m][0] for v in per_group.values() if v and not np.isnan(v[m][0])]
        out[m] = float(np.mean(vals)) if vals else float("nan")
    return out, per_group


def group_table(records, key):
    groups = {}
    for r in records:
        groups.setdefault(r[key], []).append(r)
    return {g: aggregate(rs) for g, rs in groups.items()}


def run(method_name, manifest, fs, outdir):
    method = METHODS[method_name.lower()]
    rows = load_manifest(manifest)
    os.makedirs(outdir, exist_ok=True)

    records = []
    for r in rows:
        z = np.load(r["path"]); frames = z["frames"]; label = z["label"].astype(np.float64)
        bvp = np.asarray(method(frames, fs)).reshape(-1)
        m = min(len(bvp), len(label)); bvp, lab = bvp[:m], label[:m]
        # unsupervised BVP is a raw waveform (not a derivative) -> diff_flag=False
        hl, hp, snr, macc = mtb.calculate_metric_per_video(
            bvp.copy(), lab.copy(), fs=fs, diff_flag=False, hr_method="FFT")
        records.append({"subject": r["subject"], "env": r["env"], "source": r["source"],
                        "hr_pred": float(hp), "hr_gt": float(hl), "mae": float(abs(hp - hl)),
                        "snr": float(snr), "macc": float(macc), "pred": bvp, "gt": lab})

    overall = aggregate(records)
    bal_src, per_source = balanced_overall(records, key="source")
    per_env = group_table(records, key="env")

    # ---- save ----
    with open(os.path.join(outdir, f"per_recording_{method_name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "env", "source", "hr_pred", "hr_gt", "error_bpm", "MACC", "SNR_dB"])
        for r in sorted(records, key=lambda r: r["macc"]):
            w.writerow([r["subject"], r["env"], r["source"], f"{r['hr_pred']:.1f}",
                        f"{r['hr_gt']:.1f}", f"{r['hr_pred']-r['hr_gt']:.1f}",
                        f"{r['macc']:.3f}", f"{r['snr']:.1f}"])
    report = {"method": method_name,
              "overall_pooled": {k: v for k, v in overall.items()},
              "overall_balanced_1to1": bal_src,
              "per_source": {s: m for s, m in per_source.items()},
              "per_environment": {e: m for e, m in per_env.items()}}
    with open(os.path.join(outdir, f"report_{method_name}.json"), "w") as f:
        json.dump(report, f, indent=2)
    try:
        plots.waveform_grid(records, fs, os.path.join(outdir, f"waveforms_{method_name}.png"))
        plots.spectra(records, fs, os.path.join(outdir, f"spectra_{method_name}.png"))
    except Exception as e:
        print(f"[warn] plotting skipped: {e}")

    # ---- print ----
    def line(m):
        return (f"MAE {m['MAE'][0]:.2f}+/-{m['MAE'][1]:.2f}  RMSE {m['RMSE'][0]:.2f}  "
                f"MAPE {m['MAPE'][0]:.1f}%  Pearson {m['Pearson'][0]:.3f}  "
                f"SNR {m['SNR'][0]:.1f}dB  MACC {m['MACC'][0]:.3f}")
    print(f"\n=========== UNSUPERVISED: {method_name.upper()} ===========")
    print(f"  [overall pooled  ] n={overall['n']}  {line(overall)}")
    print(f"  [overall 1:1 bal ] groups={bal_src['groups']}  MAE {bal_src['MAE']:.2f}  "
          f"RMSE {bal_src['RMSE']:.2f}  Pearson {bal_src['Pearson']:.3f}  "
          f"SNR {bal_src['SNR']:.1f}dB  MACC {bal_src['MACC']:.3f}")
    print("  --- per source ---")
    for s, m in per_source.items():
        print(f"    {s:<10} n={m['n']}  {line(m)}")
    print("  --- per environment ---")
    for e, m in sorted(per_env.items(), key=lambda kv: kv[1]['MAE'][0], reverse=True):
        print(f"    {e:<12} n={m['n']}  {line(m)}")
    print(f"[saved] {outdir}/report_{method_name}.json")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS) + ["all"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fs", type=int, default=30)
    ap.add_argument("--outdir", default="runs/unsupervised")
    args = ap.parse_args()
    methods = list(METHODS) if args.method == "all" else [args.method]
    for mth in methods:
        run(mth, args.manifest, args.fs, args.outdir)


if __name__ == "__main__":
    main()
