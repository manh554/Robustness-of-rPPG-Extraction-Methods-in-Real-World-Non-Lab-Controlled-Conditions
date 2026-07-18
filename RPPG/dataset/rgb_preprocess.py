"""
rgb_preprocess.py -- preprocessing for UNSUPERVISED methods (CHROM/POS/LGI).

These methods need RAW face frames (they compute their own RGB traces), so we
crop the face with the SAME FaceCropper used by the supervised repo (MediaPipe
Tasks -> Haar -> center) and save raw frames + ground-truth PPG, one .npz per
recording. This keeps the ROI identical to the supervised side for a fair
comparison.

Manifest (manifest_rgb.csv): path, source, subject, env
  subject = recording-name prefix before first '_'  (sub01_rest -> sub01)
  env     = the rest after the subject id            (sub01_rest -> rest)
  source  = the --source_tag (e.g. ubfc / own)       -> used for 1:1 balancing
"""
import os
import glob
import argparse
import cv2
import numpy as np

from dataset.face_detector import FaceCropper


def read_ubfc_ppg(path):
    with open(path) as f:
        rows = [r.split() for r in f.read().strip().split("\n")]
    return np.asarray(rows[0], dtype=np.float64)


def read_ppg_csv(path):
    """Raw MAX30102 ppg.csv (sample_idx, ir, t_pc) -> clean IR sequence (dedup by
    sample_idx). QC upstream rejects resets/heavy-dup recordings. Resampled to the
    video frame count later (run align.py for precise time-sync)."""
    import csv
    si, ir = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            si.append(int(float(r["sample_idx"]))); ir.append(float(r["ir"]))
    si = np.asarray(si); ir = np.asarray(ir, dtype=np.float64)
    _, idx = np.unique(si, return_index=True)
    return ir[idx].astype(np.float64)


def read_label(path):
    if path.endswith(".csv"):
        return read_ppg_csv(path)
    if os.path.basename(path) == "ground_truth.txt":
        return read_ubfc_ppg(path)
    return np.loadtxt(path).astype(np.float64)


def read_custom_ppg(path):
    return read_label(path)


def resample_ppg(ppg, target_len):
    return np.interp(np.linspace(1, len(ppg), target_len),
                     np.linspace(1, len(ppg), len(ppg)), ppg)


def _find_video(d):
    for pat in ("vid.avi", "video.*"):
        g = sorted(glob.glob(os.path.join(d, pat)))
        if g:
            return g[0]
    for ext in ("*.avi", "*.mkv", "*.mp4", "*.mov", "*.webm",
                "*.AVI", "*.MKV", "*.MP4", "*.MOV"):
        g = sorted(glob.glob(os.path.join(d, ext)))
        if g:
            return g[0]
    return None


def _find_label(d):
    for name in ("ground_truth.txt", "ppg.txt"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    cands = []
    for c in sorted(glob.glob(os.path.join(d, "*.csv"))):
        b = os.path.basename(c).lower()
        if b in ("frames.csv", "qc_report.csv") or b.startswith("manifest"):
            continue
        cands.append(c)
    ppgs = [c for c in cands if "ppg" in os.path.basename(c).lower()]
    if ppgs:
        return ppgs[0]
    return cands[0] if cands else None


def diagnose(root):
    lines = [f"  (abs path: {os.path.abspath(root)})"]
    entries = sorted(glob.glob(os.path.join(root, "*")))
    if not entries:
        lines.append("  -> folder is EMPTY or the path is wrong")
        return "\n".join(lines)
    for d in entries[:6]:
        if os.path.isdir(d):
            files = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(d, "*")))][:8]
            lines.append(f"  {os.path.basename(d)}/  ->  {files}")
        else:
            lines.append(f"  {os.path.basename(d)}  (this is a FILE, not a per-recording folder)")
    return "\n".join(lines)


def discover(root, fmt):
    clips = []
    folders = (sorted(glob.glob(os.path.join(root, "subject*"))) if fmt == "ubfc"
               else [d for d in sorted(glob.glob(os.path.join(root, "*"))) if os.path.isdir(d)])
    for d in folders:
        vid = _find_video(d); label = _find_label(d)
        if vid and label:
            clips.append((os.path.basename(d), vid, label))
    return clips


def subject_id_of(rec):
    return rec.split("_")[0]


def env_of(rec, subject):
    rest = rec[len(subject):].lstrip("_")
    return rest if rest else "all"


def read_faces(video_path, cropper):
    cap = cv2.VideoCapture(video_path); frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cropper.crop(fr))
    cap.release()
    return np.asarray(frames, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--format", choices=["ubfc", "custom"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=72)
    ap.add_argument("--source_tag", default=None)
    ap.add_argument("--qc", default="on", choices=["on", "off"])
    ap.add_argument("--face_backend", default="auto",
                    choices=["auto", "mediapipe", "haar", "center"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = args.source_tag or args.format
    cropper = FaceCropper(size=args.size, backend=args.face_backend)
    clips = discover(args.root, args.format)
    if not clips:
        raise SystemExit(f"No clips under {args.root} (format={args.format}).\n"
                         f"Each recording needs a folder with a VIDEO (.avi/.mkv/.mp4) and a\n"
                         f"LABEL (ground_truth.txt / ppg.txt / ppg.csv). What I see:\n"
                         + diagnose(args.root))

    if args.qc == "on":
        from dataset.qc import qc_scan
        print(f"[QC] scanning {len(clips)} recordings for noise/bad data...")
        kept = qc_scan(clips, out_csv=os.path.join(args.out, "qc_report.csv"),
                       fps_label=30)
        clips = [c for c in clips if c[0] in kept]
        if not clips:
            raise SystemExit("QC rejected all recordings (see qc_report.csv)")

    rows = []
    for rec, vid, label_path in clips:
        subject = subject_id_of(rec); env = env_of(rec, subject)
        frames = read_faces(vid, cropper)
        ppg = read_label(label_path)
        ppg = resample_ppg(ppg.astype(np.float64), len(frames)).astype(np.float32)
        p = os.path.join(args.out, f"{tag}_{rec}.npz")
        np.savez_compressed(p, frames=frames, label=ppg)
        rows.append(f"{p},{tag},{subject},{env}")
        print(f"[ok] {rec} (subject={subject}, env={env}): {len(frames)} frames")

    cropper.close()
    man = os.path.join(args.out, "manifest_rgb.csv")
    mode = "a" if os.path.exists(man) else "w"
    with open(man, mode) as f:
        if mode == "w":
            f.write("path,source,subject,env\n")
        f.write("\n".join(rows) + "\n")
    print(f"[done] {tag}: {len(clips)} recordings -> {man} (raw RGB, size={args.size})")


if __name__ == "__main__":
    main()
