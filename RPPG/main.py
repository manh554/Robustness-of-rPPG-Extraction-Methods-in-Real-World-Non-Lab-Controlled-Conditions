
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


CONFIG = dict(
    mode="preprocess",        

 
    root="../ubfc",             
    format="ubfc",             
    out="../cache_rgb",        
    source_tag="ubfc",         
    qc="on",                   
    size=72,
    face_backend="auto",     

 
    method="all",            
    manifest="../cache_rgb/manifest_rgb.csv",
    outdir="../runs/unsup",
    fs=30,
)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=CONFIG["mode"], choices=["preprocess", "run"])
    ap.add_argument("--root", default=CONFIG["root"])
    ap.add_argument("--format", default=CONFIG["format"], choices=["ubfc", "custom"])
    ap.add_argument("--out", default=CONFIG["out"])
    ap.add_argument("--size", type=int, default=CONFIG["size"])
    ap.add_argument("--source_tag", default=CONFIG["source_tag"])
    ap.add_argument("--qc", default=CONFIG["qc"], choices=["on", "off"])
    ap.add_argument("--face_backend", default=CONFIG["face_backend"],
                    choices=["auto", "mediapipe", "haar", "center"])
    ap.add_argument("--method", default=CONFIG["method"])
    ap.add_argument("--manifest", default=CONFIG["manifest"])
    ap.add_argument("--fs", type=int, default=CONFIG["fs"])
    ap.add_argument("--outdir", default=CONFIG["outdir"])
    args, _ = ap.parse_known_args()      # ignore Spyder's --wdir etc.

    if args.mode == "preprocess":
        argv = ["--root", args.root, "--format", args.format, "--out", args.out,
                "--size", str(args.size), "--face_backend", args.face_backend,
                "--qc", args.qc]
        if args.source_tag:
            argv += ["--source_tag", args.source_tag]
        sys.argv = ["rgb_preprocess.py"] + argv
        from dataset import rgb_preprocess
        rgb_preprocess.main()
    else:
        from predictor import main as run_main
        sys.argv = ["predictor.py", "--method", args.method, "--manifest", args.manifest,
                    "--fs", str(args.fs), "--outdir", args.outdir]
        run_main()


if __name__ == "__main__":
    main()
