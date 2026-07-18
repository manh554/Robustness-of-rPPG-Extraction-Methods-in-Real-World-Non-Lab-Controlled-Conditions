import argparse
import csv
import os
import platform
import statistics
import threading
import time

import cv2

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

try:
    import numpy as np
    import scipy.signal
except ImportError:
    np = None
    scipy = None


PORT = "auto"
BAUD = 115200
OUT = "raw_custom/subject01"
CAM = 0
DURATION = 10
FPS = 30
WIDTH = 640
HEIGHT = 480
NO_PREVIEW = False

LOCK_CAMERA = True
EXPOSURE = -6
GAIN = None
WB_TEMP = 4600
FOCUS = 0
SETTLE_SEC = 1.5

REQUIRE_FINGER = True
FINGER_IR_THRESHOLD = 50000
FINGER_HOLD_SEC = 1.0
WAIT_TIMEOUT_SEC = 120

PPG_RATE = 100
BANDPASS = False


def resolve_port(requested):
    ports = list(list_ports.comports()) if list_ports else []
    if requested not in ("auto", "ask"):
        return requested
    if not ports:
        raise SystemExit("No serial ports found.")

    def looks_arduino(p):
        s = f"{p.description} {p.manufacturer or ''} {p.product or ''}".lower()
        return any(k in s for k in ("arduino", "ch340", "ch9102", "cp210",
                                    "usb serial", "ftdi", "wch"))

    if requested == "auto":
        candidates = [p for p in ports if looks_arduino(p)]
        return (candidates or ports)[0].device

    for i, p in enumerate(ports):
        print(f"[{i}] {p.device}  {p.description}")
    return ports[int(input("Port: "))].device


def open_camera(cam, width, height, fps):
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cam, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {cam}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if LOCK_CAMERA:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,
                0.25 if platform.system() == "Windows" else 1)
        if EXPOSURE is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if FOCUS is not None:
            cap.set(cv2.CAP_PROP_FOCUS, FOCUS)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if WB_TEMP is not None:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, WB_TEMP)
        if GAIN is not None:
            cap.set(cv2.CAP_PROP_GAIN, GAIN)
        t_end = time.perf_counter() + SETTLE_SEC
        while time.perf_counter() < t_end:
            cap.read()
    return cap


class SerialReader(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.ser = serial.Serial(port, baud, timeout=1)
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        self.ser.reset_input_buffer()
        while not self._stop.is_set():
            try:
                raw = self.ser.readline().decode("ascii", "ignore").strip()
            except Exception:
                continue
            t = time.perf_counter()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split(",")
            if len(parts) != 2:
                continue
            try:
                self.samples.append((int(parts[0]), int(parts[1]), t))
            except ValueError:
                continue

    def stop(self):
        self._stop.set()
        time.sleep(0.2)
        try:
            self.ser.close()
        except Exception:
            pass


def current_ir(reader, n=20):
    s = reader.samples[-n:]
    return float(statistics.median(x[1] for x in s)) if s else 0.0


def wait_for_finger(cap, reader, no_preview):
    hold_start = None
    t_timeout = (time.perf_counter() + WAIT_TIMEOUT_SEC) if WAIT_TIMEOUT_SEC > 0 else None
    exposure = EXPOSURE if EXPOSURE is not None else cap.get(cv2.CAP_PROP_EXPOSURE)
    gain = cap.get(cv2.CAP_PROP_GAIN)

    while True:
        ok, frame = cap.read()
        ir = current_ir(reader)
        present = ir > FINGER_IR_THRESHOLD
        if present:
            hold_start = hold_start or time.perf_counter()
            held = time.perf_counter() - hold_start
        else:
            hold_start = None
            held = 0.0

        if ok and not no_preview:
            msg = (f"FINGER OK {held:0.1f}/{FINGER_HOLD_SEC:0.1f}s" if present
                   else "WAITING FOR FINGER")
            color = (0, 200, 0) if present else (0, 0, 255)
            cv2.putText(frame, f"{msg}  IR={int(ir)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("recording", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise SystemExit("Cancelled.")
            elif key in (ord("+"), ord("=")):
                exposure += 1
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            elif key in (ord("-"), ord("_")):
                exposure -= 1
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            elif key == ord("]"):
                gain += 1
                cap.set(cv2.CAP_PROP_GAIN, gain)
            elif key == ord("["):
                gain = max(0, gain - 1)
                cap.set(cv2.CAP_PROP_GAIN, gain)

        if present and held >= FINGER_HOLD_SEC:
            return
        if t_timeout and time.perf_counter() > t_timeout:
            raise SystemExit(f"Finger timeout after {WAIT_TIMEOUT_SEC}s (IR={int(ir)}).")


def _next_pow2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def rolling_fft_hr(ppg, fs, win_s=10, low=0.75, high=2.5):
    n = len(ppg)
    win = int(win_s * fs)
    hr = np.zeros(n)
    for i in range(n):
        a = max(0, i - win // 2)
        b = min(n, a + win)
        a = max(0, b - win)
        seg = ppg[a:b] - np.mean(ppg[a:b])
        if len(seg) < fs * 2:
            continue
        f, pxx = scipy.signal.periodogram(seg, fs=fs, nfft=_next_pow2(len(seg)),
                                          detrend=False)
        mask = (f >= low) & (f <= high)
        if np.any(mask):
            hr[i] = f[mask][np.argmax(pxx[mask])] * 60
    return hr


def record(cap, writer, args):
    frames = []
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        t = time.perf_counter()
        if not ok:
            continue
        writer.write(frame)
        frames.append((len(frames), t))
        elapsed = t - t0

        if not args.no_preview:
            cv2.putText(frame, f"REC {elapsed:5.1f}/{args.duration:.0f}s",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("recording", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if elapsed >= args.duration:
            break
    return frames


def align_ppg_to_frames(frames, samples, ppg_rate):
    ft = np.array([f[1] for f in frames])
    si = np.array([s[0] for s in samples])
    ir = np.array([s[1] for s in samples])
    pt_pc = np.array([s[2] for s in samples])

    # MCU index gives a jitter-free time axis; median offset maps it to PC clock
    ppg_t = si / ppg_rate + np.median(pt_pc - si / ppg_rate)
    keep = np.concatenate(([True], np.diff(ppg_t) > 0))
    aligned = np.interp(ft, ppg_t[keep], ir[keep]).astype(np.float64)
    return aligned - np.mean(aligned)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--baud", type=int, default=BAUD)
    ap.add_argument("--cam", type=int, default=CAM)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--no_preview", action="store_true", default=NO_PREVIEW)
    ap.add_argument("--ppg_rate", type=float, default=PPG_RATE)
    ap.add_argument("--bandpass", action="store_true", default=BANDPASS)
    return ap.parse_args()


def main():
    args = parse_args()
    if serial is None:
        raise SystemExit("pip install pyserial")
    if np is None or scipy is None:
        raise SystemExit("pip install numpy scipy")

    if os.path.exists(os.path.join(args.out, "video.avi")):
        ans = input(f"'{args.out}' already contains data. Overwrite? [y/N] ")
        if ans.strip().lower() != "y":
            raise SystemExit("Aborted.")
    os.makedirs(args.out, exist_ok=True)
    port = resolve_port(args.port)
    cap = open_camera(args.cam, args.width, args.height, args.fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(os.path.join(args.out, "video.avi"),
                             cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (w, h))

    reader = SerialReader(port, args.baud)
    reader.start()
    time.sleep(2.0)  # sensor warm-up

    try:
        if REQUIRE_FINGER:
            wait_for_finger(cap, reader, args.no_preview)
        frames = record(cap, writer, args)
    finally:
        reader.stop()
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    with open(os.path.join(args.out, "frames.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame_idx", "t_pc"])
        wr.writerows(frames)
    with open(os.path.join(args.out, "ppg.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["sample_idx", "ir", "t_pc"])
        wr.writerows(reader.samples)

    if not frames or not reader.samples:
        raise SystemExit("Not enough data to align.")
    dur = frames[-1][1] - frames[0][1] if len(frames) > 1 else 0
    eff_fps = (len(frames) - 1) / dur if dur > 0 else 0
    if eff_fps <= 0:
        raise SystemExit("Effective fps is 0.")

    aligned = align_ppg_to_frames(frames, reader.samples, args.ppg_rate)
    if args.bandpass:
        b, a = scipy.signal.butter(3, [0.7, 3.0], btype="band", fs=eff_fps)
        aligned = scipy.signal.filtfilt(b, a, aligned)

    np.savetxt(os.path.join(args.out, "ppg.txt"), aligned, fmt="%.6f")
    gt = np.vstack([aligned, rolling_fft_hr(aligned, eff_fps),
                    np.arange(len(aligned)) / eff_fps])
    np.savetxt(os.path.join(args.out, "ground_truth.txt"), gt, fmt="%.6f")

    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"effective_fps={eff_fps:.4f}\n")
        f.write(f"n_frames={len(frames)}\n")
        f.write(f"n_ppg_samples={len(reader.samples)}\n")
        f.write(f"clip_seconds={dur:.2f}\n")

    print(f"done: {args.out}  fps={eff_fps:.2f}  "
          f"frames={len(frames)}  ppg={len(reader.samples)}")


if __name__ == "__main__":
    main()