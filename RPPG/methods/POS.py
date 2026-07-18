"""POS -- Wang et al. (2017), Plane-Orthogonal-to-Skin.
Ported faithfully from ubicomplab/rPPG-Toolbox (MIT).
NOTE: np.mat was removed in NumPy 2.0 -> use np.asmatrix."""
import math
import numpy as np
from scipy import signal
from methods import utils


def POS(frames, fs):
    WinSec = 1.6
    RGB = utils.process_video(frames)   # [N,3] RGB
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)
    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.asmatrix(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for t in range(h.shape[1]):
                h[0, t] = h[0, t] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])
    BVP = utils.detrend(np.asmatrix(H).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype="bandpass")
    return signal.filtfilt(b, a, BVP.astype(np.double))
