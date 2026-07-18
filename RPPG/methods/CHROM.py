"""CHROM -- de Haan & Jeanne (2013), chrominance-based rPPG.
Ported faithfully from ubicomplab/rPPG-Toolbox (MIT)."""
import math
import numpy as np
from scipy import signal
from methods import utils


def CHROM(frames, fs):
    LPF, HPF, WinSec = 0.7, 2.5, 1.6
    RGB = utils.process_video(frames)   # [N,3] RGB
    FN = RGB.shape[0]
    nyq = 0.5 * fs
    B, A = signal.butter(3, [LPF / nyq, HPF / nyq], "bandpass")
    WinL = math.ceil(WinSec * fs)
    if WinL % 2:
        WinL += 1
    NWin = math.floor((FN - WinL // 2) / (WinL // 2))
    WinS, WinM, WinE = 0, WinL // 2, WinL
    S = np.zeros((WinL // 2) * (NWin + 1))
    for i in range(NWin):
        base = np.mean(RGB[WinS:WinE, :], axis=0)
        norm = np.zeros((WinE - WinS, 3))
        for t in range(WinS, WinE):
            norm[t - WinS] = np.true_divide(RGB[t], base)
        Xs = np.squeeze(3 * norm[:, 0] - 2 * norm[:, 1])
        Ys = np.squeeze(1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2])
        Xf = signal.filtfilt(B, A, Xs, axis=0)
        Yf = signal.filtfilt(B, A, Ys)
        alpha = np.std(Xf) / np.std(Yf)
        SWin = Xf - alpha * Yf
        SWin = np.multiply(SWin, signal.windows.hann(WinL))
        S[WinS:WinM] = S[WinS:WinM] + SWin[:WinL // 2]
        S[WinM:WinE] = SWin[WinL // 2:]
        WinS = WinM; WinM = WinS + WinL // 2; WinE = WinS + WinL
    return S
