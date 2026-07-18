"""Shared utilities for unsupervised rPPG methods (ported from rPPG-Toolbox).

IMPORTANT: frames in the cache are BGR (OpenCV / face_detector default). The
chrominance methods (CHROM, POS) and LGI are defined for RGB, so process_video
flips BGR->RGB to match the toolbox convention.
"""
import numpy as np
from scipy.sparse import spdiags


def process_video(frames):
    """Per-frame spatial-average colour. Input frames are BGR (cv2) -> returns
    [N, 3] in RGB order."""
    RGB = []
    for frame in frames:
        s = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(s / (frame.shape[0] * frame.shape[1]))
    RGB = np.asarray(RGB)            # [N,3] in B,G,R order (cv2)
    return RGB[:, ::-1]              # -> [N,3] in R,G,B order


def process_video_svd(frames):
    """RGB as (1, 3, N) -- the shape LGI's single global SVD expects."""
    RGB = process_video(frames)                 # [N,3] RGB
    return RGB.transpose(1, 0).reshape(1, 3, -1)  # (1,3,N)


def detrend(input_signal, lambda_value):
    n = input_signal.shape[0]
    H = np.identity(n)
    ones = np.ones(n)
    minus_twos = -2 * np.ones(n)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (n - 2), n).toarray()
    return np.dot((H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))),
                  input_signal)
