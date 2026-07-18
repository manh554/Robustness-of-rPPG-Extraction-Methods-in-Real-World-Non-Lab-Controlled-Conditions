"""LGI -- Pilz et al. (2018), Local Group Invariance.
Faithful to ubicomplab/rPPG-Toolbox: single global SVD over the [3, N] RGB matrix
(NOT per-frame), then project out the dominant direction and take the green row."""
import numpy as np
from methods import utils


def LGI(frames, fs=None):
    data = utils.process_video_svd(frames)        # (1, 3, N) RGB
    U, _, _ = np.linalg.svd(data)                 # U: (1,3,3)
    S = U[:, :, 0]                                 # (1,3) dominant left-singular vec
    S = np.expand_dims(S, 2)                       # (1,3,1)
    SST = np.matmul(S, np.swapaxes(S, 1, 2))       # (1,3,3)
    P = np.tile(np.identity(3), (S.shape[0], 1, 1)) - SST
    Y = np.matmul(P, data)                         # (1,3,N)
    return Y[:, 1, :].reshape(-1)                  # green-row residual
