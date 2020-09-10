# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np


def log1pexp(x: np.ndarray) -> np.ndarray:
    """
    Compute log(1 + exp(x)) in a numerically stable way,
    see https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf eqn 10
    :param x: numpy array of numbers
    :returns: log(1 + exp(x))
    """
    y = np.zeros_like(x)
    y[x < 18] = np.log1p(np.exp(x[x < 18]))
    y[x >= 18] = x[x >= 18] + np.exp(-x[x >= 18])
    return y
