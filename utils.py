# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from torch import Tensor


def _compute_var(query_samples: Tensor) -> Tuple[Tensor, Tensor]:
    n_chains, n_samples = query_samples.shape[:2]
    if n_chains > 1:
        per_chain_avg = query_samples.mean(1)
        b = n_samples * torch.var(per_chain_avg, dim=0)
    else:
        b = 0
    w = torch.mean(torch.var(query_samples, dim=1), dim=0)
    var_hat = (n_samples - 1) / n_samples * w + (1 / n_samples) * b
    return w, var_hat


def split_r_hat(query_samples: Tensor) -> Optional[Tensor]:
    """
    Computes r_hat given query_samples
    :param query_samples: samples of shape (num_chains, iterations) from the posterior.
    """
    n_chains, n_samples = query_samples.shape[:2]
    if n_chains < 2:
        return None
    n_chains = n_chains * 2
    n_samples = n_samples // 2
    query_samples = torch.cat(torch.split(query_samples, n_samples, dim=1)[0:2])
    w, var_hat = _compute_var(query_samples)
    return torch.sqrt(var_hat / w)


def effective_sample_size(query_samples: Tensor) -> Tensor:
    """
    Computes effective sample size given query_samples
    :param query_samples: samples of shape (num_chains, iterations) from the posterior
    """
    n_chains, n_samples, *query_dim = query_samples.shape

    samples = query_samples - query_samples.mean(dim=1, keepdim=True)
    samples = samples.transpose(1, -1)
    # computes fourier transform (with padding)
    padding = torch.zeros(samples.shape, dtype=samples.dtype)
    padded_samples = torch.cat((samples, padding), dim=-1)
    fvi = torch.rfft(padded_samples, 1, onesided=False)
    # multiply by complex conjugate
    acf = fvi.pow(2).sum(-1, keepdim=True)
    # transform back to reals (with padding)
    padding = torch.zeros(acf.shape, dtype=acf.dtype)
    padded_acf = torch.cat((acf, padding), dim=-1)
    rho_per_chain = torch.irfft(padded_acf, 1, onesided=False)

    rho_per_chain = rho_per_chain.narrow(-1, 0, n_samples)
    num_per_lag = torch.tensor(range(n_samples, 0, -1), dtype=samples.dtype)
    rho_per_chain = torch.div(rho_per_chain, num_per_lag)
    rho_per_chain = rho_per_chain.transpose(1, -1)

    rho_avg = rho_per_chain.mean(dim=0)
    w, var_hat = _compute_var(query_samples)
    if n_chains > 1:
        rho = 1 - ((w - rho_avg) / var_hat)
    else:
        rho = rho_avg / var_hat
    rho[0] = 1

    # reshape to 2d matrix where each row contains all samples for specific dim
    rho_2d = torch.stack(torch.unbind(rho, dim=0), dim=-1).reshape(-1, n_samples)
    rho_sum = torch.zeros(rho_2d.shape[0])

    for i, chain in enumerate(torch.unbind(rho_2d, dim=0)):
        total_sum = torch.tensor(0.0, dtype=samples.dtype)
        for t in range(n_samples // 2):
            rho_even = chain[2 * t]
            rho_odd = chain[2 * t + 1]
            if rho_even + rho_odd < 0:
                break
            else:
                total_sum += rho_even + rho_odd
        rho_sum[i] = total_sum

    rho_sum = torch.reshape(rho_sum, query_dim)
    return torch.div(n_chains * n_samples, -1 + 2 * rho_sum)
