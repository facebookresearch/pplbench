# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import beanmachine.ppl as bm
import numpy as np
import torch
import torch.distributions as dist
import xarray as xr
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples

from .base_bm_impl import BaseBeanMachineImplementation


class RobustRegression(BaseBeanMachineImplementation):
    def __init__(
        self,
        n: int,
        k: int,
        alpha_scale: float,
        beta_scale: float,
        beta_loc: float,
        sigma_mean: float,
    ) -> None:
        """
        :param attrs: model arguments
        """
        self.n = n
        self.k = k
        self.alpha_scale = alpha_scale
        self.beta_scale: torch.Tensor = beta_scale * torch.ones(self.k)
        self.beta_loc: torch.Tensor = beta_loc * torch.ones(self.k)
        self.sigma_mean = sigma_mean

    @bm.random_variable
    def alpha(self) -> dist.Distribution:
        return dist.Normal(0.0, self.alpha_scale)

    @bm.random_variable
    def beta(self) -> dist.Distribution:
        return dist.Normal(self.beta_loc, self.beta_scale)

    @bm.random_variable
    def nu(self) -> dist.Distribution:
        return dist.Gamma(2.0, 0.1)

    @bm.random_variable
    def sigma(self) -> dist.Distribution:
        return dist.Exponential(1 / self.sigma_mean)

    @bm.random_variable
    def X(self) -> dist.Distribution:
        return dist.Normal(torch.zeros((self.n, self.k)), torch.ones((self.n, self.k)))

    @bm.random_variable
    def Y(self) -> dist.Distribution:
        mu = torch.mv(self.X(), self.beta()) + self.alpha()
        return dist.StudentT(self.nu(), mu, self.sigma())

    def data_to_observations(self, data: xr.Dataset) -> Dict:
        """
        Take data from the model generator and convert them to a dictionary that maps
        from random variables to observations, which could be used by Bean Machine.
        :param data: A dataset from the model generator
        :returns: a dictionary that maps random variables to their corresponding
        observations
        """
        # transpose the dataset to ensure that it is the way we expect
        data = data.transpose("item", "feature")

        X_val = torch.tensor(data.X.values, dtype=torch.get_default_dtype())
        Y_val = torch.tensor(data.Y.values, dtype=torch.get_default_dtype())

        return {self.X(): X_val, self.Y(): Y_val}

    def get_queries(self) -> List:
        return [self.alpha(), self.beta(), self.nu(), self.sigma()]

    def extract_data_from_bm(self, samples: MonteCarloSamples) -> xr.Dataset:
        """
        Takes the output of Bean Machine and converts into a format expected
        by PPLBench.
        :param samples: a MonteCarloSamples object returns by Bean Machine
        :returns: a dataset over inferred parameters
        """
        alpha_samples = (
            samples.get_variable(self.alpha(), include_adapt_steps=True)
            .numpy()
            .squeeze(0)
        )
        beta_samples = (
            samples.get_variable(self.beta(), include_adapt_steps=True)
            .numpy()
            .squeeze(0)
        )
        nu_samples = (
            samples.get_variable(self.nu(), include_adapt_steps=True).numpy().squeeze(0)
        )
        sigma_samples = (
            samples.get_variable(self.sigma(), include_adapt_steps=True)
            .numpy()
            .squeeze(0)
        )

        return xr.Dataset(
            {
                "alpha": (["draw"], alpha_samples),
                "beta": (["draw", "feature"], beta_samples),
                "nu": (["draw"], nu_samples),
                "sigma": (["draw"], sigma_samples),
            },
            coords={
                "draw": np.arange(alpha_samples.shape[0]),
                "feature": np.arange(beta_samples.shape[-1]),
            },
        )
