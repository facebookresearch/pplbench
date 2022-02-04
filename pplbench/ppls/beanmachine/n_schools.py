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


class NSchools(BaseBeanMachineImplementation):
    def __init__(
        self,
        n: int,
        num_states: int,
        num_districts_per_state: int,
        num_types: int,
        scale_state: float,
        scale_district: float,
        scale_type: float,
        dof_baseline: float,
        scale_baseline: float,
        state_idx: np.ndarray,
        district_idx: np.ndarray,
        type_idx: np.ndarray,
    ) -> None:
        self.n = n
        self.num_states = num_states
        self.num_districts_per_state = num_districts_per_state
        self.num_types = num_types
        self.scale_state = scale_state
        self.scale_district = scale_district
        self.scale_type = scale_type
        self.dof_baseline = dof_baseline
        self.scale_baseline = scale_baseline
        self.state_idx = state_idx
        self.district_idx = district_idx
        self.type_idx = type_idx

    @bm.random_variable
    def sigma_state(self) -> dist.Distribution:
        return dist.HalfCauchy(self.scale_state)

    @bm.random_variable
    def sigma_district(self) -> dist.Distribution:
        return dist.HalfCauchy(self.scale_district)

    @bm.random_variable
    def sigma_type(self) -> dist.Distribution:
        return dist.HalfCauchy(self.scale_type)

    @bm.random_variable
    def beta_baseline(self) -> dist.Distribution:
        return dist.StudentT(self.dof_baseline, 0.0, self.scale_baseline)

    @bm.random_variable
    def beta_state(self) -> dist.Distribution:
        return dist.Normal(0.0, self.sigma_state()).expand((self.num_states,))

    @bm.random_variable
    def beta_type(self) -> dist.Distribution:
        return dist.Normal(0.0, self.sigma_type()).expand((self.num_types,))

    @bm.random_variable
    def beta_district(self) -> dist.Distribution:
        return dist.Normal(0.0, self.sigma_district()).expand(
            (self.num_states, self.num_districts_per_state)
        )

    @bm.random_variable
    def sigma(self) -> dist.Distribution:
        return dist.Uniform(0.5, 1.5).expand((self.n,))

    @bm.random_variable
    def Y(self) -> dist.Distribution:
        Yhat = (
            self.beta_baseline().expand((self.n,))
            + self.beta_state()[self.state_idx]
            + self.beta_district()[self.state_idx, self.district_idx]
            + self.beta_type()[self.type_idx]
        )
        return dist.Normal(Yhat, self.sigma())

    def data_to_observations(self, data: xr.Dataset) -> Dict:
        """
        Take data from the model generator and convert them to a dictionary that maps
        from random variables to observations, which could be used by Bean Machine.
        :param data: A dataset from the model generator
        :returns: a dictionary that maps random variables to their corresponding
        observations
        """
        sigma_val = torch.tensor(data.sigma.values, dtype=torch.get_default_dtype())
        Y_val = torch.tensor(data.Y.values, dtype=torch.get_default_dtype())

        return {self.sigma(): sigma_val, self.Y(): Y_val}

    def get_queries(self) -> List:
        return [
            self.sigma_state(),
            self.sigma_district(),
            self.sigma_type(),
            self.beta_baseline(),
            self.beta_state(),
            self.beta_district(),
            self.beta_type(),
        ]

    def extract_data_from_bm(self, samples: MonteCarloSamples) -> xr.Dataset:
        """
        Takes the output of Bean Machine and converts into a format expected
        by PPLBench.
        :param samples: a MonteCarloSamples object returns by Bean Machine
        :returns: a dataset over inferred parameters
        """
        numpy_samples = {}
        for node in samples:
            numpy_samples[node] = (
                samples.get_variable(node, include_adapt_steps=True).squeeze(0).numpy()
            )

        return xr.Dataset(
            {
                "sigma_state": (["draw"], numpy_samples[self.sigma_state()]),
                "sigma_district": (["draw"], numpy_samples[self.sigma_district()]),
                "sigma_type": (["draw"], numpy_samples[self.sigma_type()]),
                "beta_baseline": (["draw"], numpy_samples[self.beta_baseline()]),
                "beta_state": (["draw", "state"], numpy_samples[self.beta_state()]),
                "beta_district": (
                    ["draw", "state", "district"],
                    numpy_samples[self.beta_district()],
                ),
                "beta_type": (["draw", "type"], numpy_samples[self.beta_type()]),
            },
            coords={
                "draw": np.arange(samples.get_num_samples(include_adapt_steps=True)),
                "state": np.arange(self.num_states),
                "district": np.arange(self.num_districts_per_state),
                "type": np.arange(self.num_types),
            },
        )
