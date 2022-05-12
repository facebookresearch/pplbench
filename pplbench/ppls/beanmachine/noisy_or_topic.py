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

class NoisyOrTopic(BaseBeanMachineImplementation):
    def __init__(
        self,
        n: int,
        num_topics: int,
        num_words: int,
        edge_weight: Dict
        ) -> None:
        assert n == 1
        self.num_topics = num_topics
        self.num_words = num_words
        self.edge_weight = edge_weight

    @bm.random_variable
    def node(self, i):
        parent_accumulator = torch.tensor(0.0)
        for par, wt in enumerate(self.edge_weight[i, :]):
            if wt:
                parent_accumulator += self.node(par) * wt
        prob = 1 - torch.exp(-1 * parent_accumulator)
        return dist.Bernoulli(probs=prob)

    def data_to_observations(self, data: xr.Dataset) -> Dict:
        """
        Take data from the model generator and convert them to a dictionary that maps
        from random variables to observations, which could be used by Bean Machine.
        :param data: A dataset from the model generator
        :returns: a dictionary that maps random variables to their corresponding
        observations
        """
        S = {self.node(k + self.num_topics): torch.tensor(data.S.values[0][k], dtype=torch.get_default_dtype()) for k in range(self.num_words)}
        S[self.node(0)] = torch.tensor(1.0)
        return S

    def get_queries(self) -> List:
        return [self.node(i) for i in range(self.num_topics + 1)]


    def extract_data_from_bm(self, samples: MonteCarloSamples) -> xr.Dataset:
        """
        Takes the output of Bean Machine and converts into a format expected
        by PPLBench.
        :param samples: a MonteCarloSamples object returns by Bean Machine
        :returns: a dataset over inferred parameters
        """
        active0_samples = (
            samples.get_variable(self.node(0), include_adapt_steps=True)
            .numpy()
            .squeeze(0)
        )

        activej_samples = (
            np.concatenate(
                tuple(
                    np.expand_dims(
                        samples.get_variable(self.node(j), include_adapt_steps=True)
                        .numpy()
                        .squeeze(0),
                        1
                    )
                    for j in range(1 + self.num_topics)
                ),
                axis=1,
            )
        )

        return xr.Dataset(
            {
                "active": (["draw", "topic"], activej_samples)
            },
            coords={
                "draw": np.arange(len(active0_samples)),
                "topic": np.arange(1 + self.num_topics),
            },
        )
