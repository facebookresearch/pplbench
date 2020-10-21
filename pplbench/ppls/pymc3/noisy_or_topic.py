# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict

import numpy as np
import pymc3 as pm
import xarray as xr
from pymc3.backends.base import MultiTrace

from .base_pymc3_impl import BasePyMC3Implementation


class NoisyOrTopic(BasePyMC3Implementation):
    def __init__(
        self, n: int, num_topics: int, num_words: int, edge_weight: Dict
    ) -> None:
        assert n == 1
        self.num_topics = num_topics
        self.num_words = num_words
        self.edge_weight = edge_weight

    def get_model(self, data: xr.Dataset) -> pm.Model:
        # transpose the dataset to ensure that it is the way we expect
        data = data.transpose("sentence", "word")
        active = [None for _ in range(1 + self.num_topics)]
        with pm.Model() as model:
            S = pm.Data("S_obs", data.S.values[0])
            active[0] = pm.Bernoulli("active[0]", p=1.0)
            for j in range(1, self.num_topics + 1):
                # note: if p = 1 - exp(-w) then logit(p) = log(1-exp(-w)) + w
                w = self.edge_weight[j, :j] @ active[:j]
                topic_logit = pm.math.log1mexp(w) + w
                active[j] = pm.Bernoulli(f"active[{j}]", logit_p=topic_logit)
            w = pm.math.dot(self.edge_weight[1 + self.num_topics :], active)
            word_logit = pm.math.log1mexp(w) + w
            pm.Bernoulli("S", logit_p=word_logit, observed=S, shape=self.num_words)

        return model

    def extract_data_from_pymc3(self, samples: MultiTrace) -> xr.Dataset:
        return xr.Dataset(
            {
                "active": (
                    ["draw", "topic"],
                    np.concatenate(
                        tuple(
                            np.expand_dims(samples[f"active[{j}]"], 1)
                            for j in range(1 + self.num_topics)
                        ),
                        axis=1,
                    ),
                )
            },
            coords={
                "draw": np.arange(len(samples["active[0]"])),
                "topic": np.arange(1 + self.num_topics),
            },
        )
