# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
from typing import Dict, List

import numpy as np
import xarray as xr

from .base_jags_impl import BaseJagsImplementation


LOGGER = logging.getLogger(__name__)


class NoisyOrTopic(BaseJagsImplementation):
    def __init__(self, **attrs: Dict) -> None:
        self.num_topics: int = attrs["num_topics"]  # type: ignore
        self.num_words: int = attrs["num_words"]  # type: ignore
        self.edge_weight = attrs["edge_weight"]

    def get_vars(self) -> List[str]:
        return ["active"]

    def format_data_to_jags(self, data: xr.Dataset) -> Dict:
        # transpose the dataset to ensure that it is the way we expect
        # we assume that there is only one sentence
        data = data.transpose("sentence", "word")
        return {"S": data.S.values[0]}

    def extract_data_from_jags(self, samples: Dict) -> xr.Dataset:
        # dim 2 is the chains dimension so we squeeze it out
        return xr.Dataset(
            {
                # active dimensions are [topic, draw], we want [draw, topic]
                "active": (["draw", "topic"], samples["active"].T)
            },
            coords={
                "draw": np.arange(samples["active"].shape[1]),
                "topic": np.arange(samples["active"].shape[0]),
            },
        )

    def get_code(self) -> str:
        code = "model {\n"
        # generate the node probabilities
        code += "  prob[1] <- 1\n"  # leak node
        for node in 1 + np.arange(self.num_topics + self.num_words):
            # note that JAGS has 1-based arrays so we have to add one
            code += f"  prob[{node+1}] <- 1 - exp(-("
            for par, wt in enumerate(self.edge_weight[node, :]):
                if wt:
                    code += f"active[{par+1}]*{wt} + "
            code += "0))\n"
        # generate the topic nodes
        code += "  active[1] <- 1\n"  # leak node
        for node in 1 + np.arange(self.num_topics):
            code += f"  active[{node+1}] ~ dbern(prob[{node+1}])\n"
        # generate the word nodes
        for node in 1 + self.num_topics + np.arange(self.num_words):
            code += f"  S[{node-self.num_topics}] ~ dbern(prob[{node+1}])\n"
        code += "}"
        LOGGER.debug(code)
        return code
