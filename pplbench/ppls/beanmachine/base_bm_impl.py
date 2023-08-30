# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Dict, List

import xarray as xr
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.rv_identifier import RVIdentifier

from ..base_ppl_impl import BasePPLImplementation


class BaseBeanMachineImplementation(BasePPLImplementation):
    @abstractmethod
    def __init__(self, **model_attrs) -> None:
        ...

    @abstractmethod
    def data_to_observations(self, data: xr.Dataset) -> Dict:
        """Convert the model data into observation format used by Bean Machine"""
        ...

    @abstractmethod
    def get_queries(self) -> List[RVIdentifier]:
        """
        :returns: The list of random variables that we are interested in for a
        particular inference
        """
        ...

    @abstractmethod
    def extract_data_from_bm(self, samples: MonteCarloSamples) -> xr.Dataset:
        """Convert the result of inference into a dataset for PPLBench."""
        ...
