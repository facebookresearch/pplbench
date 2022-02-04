# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod

import beanmachine.graph as bmg
import numpy as np
import xarray as xr

from ...base_ppl_impl import BasePPLImplementation


class BaseBMGraphImplementation(BasePPLImplementation):
    @property
    @abstractmethod
    def graph(self) -> bmg.Graph:
        ...

    @abstractmethod
    def __init__(self, **model_attrs) -> None:
        """
        Builds a BMGraph object that represent the statistical model and
        records queries.
        """
        ...

    @abstractmethod
    def bind_data_to_bmgraph(self, data: xr.Dataset) -> None:
        """Add observations to the model"""
        ...

    @abstractmethod
    def format_samples_from_bmgraph(self, samples: np.ndarray) -> xr.Dataset:
        """Convert the result of inference into a dataset for PPLBench."""
        ...
