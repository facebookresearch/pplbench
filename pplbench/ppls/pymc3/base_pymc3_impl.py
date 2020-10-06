# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import abstractmethod

import pymc3 as pm
import xarray as xr
from pymc3.backends.base import MultiTrace

from ..base_ppl_impl import BasePPLImplementation


class BasePyMC3Implementation(BasePPLImplementation):
    @abstractmethod
    def __init__(self, **model_attrs) -> None:
        """
        :param attrs: model arguments
        """
        ...

    @abstractmethod
    def get_model(self, data: xr.Dataset) -> pm.Model:
        """Take the data from the model generator and build a model instance that
        represents the statistical model
        :param data: A dataset from the model generator
        :returns: An pymc3.Model that can run inference on
        """
        ...

    @abstractmethod
    def extract_data_from_pymc3(self, samples: MultiTrace) -> xr.Dataset:
        """
        Takes the output of PyMC3 inference and converts into a format expected
        by PPLBench.
        :param samples: MultiTrace object returned by PyMC3 inference method
        :returns: a dataset over inferred parameters
        """
        ...
