# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import abstractmethod
from typing import Dict, List

import xarray as xr

from ..base_ppl_impl import BasePPLImplementation


class BaseJagsImplementation(BasePPLImplementation):
    @abstractmethod
    def __init__(self, **model_attrs) -> None:
        """
        :param attrs: model arguments
        """
        ...

    @abstractmethod
    def get_code(self) -> str:
        """Get Jags code that represents the statistical model"""
        ...

    @abstractmethod
    def get_vars(self) -> List[str]:
        """
        :returns: The list of variables that are needed from inference.
        """
        ...

    @abstractmethod
    def format_data_to_jags(self, data: xr.Dataset) -> Dict:
        """
        Take data from the model generator and the previously passed model
        arguments to construct a data dictionary to pass to JAGS.
        :param data: A dataset from the model generator
        :returns: a dictionary that can be passed to JAGS
        """
        ...

    @abstractmethod
    def extract_data_from_jags(self, samples: Dict) -> xr.Dataset:
        """
        Takes the output of JAGS and converts into a format expected
        by PPLBench.
        :param samples: samples dictionary from JAGS
        :returns: a dataset over inferred parameters
        """
        ...
