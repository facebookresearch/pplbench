# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod
from typing import Dict, Type

import xarray as xr

from .base_ppl_impl import BasePPLImplementation


class BasePPLInference(ABC):
    @abstractmethod
    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        """
        Initialize the inference object
        :param impl_class: The class implementing a model in the PPL
        :param model_attrs: All of the model-specific attributes.
        """
        raise NotImplementedError

    @abstractmethod
    def compile(self, seed: int, **compile_args) -> None:
        """Compile the model representation if appropriate otherwise this can be a no-op."""
        raise NotImplementedError

    @abstractmethod
    def infer(
        self, data: xr.Dataset, num_samples: int, seed: int, **infer_args
    ) -> xr.Dataset:
        """Run inference and return samples"""
        raise NotImplementedError

    def additional_diagnostics(self, output_dir: str, prefix: str) -> None:
        """
        Save additional inference diagnostics to directory.
        This method is called after we have finished timing inference,
        so it doesn't affect the benchmark. It is optional to implement.
        :param output_dir: the directory to write the diagnostics
        :param prefix: a unique string identifying this inference object
        """
        pass
