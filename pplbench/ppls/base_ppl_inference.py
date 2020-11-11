# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod
from typing import Dict, Type

import xarray as xr

from .base_ppl_impl import BasePPLImplementation


class BasePPLInference(ABC):
    """
    The base class of all PPL inference methods.

    Attributes:
        is_adaptive: A boolean indicates whether the inference method is adaptive or
        not. The default value for is_adaptive is True. See infer() to see how this
        attribute should be used.
    """

    is_adaptive = True

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
        self,
        data: xr.Dataset,
        iterations: int,
        num_warmup: int,
        seed: int,
        **infer_args
    ) -> xr.Dataset:
        """
        Run inference and return samples. The number of samples returned by this method
        should equal to iterations. If is_adaptive is True for the class, PPL Bench
        will assume that the samples with indicies [0, num_warmup) are warm up and
        samples in [num_warmup, iterations) will be used to compute diagnostic metrics.
        Algorithms that do not use warm up samples should set is_adaptive to False when
        extending this base class and could ignore num_warmup parameter when overriding
        infer(). When is_adaptive is set to False, all samples in [0, iterations) are
        treated as valid samples and will be included in diagnostics.
        """
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
