# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, Type

import beanmachine.graph as bmg
import numpy as np
import xarray as xr

from ....lib.ppl_profiler import start_profiling_BMG
from ...base_ppl_impl import BasePPLImplementation
from ...base_ppl_inference import BasePPLInference
from .base_bmgraph_impl import BaseBMGraphImplementation


class BaseBMGraphInference(BasePPLInference):
    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        # We always expect a BMGraph Implementation here
        self.impl_class = cast(Type[BaseBMGraphImplementation], impl_class)
        self.model_attrs = model_attrs

    def compile(self, seed: int, **compile_args):
        self.impl = self.impl_class(**self.model_attrs)


class NMC(BaseBMGraphInference):
    is_adaptive = False

    def infer(
        self,
        data: xr.Dataset,
        iterations: int,
        num_warmup: int,
        seed: int,
        **infer_args
    ) -> xr.Dataset:
        self.impl.bind_data_to_bmgraph(data)
        start_profiling_BMG(self.impl.graph)
        samples = np.array(
            self.impl.graph.infer(iterations, bmg.InferenceType.NMC, seed)
        )
        return self.impl.format_samples_from_bmgraph(samples)


class GlobalMCMC(BaseBMGraphInference):
    is_adaptive = True

    def infer(
        self,
        data: xr.Dataset,
        iterations: int,
        num_warmup: int,
        seed: int,
        algorithm: str = "NUTS",
        **infer_args
    ) -> xr.Dataset:
        self.impl.bind_data_to_bmgraph(data)
        inference_cls = getattr(bmg, algorithm)
        start_profiling_BMG(self.impl.graph)
        mcmc = inference_cls(self.impl.graph, *(infer_args.values()))
        samples = np.array(mcmc.infer(iterations, seed, num_warmup))
        return self.impl.format_samples_from_bmgraph(samples)
