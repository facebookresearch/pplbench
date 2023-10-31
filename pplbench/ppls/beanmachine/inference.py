# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, Type

import beanmachine.ppl as bm
import torch
import xarray as xr
from beanmachine.ppl import inference
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.inference.vi.variational_world import VariationalWorld

from ..base_ppl_impl import BasePPLImplementation
from ..base_ppl_inference import BasePPLInference
from .base_bm_impl import BaseBeanMachineImplementation


class BaseBMInference(BasePPLInference):
    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        torch.set_default_dtype(torch.float64)
        # We always expect a Bean Machine Implementation here
        self.impl_class = cast(Type[BaseBeanMachineImplementation], impl_class)
        self.impl = self.impl_class(**model_attrs)

    def compile(self, **kwargs):
        pass


class MCMC(BaseBMInference):
    def infer(
        self,
        data: xr.Dataset,
        iterations: int,
        num_warmup: int,
        seed: int,
        algorithm: str = "CompositionalInference",
        **infer_args,
    ) -> xr.Dataset:
        bm.seed(seed)

        # Dynamically choosing the algorithm from inference module
        inference_cls = getattr(inference, algorithm)

        samples = inference_cls(**infer_args).infer(
            queries=self.impl.get_queries(),
            observations=self.impl.data_to_observations(data),
            num_samples=iterations - num_warmup,
            num_adaptive_samples=num_warmup,
            num_chains=1,
        )
        return self.impl.extract_data_from_bm(samples)


class VI(BaseBMInference):
    def infer(
        self,
        data: xr.Dataset,
        iterations: int,
        num_warmup: int,
        seed: int,
        algorithm: str = "ADVI",
        **infer_args,
    ) -> xr.Dataset:
        bm.seed(seed)
        inference_cls = getattr(inference.vi, algorithm)
        if not issubclass(inference_cls, inference.vi.autoguide.AutoGuideVI):
            raise ValueError("Only autoguide methods are supported in PPL Bench")

        vi_world = inference_cls(
            queries=self.impl.get_queries(),
            observations=self.impl.data_to_observations(data),
            **infer_args,
        ).infer(num_steps=iterations)

        samples = self._draw_mc_samples_from_vi_world(vi_world, iterations)

        return self.impl.extract_data_from_bm(samples)

    def _draw_mc_samples_from_vi_world(
        self, vi_world: VariationalWorld, num_samples: int
    ) -> MonteCarloSamples:
        samples = {}
        for query in self.impl.get_queries():
            samples[query] = (
                vi_world.get_guide_distribution(query)
                .sample((num_samples,))
                .clone()
                .detach()
            )
        return MonteCarloSamples([samples])
