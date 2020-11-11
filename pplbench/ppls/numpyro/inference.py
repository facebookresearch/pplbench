# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Dict, Type, cast

import numpy as np
import numpyro.infer as infer
import xarray as xr
from jax import random

from ..base_ppl_impl import BasePPLImplementation
from ..base_ppl_inference import BasePPLInference
from .base_numpyro_impl import BaseNumPyroImplementation


class BaseNumPyroInference(BasePPLInference):
    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        self.impl_class = cast(Type[BaseNumPyroImplementation], impl_class)
        self.impl = self.impl_class(**model_attrs)


class MCMC(BaseNumPyroInference):
    def compile(self, seed: int, **compile_args):
        # TODO: Separating the compilation time might be desirable
        # for more complex models.
        pass

    def infer(
        self,
        data: xr.Dataset,
        iterations: int,
        num_warmup: int,
        seed: int,
        algorithm: str = "NUTS",
        **infer_args,
    ) -> xr.Dataset:
        """
        See :class:`numpyro.infer.mcmc.MCMC` for the MCMC inference API.
        """
        if algorithm == "NUTS":
            kernel = infer.NUTS(self.impl.model)
        else:
            raise ValueError(f"{algorithm} algorithm not registered for NumPyro.")
        mcmc = infer.MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=iterations - num_warmup,
            **infer_args,
        )
        # Note that we need to run warmup separately to collect samples
        # from the warmup phase.
        rng_key_1, rng_key_2 = random.split(random.PRNGKey(seed))
        mcmc.warmup(rng_key_1, data, collect_warmup=True)
        warmup_samples = mcmc.get_samples()
        mcmc.run(rng_key_2, data)
        # merge samples from the warmup phase
        samples = {
            k: np.concatenate([warmup_samples[k], v])
            for k, v in mcmc.get_samples().items()
        }
        return self.impl.extract_data_from_numpyro(samples)
