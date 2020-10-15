# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, Optional, Type, cast

import pyjags
import xarray as xr

from ..base_ppl_impl import BasePPLImplementation
from ..base_ppl_inference import BasePPLInference
from .base_jags_impl import BaseJagsImplementation


class MCMC(BasePPLInference):
    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        # We always expect a BaseJagsImplementation here
        self.impl_class = cast(Type[BaseJagsImplementation], impl_class)
        self.impl = self.impl_class(**model_attrs)

    def compile(self, seed: int, **compile_args):
        # JAGS doesn't have a separate compile step.
        # The model construction requires the actual data,
        # so everything has to be done under inference.
        pass

    def infer(  # type: ignore
        self,
        data: xr.Dataset,
        num_samples: int,
        seed: int,
        adapt: Optional[int] = None,
        RNG_name: str = "base::Mersenne-Twister",
    ) -> xr.Dataset:
        """
        See https://phoenixnap.dl.sourceforge.net/project/mcmc-jags/Manuals/4.x/jags_user_manual.pdf
        for JAGS documentation.

        :param data: PPLBench dataset
        :param num_samples: number of samples to create
        :param seed: seed for random number generator
        :param adapt: the number of adaptive steps
        :param RNG_name: the name of the random number generator
        :returns: samples dataset
        """
        if adapt is None:
            adapt = num_samples

        model = pyjags.Model(
            code=self.impl.get_code(),
            data=self.impl.format_data_to_jags(data),
            chains=1,
            adapt=adapt,
            init={".RNG.seed": seed, ".RNG.name": RNG_name},
        )
        samples = model.sample(num_samples, vars=self.impl.get_vars())
        # squeeze out the chain dimension from the samples
        for varname in samples.keys():
            samples[varname] = samples[varname].squeeze(-1)
        return self.impl.extract_data_from_jags(samples)
