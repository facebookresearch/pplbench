# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Dict, Optional, Type, cast

import pymc3 as pm
import xarray as xr

from ..base_ppl_impl import BasePPLImplementation
from ..base_ppl_inference import BasePPLInference
from .base_pymc3_impl import BasePyMC3Implementation


class BasePyMC3Inference(BasePPLInference):
    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        # We always expect a PyMC3 Implementation here
        self.impl_class = cast(Type[BasePyMC3Implementation], impl_class)
        self.impl = self.impl_class(**model_attrs)

    def compile(self, **kwargs):
        pass


class MCMC(BasePyMC3Inference):
    def infer(
        self,
        data: xr.Dataset,
        num_samples: int,
        seed: int,
        tune: Optional[int] = None,
        algorithm: str = "NUTS",
        **infer_args
    ) -> xr.Dataset:
        if tune is None:
            tune = num_samples

        model = self.impl.get_model(data)
        with model:
            # Dynamically loading the step method from pymc3
            step_method = getattr(pm, algorithm)(model.vars, **infer_args)
            samples = pm.sample(
                draws=num_samples,
                tune=tune,
                step=step_method,
                random_seed=seed,
                chains=1,
                return_inferencedata=False,
                progressbar=False,
            )

        return self.impl.extract_data_from_pymc3(samples)


class VI(BasePyMC3Inference):
    def infer(
        self,
        data: xr.Dataset,
        num_samples: int,
        seed: int,
        algorithm: str = "ADVI",
        **infer_args
    ) -> xr.Dataset:

        model = self.impl.get_model(data)
        with model:
            method = getattr(pm, algorithm)(**infer_args)
            approx = method.fit(progressbar=False)
            samples = approx.sample(draws=num_samples)

        return self.impl.extract_data_from_pymc3(samples)
