# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
from typing import Dict, Type, cast

import numpy as np
import xarray as xr
from pystan import StanModel

from ..base_ppl_impl import BasePPLImplementation
from ..base_ppl_inference import BasePPLInference
from .base_stan_impl import BaseStanImplementation


class BaseStanInference(BasePPLInference):
    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        # We always expect a Stan Implementation here
        self.impl_class = cast(Type[BaseStanImplementation], impl_class)
        self.impl = self.impl_class(**model_attrs)

    def compile(self, seed: int, **compile_args):
        self.stan_model = StanModel(
            model_code=self.impl.get_code(),
            model_name=self.impl_class.__name__,
            **compile_args
        )


class MCMC(BaseStanInference):
    def infer(
        self,
        data: xr.Dataset,
        iterations: int,
        num_warmup: int,
        seed: int,
        algorithm: str = "NUTS",
        **infer_args
    ) -> xr.Dataset:
        """
        See https://pystan.readthedocs.io/en/latest/api.html#pystan.StanModel.vb
        """

        self.fit = self.stan_model.sampling(
            data=self.impl.format_data_to_stan(data),
            pars=self.impl.get_pars(),
            iter=iterations,
            warmup=num_warmup,
            chains=1,
            check_hmc_diagnostics=False,
            seed=seed,
            algorithm=algorithm,
            **infer_args
        )
        results = self.fit.extract(
            permuted=False, inc_warmup=True, pars=self.impl.get_pars()
        )
        return self.impl.extract_data_from_stan(results)


class VI(BaseStanInference):
    is_adaptive = False

    def infer(  # type: ignore
        self,
        data: xr.Dataset,
        iterations: int,
        num_warmup: int,
        seed: int,
        algorithm: str = "meanfield",
    ) -> xr.Dataset:
        """
        See https://pystan.readthedocs.io/en/latest/api.html#pystan.StanModel.vb
        """
        self.results = self.stan_model.vb(
            data=self.impl.format_data_to_stan(data),
            pars=self.impl.get_pars(),
            output_samples=iterations,
            seed=seed,
            algorithm=algorithm,
        )
        params = VI.pystan_vb_extract(self.results)
        return self.impl.extract_data_from_stan(params)

    @staticmethod
    def pystan_vb_extract(results: OrderedDict):
        """
        From: https://gist.github.com/lwiklendt/9c7099288f85b59edc903a5aed2d2d64
        Converts vb results from pystan into a format similar to fit.extract()
        where fit is returned from sampling.
        This version is modified from the above reference to add a chain dimension
        for consistency with fit.extract(..)
        :param results: returned from vb
        """
        param_specs = results["sampler_param_names"]
        samples = results["sampler_params"]
        n = len(samples[0])

        # first pass, calculate the shape
        param_shapes: dict = OrderedDict()
        for param_spec in param_specs:
            splt = param_spec.split("[")
            name = splt[0]
            if len(splt) > 1:
                idxs = [
                    int(i) for i in splt[1][:-1].split(",")
                ]  # no +1 for shape calculation because pystan already returns 1-based indexes for vb!
            else:
                idxs = []
            param_shapes[name] = np.maximum(idxs, param_shapes.get(name, idxs))

        # create arrays
        params = OrderedDict(
            [
                (name, np.nan * np.empty((n,) + tuple(shape)))
                for name, shape in param_shapes.items()
            ]
        )

        # second pass, set arrays
        for param_spec, param_samples in zip(param_specs, samples):
            splt = param_spec.split("[")
            name = splt[0]
            if len(splt) > 1:
                idxs = [
                    int(i) - 1 for i in splt[1][:-1].split(",")
                ]  # -1 because pystan returns 1-based indexes for vb!
            else:
                idxs = []
            params[name][(...,) + tuple(idxs)] = param_samples

        # finally, add the chain dimension
        for name, value in params.items():
            params[name] = np.expand_dims(value, axis=1)

        return params
