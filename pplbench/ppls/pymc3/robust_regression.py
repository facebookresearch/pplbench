# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import pymc3 as pm
import xarray as xr
from pymc3.backends.base import MultiTrace

from .base_pymc3_impl import BasePyMC3Implementation


class RobustRegression(BasePyMC3Implementation):
    def __init__(
        self,
        n: int,
        k: int,
        alpha_scale: float,
        beta_scale: float,
        beta_loc: float,
        sigma_mean: float,
    ) -> None:
        self.n = n
        self.k = k
        self.alpha_scale = alpha_scale
        self.beta_scale = beta_scale
        self.beta_loc = beta_loc
        self.sigma_mean = sigma_mean

    def get_model(self, data: xr.Dataset) -> pm.Model:
        # transpose the dataset to ensure that it is the way we expect
        data = data.transpose("item", "feature")

        with pm.Model() as model:
            X = pm.Data("x_obs", data.X.values)
            Y = pm.Data("y_obs", data.Y.values)

            alpha = pm.Normal("alpha", mu=0, sd=self.alpha_scale)
            beta = pm.Normal("beta", mu=self.beta_loc, sd=self.beta_scale, shape=self.k)
            nu = pm.Gamma("nu", alpha=2, beta=0.1)
            sigma = pm.Exponential("sigma", lam=1 / self.sigma_mean)
            mu = alpha + X.dot(beta)
            pm.StudentT("Y", nu=nu, mu=mu, sigma=sigma, observed=Y, shape=self.n)

        return model

    def extract_data_from_pymc3(self, samples: MultiTrace) -> xr.Dataset:
        return xr.Dataset(
            {
                "alpha": (["draw"], samples["alpha"]),
                "beta": (["draw", "feature"], samples["beta"]),
                "nu": (["draw"], samples["nu"]),
                "sigma": (["draw"], samples["sigma"]),
            },
            coords={
                "draw": np.arange(samples["beta"].shape[0]),
                "feature": np.arange(samples["beta"].shape[1]),
            },
        )
