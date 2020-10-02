# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, List

import numpy as np
import xarray as xr

from .base_jags_impl import BaseJagsImplementation


class LogisticRegression(BaseJagsImplementation):
    def __init__(self, **attrs: Dict) -> None:
        self.attrs = attrs

    def get_vars(self) -> List[str]:
        return ["alpha", "beta"]

    def format_data_to_jags(self, data: xr.Dataset) -> Dict:
        # transpose the dataset to ensure that it is the way we expect
        data = data.transpose("item", "feature")
        # we already have all the values to be bound except for X and Y in self.attrs
        attrs: dict = self.attrs.copy()
        attrs["X"] = data.X.values
        attrs["Y"] = data.Y.values
        return attrs

    def extract_data_from_jags(self, samples: Dict) -> xr.Dataset:
        # dim 2 is the chains dimension so we squeeze it out
        return xr.Dataset(
            {
                # alpha dimensions are [1, samples], we want [samples]
                "alpha": (["draw"], samples["alpha"].squeeze(0)),
                # beta dimensions are [k, samples], we want [samples, k]
                "beta": (["draw", "feature"], samples["beta"].T),
            },
            coords={
                "draw": np.arange(samples["beta"].shape[1]),
                "feature": np.arange(samples["beta"].shape[0]),
            },
        )

    def get_code(self) -> str:
        return """
model {
  # priors
  # note: JAGS normal distribution uses precision rather than standard deviation
  alpha ~ dnorm(0.0, 1/(alpha_scale**2));
  for (j in 1:k) {
    beta[j] ~ dnorm(beta_loc, 1/(beta_scale**2));
  }
  # likelihood
  for (i in 1:n) {
    Y[i] ~ dbern(mu[i])
    logit(mu[i]) <- alpha + inprod(beta, X[i,])
  }
}
"""
