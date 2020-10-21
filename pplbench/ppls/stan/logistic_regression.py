# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class LogisticRegression(BaseStanImplementation):
    def __init__(self, **attrs: Dict) -> None:
        """
        :param attrs: model arguments
        """
        self.attrs = attrs

    def format_data_to_stan(self, data: xr.Dataset) -> Dict:
        """
        Take data from the model generator and the previously passed model
        arguments to construct a data dictionary to pass to Stan.
        :param data: A dataset from the model generator
        :returns: a dictionary that can be passed to stan
        """
        # transpose the dataset to ensure that it is the way we expect
        data = data.transpose("item", "feature")
        # we already have all the values to be bound except for X and Y in self.attrs
        attrs: dict = self.attrs.copy()
        attrs["X"] = data.X.values
        attrs["Y"] = data.Y.values
        return attrs

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        """
        Takes the output of Stan and converts into a format expected
        by PPLBench.
        :param samples: samples dictionary from Stan
        :returns: a dataset over inferred parameters
        """
        # dim 1 is the chains dimension so we squeeze it out
        return xr.Dataset(
            {
                "alpha": (["draw"], samples["alpha"].squeeze(1)),
                "beta": (["draw", "feature"], samples["beta"].squeeze(1)),
            },
            coords={
                "draw": np.arange(samples["beta"].shape[0]),
                "feature": np.arange(samples["beta"].shape[-1]),
            },
        )

    def get_pars(self) -> List[str]:
        """
        :returns: The list of parameters that are needed from inference.
        """
        return ["alpha", "beta"]

    def get_code(self) -> str:
        """
        :returns: Returns a string that represents the Stan model.
        """
        return """
data {
  // number of observations
  int n;
  // response
  int<lower = 0, upper=1> Y[n];
  // number of columns in the design matrix X
  int k;
  // design matrix X
  // should not include an intercept
  matrix [n, k] X;
  // priors on alpha and beta
  real alpha_scale;
  real beta_scale;
  real beta_loc;
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[k] beta;
}
transformed parameters {
  vector[n] mu;

  mu = alpha + X * beta;
}
model {
  // priors
  alpha ~ normal(0.0, alpha_scale);
  beta ~ normal(beta_loc, beta_scale);
  // likelihood
  Y ~ bernoulli_logit(mu);
}
"""
