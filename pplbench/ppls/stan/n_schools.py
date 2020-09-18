# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, List

import numpy as np
import xarray as xr

from .base_stan_impl import BaseStanImplementation


class NSchools(BaseStanImplementation):
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
        # Note: self.attrs already contains the following
        # n, num_states, num_districts_per_state, num_types, dof_baseline
        # scale_baseline, scale_state, scale_district, scale_type
        # Also we need to update the value of n in the model to n_train
        attrs: dict = self.attrs.copy()
        attrs["n"] = len(data.coords["school"])
        attrs["Y"] = data.Y.values
        attrs["sigma"] = data.sigma.values
        # indices in Stan are 1-based
        attrs["state_idx"] = data.state.values + 1
        attrs["district_idx"] = data.district.values + 1
        attrs["type_idx"] = data.type.values + 1
        return attrs

    def extract_data_from_stan(self, samples: Dict) -> xr.Dataset:
        """
        Takes the output of Stan and converts into a format expected
        by PPLBench.
        :param samples: samples dictionary from Stan
        :returns: a dataset over inferred parameters
        """
        return xr.Dataset(
            {
                "beta_baseline": (["draw"], samples["beta_baseline"].squeeze(1)),
                "beta_state": (["draw", "state"], samples["beta_state"].squeeze(1)),
                "beta_district": (
                    ["draw", "state", "district"],
                    samples["beta_district"].squeeze(1),
                ),
                "beta_type": (["draw", "type"], samples["beta_type"].squeeze(1)),
            },
            coords={
                "draw": np.arange(samples["beta_baseline"].shape[0]),
                "state": np.arange(self.attrs["num_states"]),
                "district": np.arange(self.attrs["num_districts_per_state"]),
                "type": np.arange(self.attrs["num_types"]),
            },
        )

    def get_pars(self) -> List[str]:
        """
        :returns: The list of parameters that are needed from inference.
        """
        return ["beta_baseline", "beta_state", "beta_district", "beta_type"]

    def get_code(self) -> str:
        """
        :returns: Returns a string that represents the Stan model.
        """
        return """
data {
  int<lower=1> n; // number of schools
  int<lower=1> num_states;
  int<lower=1> num_districts_per_state;
  int<lower=1> num_types;
  real Y[n];
  real<lower=0> sigma[n];
  int<lower=1, upper=num_states> state_idx[n];
  int<lower=1, upper=num_districts_per_state> district_idx[n];
  int<lower=1, upper=num_types> type_idx[n];
  real<lower=0> dof_baseline;
  real<lower=0> scale_baseline;
  real<lower=0> scale_state;
  real<lower=0> scale_district;
  real<lower=0> scale_type;
}

parameters {
  real beta_baseline;
  vector[num_states] beta_state;
  matrix[num_states, num_districts_per_state] beta_district;
  vector[num_types] beta_type;
}

transformed parameters {
    vector[n] Yhat;
    Yhat = beta_baseline
            + beta_state[state_idx]
            + diagonal(beta_district[state_idx, district_idx])
            + beta_type[type_idx];
}

model {
  beta_baseline ~ student_t(dof_baseline, 0, scale_baseline);
  beta_state ~ normal(0, scale_state);
  to_vector(beta_district) ~ normal(0, scale_district);
  beta_type ~ normal(0, scale_type);
  Y ~ normal(Yhat, sigma);
}
"""
