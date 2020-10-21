# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, List

import numpy as np
import xarray as xr

from .base_jags_impl import BaseJagsImplementation


class NSchools(BaseJagsImplementation):
    def __init__(self, **attrs: Dict) -> None:
        self.attrs = attrs

    def get_vars(self) -> List[str]:
        return [
            "sigma_state",
            "sigma_district",
            "sigma_type",
            "beta_baseline",
            "beta_state",
            "beta_district",
            "beta_type",
        ]

    def format_data_to_jags(self, data: xr.Dataset) -> Dict:
        # Note: self.attrs already contains the following
        # n, num_states, num_districts_per_state, num_types, dof_baseline
        # scale_baseline, scale_state, scale_district, scale_type
        attrs: dict = self.attrs.copy()
        attrs["Y"] = data.Y.values
        attrs["sigma"] = data.sigma.values
        # indices in JAGS are 1-based
        # note: we are not doing in place increments because we shouldn't
        # modify the arrays that are passed in
        attrs["state_idx"] = attrs["state_idx"] + 1
        attrs["district_idx"] = attrs["district_idx"] + 1
        attrs["type_idx"] = attrs["type_idx"] + 1
        return attrs

    def extract_data_from_jags(self, samples: Dict) -> xr.Dataset:
        return xr.Dataset(
            {
                # JAGS adds an extra dimension for scalars
                "sigma_state": (["draw"], samples["sigma_state"].squeeze(0)),
                "sigma_district": (["draw"], samples["sigma_district"].squeeze(0)),
                "sigma_type": (["draw"], samples["sigma_type"].squeeze(0)),
                "beta_baseline": (["draw"], samples["beta_baseline"].squeeze(0)),
                # draw is the last dimension
                "beta_state": (["state", "draw"], samples["beta_state"]),
                "beta_district": (
                    ["state", "district", "draw"],
                    samples["beta_district"],
                ),
                "beta_type": (["type", "draw"], samples["beta_type"]),
            },
            coords={
                "draw": np.arange(samples["beta_baseline"].shape[-1]),
                "state": np.arange(self.attrs["num_states"]),
                "district": np.arange(self.attrs["num_districts_per_state"]),
                "type": np.arange(self.attrs["num_types"]),
            },
        )

    def get_code(self) -> str:
        return """
model {
  # priors
  # note: a Student-T distribution with k=1 is a Cauchy distribution
  # and truncating by `I(0,)` makes it a half-cauchy
  sigma_state ~ dt(0, 1/(scale_state**2), 1) I(0,)
  sigma_district ~ dt(0, 1/(scale_district**2), 1) I(0,)
  sigma_type ~ dt(0, 1/(scale_type**2), 1) I(0,)
  beta_baseline ~ dt(0, 1/(scale_baseline**2), dof_baseline)
  for (i in 1:num_states) {
    beta_state[i] ~ dnorm(0, 1/(sigma_state**2))
    for (j in 1:num_districts_per_state) {
      beta_district[i, j] ~ dnorm(0, 1/(sigma_district**2))
    }
  }
  for (i in 1:num_types) {
    beta_type[i] ~ dnorm(0, 1/(sigma_type**2))
  }
  # likelihood
  for (i in 1:n) {
    Yhat[i] <- beta_baseline + beta_state[state_idx[i]]
        + beta_district[state_idx[i], district_idx[i]] + beta_type[type_idx[i]]
    # note: JAGS normal distribution uses precision rather than standard dev.
    Y[i] ~ dnorm(Yhat[i], 1/(sigma[i]**2))
  }
}
"""
