# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
from typing import Tuple

import numpy as np
import xarray as xr
from scipy.stats import norm

from .base_model import BaseModel
from .utils import split_train_test


LOGGER = logging.getLogger(__name__)


class NSchools(BaseModel):
    """
    N Schools

    This is a generalization of a classical 8 schools model to n schools.
    The model posits that the effect of a school on a student's performance
    can be explained by the a baseline effect of all schools plus an additive
    effect of the state, the school district and the school type.

    Hyper Parameters:

        n - total number of schools
        num_states - number of states
        num_districts_per_state - number of school districts in each state
        num_types - number of school types
        scale_state - state effect scale
        scale_district - district effect scale
        scale_type - school type effect scale

    Model:


        beta_baseline = StudentT(dof_baseline, 0.0, scale_baseline)

        for s in 0 .. num_states - 1
            beta_state[s] ~ Normal(0, scale_state)

            for d in 0 .. num_districts_per_state - 1
                beta_district[s, d] ~ Normal(0, scale_district)

        for t in 0 .. num_types - 1
            beta_type[t] ~ Normal(0, scale_type)

        for i in 0 ... n - 1
            Assume we are given state[i], district[i], type[i]

            Y_hat[i] = beta_baseline + beta_state[state[i]]
                        + beta_district[state[i], district[i]]
                        + beta_type[type[i]]

            sigma[i] ~ Uniform(0.5, 1.5)

            Y[i] ~ Normal(Y_hat[i], sigma[i])

    The dataset consists of the following

        state[school]     - 0 .. num_schools - 1
        district[school]  - 0 .. num_districts_per_state - 1
        type[school]      - 0 .. num_types - 1
        Y[school]         - float
        sigma[school]     - float

    and it includes the attributes

        num_states
        num_districts_per_state
        num_types
        dof_baseline
        scale_baseline
        scale_state
        scale_district
        scale_type

    The posterior samples include the following,

        beta_baseline[draw]                  - float
        beta_state[draw, state]              - float
        beta_district[draw, state, district] - float
        beta_type[draw, type]                - float
    """

    @staticmethod
    def generate_data(  # type: ignore
        seed: int,
        n: int = 2000,
        num_states: int = 8,
        num_districts_per_state: int = 5,
        num_types: int = 5,
        dof_baseline: float = 3.0,
        scale_baseline: float = 10.0,
        scale_state: float = 1.0,
        scale_district: float = 1.0,
        scale_type: float = 1.0,
        train_frac: float = 0.5,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        See the class documentation for an explanation of the other parameters.
        :param train_frac: fraction of data to be used for training (default 0.5)
        """
        rng = np.random.default_rng(seed)
        beta_baseline = rng.standard_t(dof_baseline)
        beta_state = rng.normal(loc=0, scale=scale_state, size=num_states)
        beta_district = rng.normal(
            loc=0, scale=scale_district, size=(num_states, num_districts_per_state)
        )
        beta_type = rng.normal(loc=0, scale=scale_type, size=num_types)

        # we will randomly assign the schools to states, district, and types
        state = rng.integers(low=0, high=num_states, size=n)
        district = rng.integers(low=0, high=num_districts_per_state, size=n)
        type_ = rng.integers(low=0, high=num_types, size=n)
        sigma = rng.uniform(0.5, 1.5, size=n)

        y_hat = (
            beta_baseline
            + beta_state[state]
            + beta_district[state, district]
            + beta_type[type_]
        )
        y = rng.normal(loc=y_hat, scale=sigma)

        data = xr.Dataset(
            {
                "Y": (["school"], y),
                "sigma": (["school"], sigma),
                "state": (["school"], state),
                "district": (["school"], district),
                "type": (["school"], type_),
            },
            coords={"school": np.arange(n)},
            attrs={
                "n": n,
                "num_states": num_states,
                "num_districts_per_state": num_districts_per_state,
                "num_types": num_types,
                "dof_baseline": dof_baseline,
                "scale_baseline": scale_baseline,
                "scale_state": scale_state,
                "scale_district": scale_district,
                "scale_type": scale_type,
            },
        )
        return split_train_test(data, "school", train_frac)

    @staticmethod
    def evaluate_posterior_predictive(
        samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        """
        Computes the predictive likelihood of all the test items w.r.t. each sample.
        See the class documentation for the `samples` and `test` parameters.
        :returns: a numpy array of the same size as the sample dimension.
        """
        # transpose the datasets to be in a convenient format
        samples = samples.transpose("draw", "state", "district", "type")
        y_hat = (
            samples.beta_baseline.values[:, np.newaxis]
            + samples.beta_state.values[:, test.state.values]
            + samples.beta_district.values[:, test.state.values, test.district.values]
            + samples.beta_type.values[:, test.type.values]
        )  # size = (num_samples, n_test)
        loglike = norm.logpdf(
            test.Y.values[np.newaxis, :],
            loc=y_hat,
            scale=test.sigma.values[np.newaxis, :],
        )  # size = (num_samples, n_test)
        return loglike.sum(axis=1)  # size = (num_samples,)
