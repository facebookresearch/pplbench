# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.graph as bmg
import numpy as np
import xarray as xr

from .base_bmgraph_impl import BaseBMGraphImplementation


class NSchools(BaseBMGraphImplementation):
    def __init__(self, **attrs) -> None:
        """
        Builds a BMGraph object that represent the statistical model and
        records queries.
        """
        self.attrs = attrs
        self._graph = bmg.Graph()
        zero = self._graph.add_constant(0.0)
        # sigma_state ~ HalfCauchy(0, scale_state);
        scale_state = self._graph.add_constant_pos_real(attrs["scale_state"])
        sigma_state_prior = self._graph.add_distribution(
            bmg.DistributionType.HALF_CAUCHY, bmg.AtomicType.POS_REAL, [scale_state]
        )
        self.sigma_state = self._graph.add_operator(
            bmg.OperatorType.SAMPLE, [sigma_state_prior]
        )
        self.query_sigma_state = self._graph.query(self.sigma_state)
        # sigma_district ~ HalfCauchy(0, scale_district);
        scale_district = self._graph.add_constant_pos_real(attrs["scale_district"])
        sigma_district_prior = self._graph.add_distribution(
            bmg.DistributionType.HALF_CAUCHY, bmg.AtomicType.POS_REAL, [scale_district]
        )
        self.sigma_district = self._graph.add_operator(
            bmg.OperatorType.SAMPLE, [sigma_district_prior]
        )
        self.query_sigma_district = self._graph.query(self.sigma_district)
        # sigma_type ~ HalfCauchy(0, scale_type);
        scale_type = self._graph.add_constant_pos_real(attrs["scale_type"])
        sigma_type_prior = self._graph.add_distribution(
            bmg.DistributionType.HALF_CAUCHY, bmg.AtomicType.POS_REAL, [scale_type]
        )
        self.sigma_type = self._graph.add_operator(
            bmg.OperatorType.SAMPLE, [sigma_type_prior]
        )
        self.query_sigma_type = self._graph.query(self.sigma_type)
        # beta_baseline ~ student_t(dof_baseline, 0, scale_baseline)
        dof_baseline = self._graph.add_constant_pos_real(attrs["dof_baseline"])
        scale_baseline = self._graph.add_constant_pos_real(attrs["scale_baseline"])
        beta_baseline_prior = self._graph.add_distribution(
            bmg.DistributionType.STUDENT_T,
            bmg.AtomicType.REAL,
            [dof_baseline, zero, scale_baseline],
        )
        beta_baseline = self._graph.add_operator(
            bmg.OperatorType.SAMPLE, [beta_baseline_prior]
        )
        self.query_beta_baseline = self._graph.query(beta_baseline)
        # beta_state[ ] ~ Normal(0, sigma_state)
        # best_district[ , ] ~ Normal(0, sigma_district)
        beta_state_prior = self._graph.add_distribution(
            bmg.DistributionType.NORMAL, bmg.AtomicType.REAL, [zero, self.sigma_state]
        )
        beta_district_prior = self._graph.add_distribution(
            bmg.DistributionType.NORMAL,
            bmg.AtomicType.REAL,
            [zero, self.sigma_district],
        )
        beta_state = np.ndarray(attrs["num_states"]).tolist()
        beta_district = np.ndarray(
            (attrs["num_states"], attrs["num_districts_per_state"])
        ).tolist()
        for state in range(attrs["num_states"]):
            beta_state[state] = self._graph.add_operator(
                bmg.OperatorType.SAMPLE, [beta_state_prior]
            )
            for district in range(attrs["num_districts_per_state"]):
                beta_district[state][district] = self._graph.add_operator(
                    bmg.OperatorType.SAMPLE, [beta_district_prior]
                )
        self.query_beta_state = [
            self._graph.query(beta_state[state]) for state in range(attrs["num_states"])
        ][0]
        self.query_beta_district = [
            self._graph.query(beta_district[state][district])
            for state in range(attrs["num_states"])
            for district in range(attrs["num_districts_per_state"])
        ][0]
        # beta_type[ ] ~ Normal(0, sigma_type)
        beta_type_prior = self._graph.add_distribution(
            bmg.DistributionType.NORMAL, bmg.AtomicType.REAL, [zero, self.sigma_type]
        )
        beta_type = [
            self._graph.add_operator(bmg.OperatorType.SAMPLE, [beta_type_prior])
            for _ in range(attrs["num_types"])
        ]
        self.query_beta_type = [self._graph.query(t) for t in beta_type][0]
        # yhat[i] = beta_baseline + beta_state[state_idx[i]]
        #   + beta_district[state_idx[i], district_idx[i]] + beta_type[type_idx[i]]
        # yhat[i] ~ Normal(yhat[i], sigma[i])
        # sigma[i] ~ Flat
        sigma_prior = self._graph.add_distribution(
            bmg.DistributionType.FLAT, bmg.AtomicType.POS_REAL, []
        )
        self.sigma = np.ndarray(attrs["n"]).tolist()
        self.y = np.ndarray(attrs["n"]).tolist()
        for i in range(attrs["n"]):
            self.sigma[i] = self._graph.add_operator(
                bmg.OperatorType.SAMPLE, [sigma_prior]
            )
            yhat = self._graph.add_operator(
                bmg.OperatorType.ADD,
                [
                    beta_baseline,
                    beta_state[attrs["state_idx"][i]],
                    beta_district[attrs["state_idx"][i]][attrs["district_idx"][i]],
                    beta_type[attrs["type_idx"][i]],
                ],
            )
            y_prior = self._graph.add_distribution(
                bmg.DistributionType.NORMAL, bmg.AtomicType.REAL, [yhat, self.sigma[i]]
            )
            self.y[i] = self._graph.add_operator(bmg.OperatorType.SAMPLE, [y_prior])

    @property
    def graph(self) -> bmg.Graph:
        return self._graph

    def bind_data_to_bmgraph(self, data: xr.Dataset) -> None:
        """Add observations to the model"""
        self._graph.remove_observations()  # remove observations from previous trial
        for i, (y_val, sigma_val) in enumerate(zip(data.Y.values, data.sigma.values)):
            self._graph.observe(self.y[i], y_val)
            self._graph.observe(self.sigma[i], sigma_val)

    def format_samples_from_bmgraph(self, samples: np.ndarray) -> xr.Dataset:
        """Convert the result of inference into a dataset for PPLBench."""
        beta_state = samples[
            :, self.query_beta_state : self.query_beta_state + self.attrs["num_states"]
        ]
        beta_district = samples[
            :,
            self.query_beta_district : self.query_beta_district
            + self.attrs["num_states"] * self.attrs["num_districts_per_state"],
        ].reshape(-1, self.attrs["num_states"], self.attrs["num_districts_per_state"])
        beta_type = samples[
            :, self.query_beta_type : self.query_beta_type + self.attrs["num_types"]
        ]
        return xr.Dataset(
            {
                "sigma_state": (["draw"], samples[:, self.query_sigma_state]),
                "sigma_district": (["draw"], samples[:, self.query_sigma_district]),
                "sigma_type": (["draw"], samples[:, self.query_sigma_type]),
                "beta_baseline": (["draw"], samples[:, self.query_beta_baseline]),
                "beta_state": (["draw", "state"], beta_state),
                "beta_district": (["draw", "state", "district"], beta_district),
                "beta_type": (["draw", "type"], beta_type),
            },
            coords={
                "draw": np.arange(samples.shape[0]),
                "state": np.arange(self.attrs["num_states"]),
                "district": np.arange(self.attrs["num_districts_per_state"]),
                "type": np.arange(self.attrs["num_types"]),
            },
        )
