# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import beanmachine.graph as bmg
import numpy as np
import xarray as xr

from .base_bmgraph_impl import BaseBMGraphImplementation


class LogisticRegression(BaseBMGraphImplementation):
    def __init__(self, **attrs) -> None:
        """
        Builds a BMGraph object that represent the statistical model and
        records queries.
        """
        self.attrs = attrs
        self._graph = bmg.Graph()
        self.x_samples: List[List[int]] = []
        self.y_samples: List[int] = []
        self._initialize_graph()

    def _initialize_graph(self):
        n = self.attrs["n"]
        k = self.attrs["k"]
        alpha_scale = self.attrs["alpha_scale"]
        beta_scale = self.attrs["beta_scale"]
        beta_loc = self.attrs["beta_loc"]

        zero = self._graph.add_constant(0.0)
        pos_one = self._graph.add_constant_pos_real(1.0)
        alpha_scale_constant = self._graph.add_constant_pos_real(alpha_scale)
        beta_scale_constant = self._graph.add_constant_pos_real(beta_scale)
        beta_loc_constant = self._graph.add_constant(beta_loc)

        alpha_dist = self._graph.add_distribution(
            bmg.DistributionType.NORMAL,
            bmg.AtomicType.REAL,
            [zero, alpha_scale_constant],
        )
        alpha_sample = self._graph.add_operator(bmg.OperatorType.SAMPLE, [alpha_dist])
        self._graph.query(alpha_sample)

        beta_sample_list = []
        for _ in range(k):
            beta_dist = self._graph.add_distribution(
                bmg.DistributionType.NORMAL,
                bmg.AtomicType.REAL,
                [beta_loc_constant, beta_scale_constant],
            )
            beta_sample = self._graph.add_operator(bmg.OperatorType.SAMPLE, [beta_dist])
            beta_sample_list.append(beta_sample)
            self._graph.query(beta_sample)

        for _ in range(n):
            x_beta_list = []
            x_samples_i = []
            for j in range(k):
                x_dist = self._graph.add_distribution(
                    bmg.DistributionType.NORMAL,
                    bmg.AtomicType.REAL,
                    [zero, pos_one],
                )
                x_sample = self._graph.add_operator(bmg.OperatorType.SAMPLE, [x_dist])
                x_samples_i.append(x_sample)
                x_beta = self._graph.add_operator(
                    bmg.OperatorType.MULTIPLY, [x_sample, beta_sample_list[j]]
                )
                x_beta_list.append(x_beta)
            self.x_samples.append(x_samples_i)
            x_beta_sum = self._graph.add_operator(bmg.OperatorType.ADD, x_beta_list)
            mu = self._graph.add_operator(
                bmg.OperatorType.ADD, [x_beta_sum, alpha_sample]
            )
            y_dist = self._graph.add_distribution(
                bmg.DistributionType.BERNOULLI_LOGIT,
                bmg.AtomicType.BOOLEAN,
                [mu],
            )
            y_sample = self._graph.add_operator(bmg.OperatorType.SAMPLE, [y_dist])
            self.y_samples.append(y_sample)

    def bind_data_to_bmgraph(self, data: xr.Dataset) -> None:
        self._graph.remove_observations()

        x_data = data.X.values
        y_data = data.Y.values
        n = self.attrs["n"]
        k = self.attrs["k"]

        for i in range(n):
            for j in range(k):
                self._graph.observe(self.x_samples[i][j], x_data[i][j])
            self._graph.observe(self.y_samples[i], y_data[i])

    @property
    def graph(self) -> bmg.Graph:
        return self._graph

    def format_samples_from_bmgraph(self, samples: np.ndarray) -> xr.Dataset:
        """Convert the result of inference into a dataset for PPLBench."""
        num_iterations = samples.shape[0]

        alpha_samples = samples[:, 0]
        beta_samples = samples[:, 1:]
        feature_size = len(beta_samples[0])

        return xr.Dataset(
            {
                "alpha": (["draw"], alpha_samples),
                "beta": (["draw", "feature"], beta_samples),
            },
            coords={
                "draw": np.arange(num_iterations),
                "feature": np.arange(feature_size),
            },
        )
