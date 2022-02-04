# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import beanmachine.graph as bmg
import numpy as np
import xarray as xr

from .base_bmgraph_impl import BaseBMGraphImplementation


class CrowdSourcedAnnotation(BaseBMGraphImplementation):
    def __init__(self, **attrs) -> None:
        """
        Builds a BMGraph object that represent the statistical model and
        records queries.
        """
        self.attrs = attrs
        self._graph = bmg.Graph()

    def create_initial_graph(self) -> None:
        self._graph = bmg.Graph()
        expected_correctness = self.attrs["expected_correctness"]
        concentration = self.attrs["concentration"]
        self.num_categories = self.attrs["num_categories"]
        assert self.num_categories == 2

        # confusion matrix prior
        # if we have 2 classes, this looks like
        # [[8.0, 2.0], [2.0, 8.0]]
        self.confusion_matrix_prior = np.ones(
            (self.num_categories, self.num_categories)
        )
        for c in range(self.num_categories):
            for c_prime in range(self.num_categories):
                self.confusion_matrix_prior[c][c_prime] = (
                    expected_correctness
                    if (c == c_prime)
                    else ((1 - expected_correctness) / (self.num_categories - 1))
                )
        self.confusion_matrix_prior *= concentration

        self.num_labels_per_item = self.attrs["num_labels_per_item"]
        self.n = self.attrs["n"]  # number of items
        k = self.attrs["k"]  # total number of labelers

        # prev ~ Dirichlet([0.5, 0.5])
        prev, comp_prev = self.dirichlet([0.5, 0.5])

        log_prev = self._graph.add_operator(bmg.OperatorType.LOG, [prev])
        log_comp_prev = self._graph.add_operator(bmg.OperatorType.LOG1MEXP, [log_prev])
        self.log_prev_simplex = [log_prev, log_comp_prev]

        self.confusion_matrix = np.zeros(
            (k, self.num_categories, self.num_categories), dtype=np.int32
        )
        self.log_confusion_matrix = np.zeros(
            (k, self.num_categories, self.num_categories), dtype=np.int32
        )
        for labeler in range(k):
            for true_category in range(self.num_categories):
                # given a labeler and true category, probability of labeling
                # 0 and 1 ~ Dirichlet(confusion_matrix_prior[true_category])
                sample1, sample2 = self.dirichlet(
                    self.confusion_matrix_prior[true_category]
                )
                self.confusion_matrix[labeler, true_category] = [sample1, sample2]
                log_sample1 = self._graph.add_operator(bmg.OperatorType.LOG, [sample1])
                log_sample2 = self._graph.add_operator(bmg.OperatorType.LOG, [sample2])
                self.log_confusion_matrix[labeler, true_category] = [
                    log_sample1,
                    log_sample2,
                ]

    def dirichlet(self, alphas: List[float]) -> Tuple[int, int]:
        """
        helper function to compute dirichlet of size 2
        equivalent to beta distribution and its complement
        """
        alpha = [
            self._graph.add_constant_pos_real(alphas[i]) for i in range(len(alphas))
        ]

        alpha_dist = self._graph.add_distribution(
            bmg.DistributionType.BETA, bmg.AtomicType.PROBABILITY, alpha
        )
        sample1 = self._graph.add_operator(bmg.OperatorType.SAMPLE, [alpha_dist])
        sample2 = self._graph.add_operator(bmg.OperatorType.COMPLEMENT, [sample1])
        self._graph.query(sample1)
        self._graph.query(sample2)
        return sample1, sample2

    @property
    def graph(self) -> bmg.Graph:
        return self._graph

    def bind_data_to_bmgraph(self, data: xr.Dataset) -> None:
        """Add observations to the model"""
        self.create_initial_graph()
        self.labels = data.labels.values
        self.labelers = data.labelers.values

        for i in range(self.n):
            log_prob = [None for _ in range(self.num_categories)]
            for c in range(self.num_categories):
                inner_sum = [self.log_prev_simplex[c]]
                for j in range(self.num_labels_per_item):
                    label = self.labels[i][j]
                    labeler = self.labelers[i][j]
                    inner_sum.append(self.log_confusion_matrix[labeler, c, label])
                log_prob[c] = self._graph.add_operator(bmg.OperatorType.ADD, inner_sum)

            joint_log_prob_i = self._graph.add_operator(
                bmg.OperatorType.LOGSUMEXP, log_prob
            )
            self._graph.add_factor(bmg.FactorType.EXP_PRODUCT, [joint_log_prob_i])

    def format_samples_from_bmgraph(self, samples: np.ndarray) -> xr.Dataset:
        """Convert the result of inference into a dataset for PPLBench."""
        # samples is of size (num_iterations, 2 + k*2*2)

        num_iterations = samples.shape[0]
        num_categories = self.attrs["num_categories"]

        prev = samples[:, 0:num_categories]  # size (num_iterations, num_categories)
        confusion_matrix = np.zeros(
            (num_iterations, self.attrs["k"], num_categories, num_categories)
        )
        counter = 0
        for labeler in range(self.attrs["k"]):
            for true_category in range(num_categories):
                for observed_category in range(num_categories):
                    confusion_matrix[
                        :, labeler, true_category, observed_category
                    ] = samples[:, num_categories + counter]
                    counter += 1

        return xr.Dataset(
            {
                "prev": (["draw", "true_category"], prev),
                "confusion_matrix": (
                    ["draw", "labeler", "true_category", "obs_category"],
                    confusion_matrix,
                ),
            },
            coords={
                "draw": np.arange(num_iterations),
                "true_category": np.arange(num_categories),
                "obs_category": np.arange(num_categories),
                "labeler": np.arange(self.attrs["k"]),
            },
        )
