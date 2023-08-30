# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import beanmachine.ppl as bm
import numpy as np
import torch
import torch.distributions as dist
import xarray as xr
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples

from .base_bm_impl import BaseBeanMachineImplementation


class CrowdSourcedAnnotation(BaseBeanMachineImplementation):
    def __init__(
        self,
        n: int,
        k: int,
        num_categories: int,
        expected_correctness: float,
        num_labels_per_item: int,
        concentration: float,
    ) -> None:
        """
        :param attrs: model arguments
        """
        self.n = n  # Total number of items
        self.k = k  # Total number of labelers
        self.num_categories = num_categories  # Number of label classes
        self.expected_correctness = (
            expected_correctness  # Prior belief on correctness of labelers
        )
        self.num_labels_per_item = num_labels_per_item  # Number of labels per item
        self.concentration = concentration  # Strength of prior on expected_correctness
        self.alpha = torch.ones(
            num_categories, num_categories, dtype=torch.get_default_dtype()
        )
        for c in range(self.num_categories):
            for c_prime in range(self.num_categories):
                self.alpha[c][c_prime] = (
                    self.expected_correctness
                    if (c == c_prime)
                    else ((1 - self.expected_correctness) / (self.num_categories - 1))
                )
        self.alpha *= self.concentration
        self.labelers = torch.zeros(n, num_labels_per_item, dtype=torch.int)

    @bm.random_variable
    def confusion_matrix(self, j: int, c: int) -> dist.Distribution:
        """
        Confusion matrix for each labeler (j) and category (c), where each row is a
        Dirichlet distribution.
        """
        return dist.Dirichlet(self.alpha[c])

    @bm.random_variable
    def prev(self) -> dist.Distribution:
        """
        Prevalance for each of the categories in a Dirichlet distribution so it adds up
        to 1.
        """
        return dist.Dirichlet(
            torch.ones(self.num_categories) * (1.0 / self.num_categories)
        )

    @bm.random_variable
    def true_label(self, i: int) -> dist.Distribution:
        """
        True label distribution for each item (i) given the prevalence.
        """
        return dist.Categorical(self.prev())

    @bm.random_variable
    def label(self, i: int, j: int) -> dist.Distribution:
        """
        Observed label distribution for each item (i) and label (j).
        """
        labeler = self.labelers[i, j].item()
        return dist.Categorical(
            self.confusion_matrix(labeler, self.true_label(i).item())
        )

    def data_to_observations(self, data: xr.Dataset) -> Dict:
        """
        Take data from the model generator and convert them to a dictionary that maps
        from random variables to observations, which could be used by Bean Machine.
        :param data: A dataset from the model generator
        :returns: a dictionary that maps random variables to their corresponding
        observations
        """
        # transpose the dataset to ensure that it is the way we expect
        data = data.transpose("item", "item_label")

        labelers_val = torch.tensor(
            data.labelers.values, dtype=torch.get_default_dtype()
        )
        labels_val = torch.tensor(data.labels.values, dtype=torch.get_default_dtype())

        self.labelers = labelers_val
        observations = {}
        for i in range(self.n):
            for j in range(self.num_labels_per_item):
                observations[self.label(i, j)] = labels_val[i, j]

        return observations

    def get_queries(self) -> List:
        confusion_matrix_variables = []
        for j in range(self.k):
            for c in range(self.num_categories):
                confusion_matrix_variables.append(self.confusion_matrix(j, c))
        return [self.prev()] + confusion_matrix_variables

    def extract_data_from_bm(self, samples: MonteCarloSamples) -> xr.Dataset:
        """
        Takes the output of Bean Machine and converts into a format expected
        by PPLBench.
        :param samples: a MonteCarloSamples object returns by Bean Machine
        :returns: a dataset over inferred parameters
        """

        prev_samples = (
            samples.get_variable(self.prev(), include_adapt_steps=True).detach().numpy()
        ).squeeze(0)

        individual_reviewer_samples = []
        for j in range(self.k):
            category_samples = []
            for c in range(self.num_categories):
                category_samples.append(
                    # Shape is (1, num_iterations, num_categories)
                    samples.get_variable(
                        self.confusion_matrix(j, c), include_adapt_steps=True
                    )
                    .detach()
                    .numpy()
                )
            # Collect samples from every category for each reviewer
            # The shape of np.stack(category_samples, axis=1).squeeze(0) is
            # (num_categories, num_iterations, num_categories)
            individual_reviewer_samples.append(
                np.stack(category_samples, axis=1).squeeze(0)
            )

        # Combine all reviewers to create the final confusion matrix
        # The shape of individual_reviewer_samples is (k, num_categories,
        # num_iterations, num_categories)
        confusion_matrix_samples = np.array(individual_reviewer_samples)

        # Swap axes so they're in the correct order for xr.Dataset format
        # The final shape of confusion_matrix_samples is (num_iterations, k,
        # num_categories, num_categories)
        confusion_matrix_samples = np.swapaxes(
            np.swapaxes(confusion_matrix_samples, 0, 2), 1, 2
        )

        return xr.Dataset(
            {
                "prev": (["draw", "num_categories"], prev_samples),
                "confusion_matrix": (
                    ["draw", "labeler", "true_category", "obs_category"],
                    confusion_matrix_samples,
                ),
            },
            coords={
                "draw": np.arange(prev_samples.shape[0]),
                "num_categories": np.arange(prev_samples.shape[-1]),
                "labeler": np.arange(confusion_matrix_samples.shape[1]),
                "true_category": np.arange(confusion_matrix_samples.shape[-2]),
                "obs_category": np.arange(confusion_matrix_samples.shape[-1]),
            },
        )
