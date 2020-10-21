# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import xarray as xr


class BaseModel(ABC):
    @staticmethod
    @abstractmethod
    def generate_data(seed: int, n: int, **model_args) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Generate data from the model by forward sampling from the prior distribution.
        :param seed: A random number generator seed.
        :param n: The number of items in the dataset.
        :returns: A tuple of training and test datasets.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def evaluate_posterior_predictive(
        post_samples: xr.Dataset, test: xr.Dataset
    ) -> np.ndarray:
        """
        Evaluate and return the predictive log likelihood (PLL).
        :param post_samples: The posterior samples from a single MCMC chain.
            The `post_samples` object must have a `sample` dimension.
        :param test: The test data previously created by `generate_data`.
        :returns: The PLL of each sample on the test data.
        """
        raise NotImplementedError

    @staticmethod
    def additional_metrics(
        output_dir: str, post_samples: xr.Dataset, train: xr.Dataset, test: xr.Dataset
    ) -> None:
        """
        Compute model-specific metrics other than the standard PLL
        and write them out. This method is optional.
        :param output_dir: directory to store the results
        :post_samples: all posterior samples; with coords (ppl, chain, draw, ...)
        :train: training dataset
        :test: test dataset
        """
        pass
