# Copyright (c) Facebook, Inc. and its affiliates
import time
from typing import Dict, List, Tuple

import beanmachine.ppl as bm
import numpy as np
import torch
import torch.distributions as dist
import torch.tensor as tensor
from torch import Tensor

from ..pplbench_ppl import PPLBenchPPL


"""
For model definition, see models/robust_regression_model.py
"""


class RobustRegressionModel(object):
    def __init__(
        self,
        N: int,
        K: int,
        scale_alpha: float,
        scale_beta: List[float],
        loc_beta: float,
        rate_sigma: float,
        num_samples: int,
        inference_type: str,
        X: Tensor,
        Y: Tensor,
    ) -> None:
        self.N = N
        self.K = K
        self.scale_alpha = scale_alpha
        self.scale_beta = scale_beta
        self.loc_beta = loc_beta
        self.rate_sigma = rate_sigma
        self.num_samples = num_samples
        self.inference_type = inference_type
        self.X = torch.cat((torch.ones((1, N)), X))
        self.Y = Y

    @bm.random_variable
    def nu(self):
        return dist.Gamma(2.0, 0.1)

    @bm.random_variable
    def sigma(self):
        return dist.Exponential(self.rate_sigma)

    @bm.random_variable
    def beta(self):
        return dist.Normal(
            tensor([0.0] + [self.loc_beta] * self.K),
            tensor([self.scale_alpha] + self.scale_beta),
        )

    @bm.random_variable
    def y(self):
        # Compute X * Beta
        beta_ = self.beta().reshape((1, self.beta().shape[0]))
        mu = beta_.mm(self.X)
        return dist.StudentT(self.nu(), mu, self.sigma())

    def infer(self):
        if self.inference_type == "mcmc":
            mh = bm.SingleSiteNewtonianMonteCarlo()
            start_time = time.time()
            samples = mh.infer(
                [self.beta(), self.nu(), self.sigma()],
                {self.y(): self.Y},
                self.num_samples,
                1,
            ).get_chain()
        elif self.inference_type == "vi":
            print("ImplementationError; exiting...")
            exit(1)
        elapsed_time_sample_beanmachine = time.time() - start_time
        return (samples, elapsed_time_sample_beanmachine)


class RobustRegression(PPLBenchPPL):
    def obtain_posterior(
        self, data_train: Tuple[np.ndarray, np.ndarray], args_dict: Dict, model=None
    ) -> Tuple[List, Dict]:
        """
        Beanmachine impmementation of robust regression model.

        :param data_train: tuple of np.ndarray (x_train, y_train)
        :param args_dict: a dict of model arguments
        :returns: samples_beanmachine(dict): posterior samples of all parameters
        :returns: timing_info(dict): compile_time, inference_time
        """
        # shape of x_train: (num_features, num_samples)
        x_train, y_train = data_train
        x_train = tensor(x_train, dtype=torch.float32)
        y_train = tensor(y_train, dtype=torch.float32)
        N = int(x_train.shape[1])
        K = int(x_train.shape[0])

        alpha_scale, beta_scale, beta_loc, sigma_mean = args_dict["model_args"]
        beta_scale = [beta_scale] * K
        num_samples = args_dict["num_samples_beanmachine-vectorized"]
        inference_type = args_dict["inference_type"]

        start_time = time.time()
        robust_regression_model = RobustRegressionModel(
            N,
            K,
            alpha_scale,
            beta_scale,
            beta_loc,
            1.0 / sigma_mean,
            num_samples,
            inference_type,
            x_train,
            y_train,
        )
        elapsed_time_compile_beanmachine = time.time() - start_time
        samples, elapsed_time_sample_beanmachine = robust_regression_model.infer()

        # repackage samples into format required by PPLBench
        # List of dict, where each dict has key = param (string), value = value of param
        param_keys = ["beta", "nu", "sigma"]
        samples_formatted = []
        for i in range(num_samples):
            sample_dict = {}
            for j, parameter in enumerate(samples.get_rv_names()):
                if j == 0:
                    sample_dict[param_keys[j]] = (
                        samples.get_variable(parameter)[i][1:]
                        .detach()
                        .numpy()
                        .reshape(1, K)
                    )
                    sample_dict["alpha"] = (
                        samples.get_variable(parameter)[i][0].detach().numpy()
                    )
                else:
                    sample_dict[param_keys[j]] = samples[parameter][i].item()
            samples_formatted.append(sample_dict)

        timing_info = {
            "compile_time": elapsed_time_compile_beanmachine,
            "inference_time": elapsed_time_sample_beanmachine,
        }
        return (samples_formatted, timing_info)
