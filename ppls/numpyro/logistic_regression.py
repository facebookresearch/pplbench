# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import time

import jax.numpy as np  # pyre-ignore
import numpy as onp
from jax import random
from numpyro.distributions import Bernoulli, Normal  # pyre-ignore
from numpyro.infer import MCMC, NUTS
from numpyro.primitives import sample

from ..pplbench_ppl import PPLBenchPPL


def logistic_regression(x, y, model_args):
    ndims = np.shape(x)[-1]
    scale_alpha, scale_beta, loc_beta, _ = model_args

    beta = sample("beta", Normal(loc_beta, np.ones(ndims) * scale_beta))
    alpha = sample("alpha", Normal(0.0, scale_alpha))
    return sample("y", Bernoulli(logits=x @ beta + alpha), obs=y)


class LogisticRegression(PPLBenchPPL):
    def obtain_posterior(self, data_train, args_dict, model=None):
        """
        Numpyro implementation of logistic regression model.

        Inputs:
        - data_train(tuple of np.ndarray): x_train, y_train
        - args_dict: a dict of model arguments
        Returns:
        - samples_numpyro(dict): posterior samples of all parameters
        - timing_info(dict): compile_time, inference_time
        """
        model_args = args_dict["model_args"]

        # x_train is (num_features, num_observations)
        x_train, y_train = data_train
        # x_train is (num_observations, num_features)
        x_train = x_train.T
        start_time = time.time()

        num_warmup = 500
        num_samples = args_dict["num_samples_numpyro"] - num_warmup

        assert num_samples > 0
        # Run inference to generate samples from the posterior
        kernel = NUTS(model=logistic_regression)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.warmup(
            random.PRNGKey(1),
            x=x_train,
            y=y_train,
            model_args=model_args,
            collect_warmup=True,
        )
        warm_up_samples = mcmc.get_samples()
        mcmc.run(random.PRNGKey(1), x=x_train, y=y_train, model_args=model_args)
        samples_numpyro = mcmc.get_samples()
        elapsed_time_sample_pyro = time.time() - start_time

        # repackage samples into shape required by PPLBench
        samples = []

        for i in range(num_warmup):
            sample_dict = {}
            sample_dict["alpha"] = onp.asarray(warm_up_samples["alpha"][i])
            sample_dict["beta"] = onp.asarray(warm_up_samples["beta"][i])
            samples.append(sample_dict)
        for i in range(num_samples):
            sample_dict = {}
            sample_dict["alpha"] = onp.asarray(samples_numpyro["alpha"][i])
            sample_dict["beta"] = onp.asarray(samples_numpyro["beta"][i])
            samples.append(sample_dict)
        timing_info = {
            "compile_time": 0,
            "inference_time": elapsed_time_sample_pyro,
        }  # no compiliation for pyro
        return (samples, timing_info)
