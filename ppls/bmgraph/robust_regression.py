# Copyright (c) Facebook, Inc. and its affiliates
# For model definition, see models/robust_regression_model.py
# Test this as follows:
# python PPLBench.py -m robust_regression -l bmgraph -n 2000 -k 40 --trials 1
import torch  # isort:skip torch has to be imported before bmgraph # noqa: F401
import time
from typing import Any, Dict, List, Tuple

# TODO: why is pyre not able to find beanmachne.graph?
# pyre-ignore-all-errors
import beanmachine.graph as bmgraph
import numpy as np


def obtain_posterior(
    data_train: Tuple[Any, Any], args_dict: Dict, model
) -> Tuple[List, Dict]:
    """
    Beanmachine impmementation of logisitc regression model.

    :param data_train: tuple of np.ndarray (x_train, y_train)
    :param args_dict: a dict of model arguments
    :returns: samples_beanmachine(dict): posterior samples of all parameters
    :returns: timing_info(dict): compile_time, inference_time
    """
    x_train, y_train = data_train
    K, N = x_train.shape  # K = num features, N = num observations
    assert y_train.shape == (N,)
    assert model is None
    num_samples = int(args_dict["num_samples"])
    alpha_scale, beta_scale, beta_loc, sigma_mean = args_dict["model_args"]
    # compile the model
    compile_start = time.time()
    g = bmgraph.Graph()
    # nu ~ Gamma(shape = 2, rate = 0.1)
    two = g.add_constant_pos_real(2.0)
    pt_one = g.add_constant_pos_real(0.1)
    nu_prior = g.add_distribution(
        bmgraph.DistributionType.GAMMA, bmgraph.AtomicType.POS_REAL, [two, pt_one]
    )
    nu = g.add_operator(bmgraph.OperatorType.SAMPLE, [nu_prior])
    # sigma ~ Exponential(sigma_mean) = Gamma(shape = 1, rate = 1/sigma_mean)
    one = g.add_constant_pos_real(1.0)
    inv_sigma_mean = g.add_constant_pos_real(float(1.0 / sigma_mean))
    sigma_prior = g.add_distribution(
        bmgraph.DistributionType.GAMMA,
        bmgraph.AtomicType.POS_REAL,
        [one, inv_sigma_mean],
    )
    sigma = g.add_operator(bmgraph.OperatorType.SAMPLE, [sigma_prior])
    # alpha ~ Normal(0, alpha_scale)
    zero = g.add_constant(0.0)
    alpha_scale = g.add_constant_pos_real(float(alpha_scale))
    alpha_prior = g.add_distribution(
        bmgraph.DistributionType.NORMAL, bmgraph.AtomicType.REAL, [zero, alpha_scale]
    )
    alpha = g.add_operator(bmgraph.OperatorType.SAMPLE, [alpha_prior])
    # beta_j ~ Normal(beta_loc, beta_scale) for j in range(K)
    beta_scale = g.add_constant_pos_real(float(beta_scale))
    beta_loc = g.add_constant(float(beta_loc))
    beta_prior = g.add_distribution(
        bmgraph.DistributionType.NORMAL, bmgraph.AtomicType.REAL, [beta_loc, beta_scale]
    )
    beta = [g.add_operator(bmgraph.OperatorType.SAMPLE, [beta_prior]) for _ in range(K)]
    # mean_i = alpha + sum(x_j_i * beta_j for j in range(K))
    for i in range(N):
        sum_i = []
        for j in range(K):
            x_j_i = g.add_constant(float(x_train[j, i]))
            x_j_i_times_beta_j = g.add_operator(
                bmgraph.OperatorType.MULTIPLY, [x_j_i, beta[j]]
            )
            sum_i.append(x_j_i_times_beta_j)
        sum_i.append(alpha)
        mean_i = g.add_operator(bmgraph.OperatorType.ADD, sum_i)
        # y_i ~ StudentT(nu, mean_i, sigma)
        studentt_i = g.add_distribution(
            bmgraph.DistributionType.STUDENT_T,
            bmgraph.AtomicType.REAL,
            [nu, mean_i, sigma],
        )
        y_i = g.add_operator(bmgraph.OperatorType.SAMPLE, [studentt_i])
        g.observe(y_i, float(y_train[i]))
    # We want P (nu, sigma, alpha, beta_j for j in range(K)
    #            | x_i, y_i  for i in range(N))
    g.query(nu)
    g.query(sigma)
    g.query(alpha)
    for j in range(K):
        g.query(beta[j])
    # we will include the inference time for one sample into the compile time
    # and subtract this from inference time later
    infer_setup_start = time.time()
    g.infer(1, bmgraph.InferenceType.NMC)
    infer_setup_time = time.time() - infer_setup_start
    compile_time = time.time() - compile_start
    infer_start = time.time()
    samples = g.infer(
        num_samples, bmgraph.InferenceType.NMC, int(time.time() * 1000) % 2 ** 31
    )
    infer_time = time.time() - infer_start - infer_setup_time
    timing_info = {"compile_time": compile_time, "inference_time": infer_time}
    print(f"bmgraph inference time {infer_time:.1f}")
    sample_dicts = []
    for sample in samples:
        dict_ = {}
        dict_["nu"] = sample[0]
        dict_["sigma"] = sample[1]
        dict_["alpha"] = sample[2]
        dict_["beta"] = np.array(sample[3:])
        sample_dicts.append(dict_)
    return sample_dicts, timing_info
