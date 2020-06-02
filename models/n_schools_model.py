# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Module for N Schools Model

Module contains following methods:
get_defaults()
generate_model()
generate_data()
evaluate_posterior_predictive()

Model Specification:
A hierarchical model depicting school-level student SAT score improvement

average school effect:

beta_0 ~ StudentT(dof=3, loc=0.0, scale=10.0)

state level effect:

sd_state ~ HalfCauchy(1)

for i in range(num_states):
    beta_state[i] ~ Normal(0, sd_state)

district level effect, nested within state:

sd_district ~ HalfCauchy(1)

for i in range(num_states):
    for j in range(num_districts):
        beta_district[i, j] ~ Normal(0, sd_district)

non-nested school type effect:

sd_type ~ HalfCauchy(1)

for k in range(num_school_types):
    beta_type[k]  ~ Normal(0, sd_type)

SAT score mean % change for n th school:

y[n] ~ Normal(beta_0 + beta_state[i[n]] + beta_district[i[n], j[n]] + beta_type[k[n]], sei[n])

where the i[n], j[n], and k[n] are the state, district, and type respectively of
the n th school and sei[n] is the standard deviation of the observed effect at the school.

Inference Task:
    Given y[n], i[n], j[n], k[n], and sei[n]
    Infer beta_0, beta_state[], beta_district[,], and beta_type[]
    Also infer sd_type, sd_state, and sd_district

Model specific arguments:
    (num_states, num_districts, num_school_types)
"""

import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
from scipy.stats import norm


def get_defaults():
    defaults = {
        "n": 2000,
        "k": 10,
        "runtime": 200,
        "model_args": [8, 5, 5],
        "train_test_ratio": 0.5,
    }
    return defaults


def generate_model(args_dict):
    """
    Generate parameters for n schools model.

    :param args_dict: arguments dictionary
    :returns: None
    """
    return None


def generate_data(args_dict, model):
    """
    Generate data for n schools model.

    :param args_dict: arguments dictionary with number of states, districts, and types
        args_dict["model_args"] = [num_states, num_districts, num_types]
    :param model: None
    :returns: generated_data(dict) = {
        data_train: pandas DataFrame with school Y_i, sei, state, district, and type
        data_test: a duplicate pandas DataFrame with the exact same combination of
            states, districts, and types, with different generated Y_i's.
    """
    print("Generating data")

    num_states, num_districts, num_types = [int(x) for x in args_dict["model_args"]]

    num_schools = int(args_dict["n"])

    beta_0 = dist.StudentT(3, 0.0, 10.0).sample()

    sd_state = dist.HalfCauchy(1.0).sample()
    sd_district = dist.HalfCauchy(1.0).sample()
    sd_type = dist.HalfCauchy(1.0).sample()

    beta_states = dist.Normal(0.0, sd_state).sample((num_states,))
    beta_districts = dist.Normal(0.0, sd_district).sample((num_states, num_districts))
    beta_types = dist.Normal(0.0, sd_type).sample((num_types,))

    data_train = []
    data_test = []
    for _ in range(num_schools):
        state = torch.randint(num_states, ()).item()
        district = torch.randint(num_districts, ()).item()
        school_type = torch.randint(num_types, ()).item()

        beta_state = beta_states[state]
        beta_district = beta_districts[state][district]
        beta_type = beta_types[school_type]

        yhat = beta_0 + beta_state + beta_district + beta_type
        sei = dist.Uniform(0.5, 1.5).sample().item()

        yi = dist.Normal(yhat, sei).sample().item()
        data_train.append((yi, sei, state, district, school_type))

        test_yi = dist.Normal(yhat, sei).sample().item()
        data_test.append((test_yi, sei, state, district, school_type))

    data_train = pd.DataFrame(
        data_train, columns=["yi", "sei", "state", "district", "type"]
    )
    data_test = pd.DataFrame(
        data_test, columns=["yi", "sei", "state", "district", "type"]
    )
    return {"data_train": data_train, "data_test": data_test}


def evaluate_posterior_predictive(samples, data_test, model):
    """
    Computes the likelihood of held-out testset wrt parameter samples

    :param samples: posterior samples in a form of a pandas DataFrame
    with columns as different random varialbles and rows are
    samples (iterations of MCMC).

    :param data_test: test data
    :param model: dictionary with number of states, districts, and school types
    :returns: log-likelihoods of data wrt parameter samples
    """
    pll_df = pd.DataFrame()
    rvs = samples.columns
    for index, row in data_test.iterrows():
        yhat = samples["beta_0"].copy().values

        # dealing with school type
        ky_tmp = f"beta_type_{int(row.type)}"
        if ky_tmp not in rvs:
            continue
        else:
            yhat += samples[ky_tmp].copy().values

        # dealing with state
        ky_tmp = f"beta_state_{int(row.state)}"
        if ky_tmp not in rvs:
            continue
        else:
            yhat += samples[ky_tmp].copy().values

        # dealing with state:district
        ky_tmp = f"beta_state_{int(row.state)}_district_{int(row.district)}"
        if ky_tmp not in rvs:
            continue
        else:
            yhat += samples[ky_tmp].copy().values

        likelihood = norm.pdf(yhat, loc=0, scale=row.sei)
        pll_df[f"yhati[{index}]"] = likelihood

    pll = np.log(pll_df.sum(axis=1)).copy().values
    return pll
