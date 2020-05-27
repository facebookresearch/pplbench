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
beta_state ~ Normal(0, sd_state)

district level effect, nested within state:

sd_state_district ~ HalfCauchy(1)
beta_state_district ~ Normal(0, sd_state_district)

non-nested school type effect:

sd_type ~ HalfCauchy(1)
beta_type  ~ Normal(0, sd_type)

SAT score mean % change for i th school:

y_i ~ Normal(beta_0 + beta_state + beta_state_district + beta_type, sei)

where the state, district, and type are of the i th school and sei is the standard
deviation of the observed effect at the school.

Model specific arguments:
return number of states, districts, and school types
"""

import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
import torch.tensor as tensor
from tqdm import tqdm


def get_defaults():
    defaults = {"n": 2000, "k": 10, "model_args": [8, 5, 5], "train_test_ratio": 0.5}
    return defaults


def generate_model(args_dict):
    """
    Generate parameters for seismic model.

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

    sd_states = dist.HalfCauchy(torch.ones(num_states)).sample()
    sd_districts = dist.HalfCauchy(torch.ones(num_states, num_districts)).sample()
    sd_types = dist.HalfCauchy(torch.ones(num_types)).sample()

    beta_states = dist.Normal(0.0, sd_states).sample()
    beta_districts = dist.Normal(0.0, sd_districts).sample()
    beta_types = dist.Normal(0.0, sd_types).sample()

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

    :param samples: list of dictionaries with format {
        "beta_0": value of beta_0
        ("beta_state", s (int)): value of beta_state for state s
        ("beta_district", s (int), d(int)): value of beta_state for state s district d
        ("beta_type", t (int)): value of beta_type for school type t
    }
    :param data_test: test data
    :param model: dictionary with number of states, districts, and school types
    :returns: log-likelihoods of data wrt parameter samples
    """
    pred_llh = []
    for s in tqdm(samples, desc="eval", leave=False):
        llh = tensor(0.0)
        for _, row in data_test.iterrows():
            yi = row["yi"]
            sei = row["sei"]
            state = row["state"]
            district = row["district"]
            school_type = row["type"]

            beta_0 = s["beta_0"]
            beta_state = s[("beta_state", state)]
            beta_district = s[("beta_district", state, district)]
            beta_type = s[("beta_type", school_type)]

            yhat = beta_0 + beta_state + beta_district + beta_type
            llh += dist.Normal(yhat, sei).log_prob(yi).sum()

        pred_llh.append(llh)

    # return as a numpy array
    return np.array(pred_llh)
