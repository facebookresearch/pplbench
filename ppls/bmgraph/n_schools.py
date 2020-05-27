# Copyright (c) Facebook, Inc. and its affiliates
# For model definition, see models/n_schools_model.py
# Test this as follows:
# python PPLBench.py -m n_schools -l bmgraph -n 2000 --trials 1
import torch  # isort:skip torch has to be imported before bmgraph # noqa: F401
import time
from typing import Any, Dict, List, Tuple

# TODO: why is pyre not able to find beanmachne.graph?
# pyre-ignore-all-errors
import beanmachine.graph as bmg
import numpy as np


def obtain_posterior(
    data_train: Tuple[Any, Any], args_dict: Dict, model
) -> Tuple[List, Dict]:
    num_states, num_districts, num_types = [int(x) for x in args_dict["model_args"]]
    compile_start = time.time()
    g = bmg.Graph()
    nodeidx = {}  # record the node ids of the beta_.. variables
    queryidx = {}  # also the query index of the beta_.. variables

    # beta_0 ~ StudentT(3, 0, 10)
    three = g.add_constant_pos_real(3.0)
    zero = g.add_constant(0.0)
    ten = g.add_constant_pos_real(10.0)
    beta_0_prior = g.add_distribution(
        bmg.DistributionType.STUDENT_T, bmg.AtomicType.REAL, [three, zero, ten]
    )
    nodeidx["beta_0"] = g.add_operator(bmg.OperatorType.SAMPLE, [beta_0_prior])

    # sd_state ~ HalfCauchy(1)
    # sd_district ~ HalfCauchy(1)
    # sd_type ~ HalfCauchy(1)
    # beta_xxx ~ Normal(0, sd_xxx)
    one = g.add_constant_pos_real(1.0)
    sd_prior = g.add_distribution(
        bmg.DistributionType.HALF_CAUCHY, bmg.AtomicType.POS_REAL, [one]
    )
    for state in range(num_states):
        sd_state = g.add_operator(bmg.OperatorType.SAMPLE, [sd_prior])
        beta_state_prior = g.add_distribution(
            bmg.DistributionType.NORMAL, bmg.AtomicType.REAL, [zero, sd_state]
        )
        nodeidx["beta_state", state] = g.add_operator(
            bmg.OperatorType.SAMPLE, [beta_state_prior]
        )
        for district in range(num_districts):
            sd_state_district = g.add_operator(bmg.OperatorType.SAMPLE, [sd_prior])
            beta_state_district_prior = g.add_distribution(
                bmg.DistributionType.NORMAL,
                bmg.AtomicType.REAL,
                [zero, sd_state_district],
            )
            nodeidx["beta_district", state, district] = g.add_operator(
                bmg.OperatorType.SAMPLE, [beta_state_district_prior]
            )
    for type_ in range(num_types):
        sd_type = g.add_operator(bmg.OperatorType.SAMPLE, [sd_prior])
        beta_type_prior = g.add_distribution(
            bmg.DistributionType.NORMAL, bmg.AtomicType.REAL, [zero, sd_type]
        )
        nodeidx["beta_type", type_] = g.add_operator(
            bmg.OperatorType.SAMPLE, [beta_type_prior]
        )

    # yhat = beta_0 + beta_state + beta_state_district + beta_type
    # y ~ N(yhat, sei)
    for _, row in data_train.iterrows():
        yhat = g.add_operator(
            bmg.OperatorType.ADD,
            [
                nodeidx["beta_0"],
                nodeidx["beta_state", row.state],
                nodeidx["beta_district", row.state, row.district],
                nodeidx["beta_type", row.type],
            ],
        )
        sei = g.add_constant_pos_real(row.sei)
        y_prior = g.add_distribution(
            bmg.DistributionType.NORMAL, bmg.AtomicType.REAL, [yhat, sei]
        )
        y = g.add_operator(bmg.OperatorType.SAMPLE, [y_prior])
        g.observe(y, row.yi)

    # query all the beta_... nodes
    for nodename, nodeid in nodeidx.items():
        queryidx[nodename] = g.query(nodeid)

    compile_time = time.time() - compile_start
    infer_start = time.time()
    seed = np.random.randint(1000, 1000000)
    samples = g.infer(args_dict["num_samples"], bmg.InferenceType.NMC, seed)
    infer_time = time.time() - infer_start
    timing_info = {"compile_time": compile_time, "inference_time": infer_time}

    sample_dicts = []
    for sample in samples:
        dict_ = {}
        for nodename, queryid in queryidx.items():
            dict_[nodename] = sample[queryid]
        sample_dicts.append(dict_)

    print(f"bmgraph inference time {infer_time:.1f}")

    return sample_dicts, timing_info
