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
import pandas as pd

from ..pplbench_ppl import PPLBenchPPL


class NSchools(PPLBenchPPL):
    def obtain_posterior(
        self, data_train: Tuple[Any, Any], args_dict: Dict, model
    ) -> Tuple[List, Dict]:
        num_states, num_districts, num_types = [int(x) for x in args_dict["model_args"]]
        compile_start = time.time()
        g = bmg.Graph()
        nodeidx = {}  # record the node ids of the latent variables
        queryidx = {}  # also the query index of the latent variables

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
        one = g.add_constant_pos_real(1.0)
        sd_prior = g.add_distribution(
            bmg.DistributionType.HALF_CAUCHY, bmg.AtomicType.POS_REAL, [one]
        )
        sd_state = g.add_operator(bmg.OperatorType.SAMPLE, [sd_prior])
        sd_district = g.add_operator(bmg.OperatorType.SAMPLE, [sd_prior])
        sd_type = g.add_operator(bmg.OperatorType.SAMPLE, [sd_prior])
        nodeidx["sd_state"] = sd_state
        nodeidx["sd_district"] = sd_district
        nodeidx["sd_type"] = sd_type

        # beta_state[i] ~ Normal(0, sd_state)
        # best_district[i, j] ~ Normal(0, sd_district)
        beta_state_prior = g.add_distribution(
            bmg.DistributionType.NORMAL, bmg.AtomicType.REAL, [zero, sd_state]
        )
        beta_district_prior = g.add_distribution(
            bmg.DistributionType.NORMAL, bmg.AtomicType.REAL, [zero, sd_district]
        )
        for state in range(num_states):
            nodeidx[f"beta_state_{state}"] = g.add_operator(
                bmg.OperatorType.SAMPLE, [beta_state_prior]
            )
            for district in range(num_districts):
                nodeidx[f"beta_state_{state}_district_{district}"] = g.add_operator(
                    bmg.OperatorType.SAMPLE, [beta_district_prior]
                )
        for type_ in range(num_types):
            beta_type_prior = g.add_distribution(
                bmg.DistributionType.NORMAL, bmg.AtomicType.REAL, [zero, sd_type]
            )
            nodeidx[f"beta_type_{type_}"] = g.add_operator(
                bmg.OperatorType.SAMPLE, [beta_type_prior]
            )

        # yhat[n] = beta_0 + beta_state[i[n]] + beta_district[i[n], j[n]] + beta_type[k[n]]
        # y[n] ~ N(yhat[n], sei[n])
        for _, row in data_train.iterrows():
            yhat = g.add_operator(
                bmg.OperatorType.ADD,
                [
                    nodeidx["beta_0"],
                    nodeidx[f"beta_state_{row.state}"],
                    nodeidx[f"beta_state_{row.state}_district_{row.district}"],
                    nodeidx[f"beta_type_{row.type}"],
                ],
            )
            sei = g.add_constant_pos_real(row.sei)
            y_prior = g.add_distribution(
                bmg.DistributionType.NORMAL, bmg.AtomicType.REAL, [yhat, sei]
            )
            y = g.add_operator(bmg.OperatorType.SAMPLE, [y_prior])
            g.observe(y, row.yi)

        # query all the beta_... nodes and the sd_.. nodes
        for nodename, nodeid in nodeidx.items():
            queryidx[nodename] = g.query(nodeid)

        compile_time = time.time() - compile_start
        infer_start = time.time()
        seed = np.random.randint(1000, 1000000)
        samples = np.array(
            g.infer(args_dict["num_samples"], bmg.InferenceType.NMC, seed)
        )
        infer_time = time.time() - infer_start
        timing_info = {"compile_time": compile_time, "inference_time": infer_time}

        sample_dataframe = pd.DataFrame()
        for nodename, queryid in queryidx.items():
            sample_dataframe[nodename] = samples[:, queryid]
        print(f"bmgraph inference time {infer_time:.1f}")

        return sample_dataframe, timing_info
