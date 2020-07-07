# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import time
from typing import Dict

import beanmachine.ppl as bm
import pandas as pd
import torch.distributions as dist
from beanmachine.ppl.experimental.inference_compilation.ic_infer import ICInference
from beanmachine.ppl.model.statistical_model import RVIdentifier
from torch import Tensor, tensor

from ..pplbench_ppl import PPLBenchPPL


class NSchoolsModel:
    def __init__(
        self, df, num_samples, sd_district_scale=1, sd_state_scale=1, sd_type_scale=1
    ):
        self._df = df
        self._num_samples = num_samples
        self._sd_district_scale = sd_district_scale
        self._sd_state_scale = sd_state_scale
        self._sd_type_scale = sd_type_scale
        super().__init__()

        self.queries = [
            self.beta_0(),
            self.sd_type(),
            self.sd_state(),
            self.sd_district(),
        ]
        for n in range(self._df.shape[0]):
            self.queries += [
                self.beta_type(self._df.type[n]),
                self.beta_state(self._df.state[n]),
                self.beta_district(self._df.state[n], self._df.district[n]),
            ]
        self.queries = list(set(self.queries))
        self.observations: Dict[RVIdentifier, Tensor] = {
            self.y(i): tensor(self._df.yi[i]).float() for i in range(self._df.shape[0])
        }

        self._icmh = ICInference()
        self._icmh.compile(list(self.observations.keys()))

    @bm.random_variable
    def beta_0(self):
        return dist.StudentT(3, loc=0.0, scale=10.0)

    @bm.random_variable
    def sd_district(self):
        return dist.HalfCauchy(self._sd_district_scale)

    @bm.random_variable
    def sd_state(self):
        return dist.HalfCauchy(self._sd_state_scale)

    @bm.random_variable
    def sd_type(self):
        return dist.HalfCauchy(self._sd_type_scale)

    @bm.random_variable
    def beta_state(self, state):
        return dist.Normal(0, self.sd_state().item())

    @bm.random_variable
    def beta_district(self, state, district):
        return dist.Normal(0, self.sd_district().item())

    @bm.random_variable
    def beta_type(self, type):
        return dist.Normal(0, self.sd_type().item())

    @bm.random_variable
    def y(self, i):
        y_hat_i = (
            self.beta_0().item()
            + self.beta_state(self._df.state[i]).item()
            + self.beta_district(self._df.state[i], self._df.district[i]).item()
            + self.beta_type(self._df.type[i]).item()
        )
        return dist.Normal(y_hat_i, self._df.sei[i].item())

    def infer(self):
        return self._icmh.infer(
            self.queries, self.observations, num_samples=self._num_samples, num_chains=1
        ).get_chain()


class NSchools(PPLBenchPPL):
    def obtain_posterior(self, data_train, args_dict: Dict, model):
        """
        Beanmachine inference compilation (IC) impmementation of n-schools.

        :param data_train:
        :param args_dict: a dict of model arguments
        :returns: samples_beanmachine(dict): posterior samples of all parameters
        :returns: timing_info(dict): compile_time, inference_time
        """
        num_samples = args_dict["num_samples_beanmachine"]
        num_states, num_districts, num_types = [int(x) for x in args_dict["model_args"]]

        compile_start = time.time()
        model = NSchoolsModel(data_train, num_samples=num_samples)
        elapsed_time_compile_beanmachine = time.time() - compile_start

        inference_start = time.time()
        samples = model.infer()
        elapsed_time_inference_beanmachine = time.time() - inference_start

        timing_info = {
            "compile_time": elapsed_time_compile_beanmachine,
            "inference_time": elapsed_time_inference_beanmachine,
        }

        samples_formatted = pd.DataFrame()
        samples_formatted["beta_0"] = samples.get_variable(model.beta_0())
        samples_formatted["sd_type"] = samples.get_variable(model.sd_type())
        samples_formatted["sd_state"] = samples.get_variable(model.sd_state())
        samples_formatted["sd_district"] = samples.get_variable(model.sd_district())
        for n in range(data_train.shape[0]):
            type_ = data_train.type[n]
            samples_formatted[f"beta_type_{type_}"] = samples.get_variable(
                model.beta_type(type_)
            )

            state = data_train.state[n]
            samples_formatted[f"beta_state_{state}"] = samples.get_variable(
                model.beta_state(state)
            )

            district = data_train.district[n]
            samples_formatted[
                f"beta_state_{state}_district_{district}"
            ] = samples.get_variable(model.beta_district(state, district))

        return (samples_formatted, timing_info)
