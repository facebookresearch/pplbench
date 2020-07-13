# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import time
from typing import Dict

import beanmachine.ppl as bm
import pandas as pd
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.inference_compilation.ic_infer import ICInference
from beanmachine.ppl.model.statistical_model import RVIdentifier
from torch import Tensor, tensor

from ..pplbench_ppl import PPLBenchPPL


class NSchoolsVectorizedModel:
    def __init__(
        self,
        df,
        num_samples,
        num_districts,
        num_states,
        num_types,
        sd_district_scale=1,
        sd_state_scale=1,
        sd_type_scale=1,
    ):
        self._df = df
        self._num_samples = num_samples
        self._sd_district_scale = sd_district_scale
        self._sd_state_scale = sd_state_scale
        self._sd_type_scale = sd_type_scale
        super().__init__()

        self._u_sizes = [num_districts * num_states, num_states, num_types]
        self.num_levels = len(self._u_sizes)
        self.torch_re_maps = []
        for i in range(self.num_levels):
            re_map_i = torch.zeros((self._df.shape[0], self._u_sizes[i]))
            for j in range(self._df.shape[0]):
                if i == 0:
                    # beta_district
                    re_map_i[
                        j,
                        self._df.state[j] * self._df.district[j] + self._df.district[j],
                    ] = 1
                elif i == 1:
                    # beta_state
                    re_map_i[j, self._df.state[j]] = 1
                elif i == 2:
                    # beta_type
                    re_map_i[j, self._df.type[j]] = 1
                else:
                    raise Exception("Should not occur.")
            self.torch_re_maps.append(re_map_i)

        self.queries = [self.beta_0()]
        for i in range(self.num_levels):
            self.queries += [self.sds(i), self.u(i)]
        self.observations: Dict[RVIdentifier, Tensor] = {
            self.y(): tensor(self._df.yi).float()
        }

        self._icmh = ICInference()
        self._icmh.compile(list(self.observations.keys()))

    @bm.random_variable
    def beta_0(self):
        return dist.StudentT(3, loc=0.0, scale=10.0)

    @bm.random_variable
    def sds(self, i):
        assert i < self.num_levels
        return dist.HalfCauchy(
            [self._sd_district_scale, self._sd_state_scale, self._sd_type_scale][i]
        )

    @bm.random_variable
    def u(self, i):
        assert i < self.num_levels
        return dist.Normal(torch.zeros(self._u_sizes[i]), self.sds(i).item())

    @bm.random_variable
    def y(self):
        yhat = self.beta_0() + sum(
            torch.mm(self.torch_re_maps[i], self.u(i).reshape(self._u_sizes[i], 1))
            for i in range(self.num_levels)
        )
        yhat = yhat.reshape(-1)
        return dist.Normal(yhat, tensor(self._df.sei).float())

    def infer(self):
        return self._icmh.infer(
            self.queries, self.observations, num_samples=self._num_samples, num_chains=1
        ).get_chain()


class NSchoolsVectorized(PPLBenchPPL):
    def obtain_posterior(self, data_train, args_dict: Dict, model):
        """
        Beanmachine vectorized implementation of n-schools.

        :param data_train:
        :param args_dict: a dict of model arguments
        :returns: samples_beanmachine(dict): posterior samples of all parameters
        :returns: timing_info(dict): compile_time, inference_time
        """
        num_samples = args_dict["num_samples_beanmachine"]
        num_states, num_districts, num_types = [int(x) for x in args_dict["model_args"]]

        compile_start = time.time()
        model = NSchoolsVectorizedModel(
            data_train,
            num_samples=num_samples,
            num_districts=num_districts,
            num_states=num_states,
            num_types=num_types,
        )
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
        for i, level in enumerate(["district", "state", "type"]):
            samples_formatted[f"sd_{level}"] = samples.get_variable(model.sds(i))

            u_i = samples.get_variable(model.u(i))
            for j in range(model._u_sizes[i]):
                if i == 0:
                    # beta_district
                    state = int(j / num_districts)
                    district = j % num_districts
                    samples_formatted[f"beta_state_{state}_district_{district}"] = u_i[
                        :, j
                    ]
                else:
                    samples_formatted[f"beta_{level}_{j}"] = u_i[:, j]

        return (samples_formatted, timing_info)
