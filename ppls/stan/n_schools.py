# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import time

import numpy as np
import pystan

from ..pplbench_ppl import PPLBenchPPL


CODE = """
data {
  int<lower=1> n;                   // no. of observations
  int<lower=1> p;                   // no. of population-level effects
  int<lower=1> q;                   // no. of random terms
  vector[n] yi;                     // effect sizes
  vector<lower=0>[n] sei;           // s.e. of effect sizes
  matrix[n, p] X;                   // population-level design matrix
  int ranges[q + 1, 2];             // Adding 1 more row than necessary to
                                    // handle singleton dimension issue
  int<lower=1> sum_ns;
  int<lower=1> mapping[n, q];
  real sd_tau;
}

transformed data {
  matrix[n, p] X_c;                  // centered X
  if (p > 1) {
    X_c[, 1] = X[, 1];
    for (i in 2:p) {
      X_c[, i] = X[, i] - mean(X[, i]);
    }
  } else {
    X_c = X;
  }
}

parameters {
  vector[p] beta_c;                 // fixed effects
  real<lower=0> sds[q];             // group-level standard deviations
  vector[sum_ns] u_unscaled;        // unscaled random effects
}

transformed parameters {
  vector[sum_ns] u;                 // random effects
  vector[n] yhati;                  // true effects

  for (j in 1:q) {
    u[ranges[j,1]:ranges[j,2]] = sds[j] * u_unscaled[ranges[j,1]:ranges[j,2]];
  }

  yhati = X_c * beta_c;
  for (i in 1:n) {
    for (j in 1:q) {
       yhati[i] += u[ranges[j,1] + mapping[i,j] - 1];
    }
  }
}
model {

  // prior for fixed effects
  target += student_t_lpdf(beta_c[1] | 3, 0, 10);

  // priors for random effects
  for (j in 1:q) {
    target += cauchy_lpdf(sds[j] | 0, sd_tau) - 1 * cauchy_lccdf(0 | 0, sd_tau);
    target += normal_lpdf(u_unscaled[ranges[j,1]:ranges[j,2]] | 0, 1);
  }

  target += normal_lpdf(yi | yhati, sei);
}
"""


class NSchools(PPLBenchPPL):
    # define factorize
    def _factorize(self, random_effect, data):
        combined = (
            data[random_effect]
            .apply(lambda row: "____".join(row.values.astype(str)), axis=1)
            .values
        )
        lvl2num = {}
        i = 1
        for val in combined:
            if val not in lvl2num:
                lvl2num[val] = i
                i += 1
        num2lvl = {i: v for v, i in lvl2num.items()}
        random_effect_level_map = {"num2lvl": num2lvl, "lvl2num": lvl2num}
        return random_effect_level_map

    # get _get_n_levels
    def _get_n_levels(self, data):
        ns = np.array([], dtype=int)
        for random_effect in [["state", "district"], ["type"]]:
            for i in range(len(random_effect), 0, -1):
                curr_lvls = random_effect[:i]
                lvl_map = self._factorize(curr_lvls, data)
                lvl2num = lvl_map["lvl2num"]
                ns = np.append(ns, len(lvl2num))
        return ns

    # define mapping
    def _get_mapping(self, data):
        mapping = np.empty((0, data.shape[0]), dtype=int)
        for random_effect in [["state", "district"], ["type"]]:
            for i in range(len(random_effect), 0, -1):
                curr_lvls = random_effect[:i]
                lvl_map = self._factorize(curr_lvls, data)
                lvl2num = lvl_map["lvl2num"]
                combined = (
                    data[curr_lvls]
                    .apply(lambda row: "____".join(row.values.astype(str)), axis=1)
                    .values
                )
                curr_map = np.array([lvl2num[x] for x in combined])
                mapping = np.append(mapping, [curr_map], axis=0)

        return mapping.T

    # define ranges
    def _get_ranges(self, data):
        n_levels = self._get_n_levels(data)
        cumsum_ns = np.cumsum(n_levels, dtype=int)
        tmp = np.concatenate((cumsum_ns, np.array([0], dtype=int)))
        start = np.concatenate(
            (np.array([1], dtype=int), 1 + cumsum_ns[:-1], np.array([0], dtype=int))
        )
        ranges = np.array((start, tmp)).T
        return ranges

    def obtain_posterior(self, data_train, args_dict, model=None):
        """
      Stan impmementation of n-school model.

      Inputs:
      - data_train
      - args_dict: a dict of model arguments
      Returns:
      - samples_stan(dict): posterior samples of all parameters
      - timing_info(dict): compile_time, inference_time
      """
        global CODE

        # mapping
        mapping = self._get_mapping(data_train)

        # ranges
        ranges = self._get_ranges(data_train)

        # n_levels
        n_levels = self._get_n_levels(data_train)

        # define sum_ns
        sum_ns = np.sum(n_levels)

        # generating stan mapping to state/district/type values!
        re_map_to_stan_params = {}
        for i in range(data_train.shape[0]):
            re_q_position = 0
            for random_effect in [["state", "district"], ["type"]]:
                for j in range(len(random_effect), 0, -1):
                    curr_lvls = random_effect[:j]
                    combined = "beta_" + "_".join(
                        [lvl + "_" + str(data_train[lvl].iloc[i]) for lvl in curr_lvls]
                    )
                    if combined not in re_map_to_stan_params:
                        re_map_to_stan_params[combined] = (ranges[re_q_position][0]) + (
                            mapping[i][re_q_position] - 1
                        )
                    re_q_position += 1

        data_stan = {
            "n": data_train.shape[0],
            "p": 1,
            "q": 3,
            "yi": data_train["yi"].values,
            "sei": [i.item() for i in data_train["sei"].values],
            "X": np.ones(shape=data_train.shape[0])[..., None],
            "ranges": ranges,
            "sum_ns": sum_ns,
            "mapping": mapping,
            "sd_tau": 1.0,
        }

        # compile the model, time it
        start_time = time.time()
        model = pystan.StanModel(model_code=CODE, model_name="n_schools")
        elapsed_time_compile_stan = time.time() - start_time

        if args_dict["inference_type"] == "vi":
            raise Exception("VI is not supported for the N_schools model.")

        elif args_dict["inference_type"] == "mcmc":
            # sample the parameter posteriors, time it
            start_time = time.time()
            fit = model.sampling(
                data=data_stan,
                iter=int(args_dict["num_samples_stan"]),
                chains=1,
                check_hmc_diagnostics=False,
            )
            samples_stan = fit.to_dataframe(
                pars=["beta_c", "sds", "u"],
                permuted=False,
                inc_warmup=True,
                diagnostics=False,
            )
            elapsed_time_sample_stan = time.time() - start_time

            # rename columns to be similar to BMG n_school
            samples_stan.rename(
                columns={f"u[{v}]": k for k, v in re_map_to_stan_params.items()},
                inplace=True,
            )

            samples_stan.rename(columns={"beta_c[1]": "beta_0"}, inplace=True)

            timing_info = {
                "compile_time": elapsed_time_compile_stan,
                "inference_time": elapsed_time_sample_stan,
            }

            return (samples_stan, timing_info)
