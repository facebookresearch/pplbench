# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pickle
import time

import numpy as np
import pystan


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

generated quantities{
  real log_lik = 0.0;
  real log_post = 0.0;
  vector[q] cauchy_prior;
  vector[q] normal_prior;
  real student_t_prior = 0.0;

  for (j in 1:q){
    cauchy_prior[j] = 0.0;
    normal_prior[j] = 0.0;
  }

  student_t_prior = student_t_lpdf(beta_c[1] | 3, 0, 10);
  log_post += student_t_prior;
  for (j in 1:q) {
    cauchy_prior[j] = cauchy_lpdf(sds[j] | 0, sd_tau) - 1 * cauchy_lccdf(0 | 0, sd_tau);
    log_post += cauchy_prior[j];
    normal_prior[j] = normal_lpdf(u_unscaled[ranges[j,1]:ranges[j,2]] | 0, 1);
    log_post += normal_prior[j];
  }

  log_lik += normal_lpdf(yi | yhati, sei);
  log_post += log_lik;
}
"""


def obtain_posterior(data_train, args_dict, model=None):
    """
    Stan impmementation of n-school model.

    Inputs:
    - data_train(tuple of np.ndarray): x_train, y_train
    - args_dict: a dict of model arguments
    Returns:
    - samples_stan(dict): posterior samples of all parameters
    - timing_info(dict): compile_time, inference_time
    """
    global CODE
    x_train, y_train = data_train
    N = int(x_train.shape[1])
    K = int(x_train.shape[0])
    alpha_scale, beta_scale, beta_loc, _ = args_dict["model_args"]

    data_stan = {
        "N": N,
        "K": K,
        "X": x_train.T,
        "y": y_train,
        "scale_alpha": alpha_scale,
        "scale_beta": beta_scale * np.ones(K),
        "beta_loc": beta_loc * np.ones(K),
    }

    code_loaded = None
    pkl_filename = os.path.join(args_dict["output_dir"], "stan_logisticRegression.pkl")
    if os.path.isfile(pkl_filename):
        model, code_loaded, elapsed_time_compile_stan = pickle.load(
            open(pkl_filename, "rb")
        )
    if code_loaded != CODE:
        # compile the model, time it
        start_time = time.time()
        model = pystan.StanModel(model_code=CODE, model_name="logistic_regression")
        elapsed_time_compile_stan = time.time() - start_time
        # save it to the file 'model.pkl' for later use
        with open(pkl_filename, "wb") as f:
            pickle.dump((model, CODE, elapsed_time_compile_stan), f)

    if args_dict["inference_type"] == "mcmc":
        # sample the parameter posteriors, time it
        start_time = time.time()
        fit = model.sampling(
            data=data_stan,
            iter=int(args_dict["num_samples_stan"]),
            chains=1,
            check_hmc_diagnostics=False,
        )
        samples_stan = fit.extract(
            pars=["beta_c", "sds", "u_unscaled"], permuted=False, inc_warmup=True
        )
        elapsed_time_sample_stan = time.time() - start_time

    elif args_dict["inference_type"] == "vi":
        # sample the parameter posteriors, time it
        start_time = time.time()
        fit = model.vb(data=data_stan, iter=args_dict["num_samples_stan"])
        samples_stan = fit.extract(
            pars=["beta_c", "sds", "u_unscaled"], permuted=False, inc_warmup=True
        )
        elapsed_time_sample_stan = time.time() - start_time

    # repackage samples into shape required by PPLBench
    samples = []
    for i in range(int(args_dict["num_samples_stan"])):
        sample_dict = {}
        for parameter in samples_stan.keys():
            sample_dict[parameter] = samples_stan[parameter][i]
        samples.append(sample_dict)
    timing_info = {
        "compile_time": elapsed_time_compile_stan,
        "inference_time": elapsed_time_sample_stan,
    }

    return (samples, timing_info)
