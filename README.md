# Getting Started with PPL Bench

## What is PPL Bench?

PPL Bench is a new benchmark framework for evaluating the performance of probabilistic programming languages (PPLs).

## Purpose of PPL Bench

The purpose of PPL Bench as a probabilistic programming benchmark is two-fold.

1) To provide researchers with a framework to evaluate improvements in PPLs in a standardized setting.
2) To enable users to pick the PPL that is most suited for their modeling application.

Typically, comparing different ML systems requires duplicating huge segments of work: generating data, running analysis, determining predictive performance, and comparing across implementations. PPL Bench automates nearly all of this workflow.

## Running

From the PPLBench directory, run the following command:

```python PPLBench.py -m [model] -l [ppls]  -k [covariates] -n [observations] -s [samples] --trials [trials]```

An example command would be:

```python PPLBench.py -m logisic_regression -l stan,pymc3 -k 10 -n 20000 -s 1000 --trials 2```

To see supported models, PPL implementations and command line arguments, type:

`python PPLBench.py -h`

To get started, there are reference models and PPL implementations. Please submit pull requests to modify an existing PPL implementation or to add a new PPL or model.

### Installing:

Following is the procedure to install PPLBench :

1. Download/Clone PPLBench:
    `git clone https://github.com/facebookresearch/pplbench.git`

2. Installing dependencies:
    1. Enter a virtual (or conda) environment
    2. PPLBench core:
        `pip install -r requirements.txt`
    3. PPLs (Only need to install the ones which you want to benchmark):
        1. Stan
            1. `pip install pystan`
        2. Jags (Tested on Ubuntu 18.04)
            1. `sudo apt-get install jags`
            2. `sudo apt install pkg-config`
            3. `pip install pyjags`
        3. pymc3
            1. `pip install pymc3`
        4. pyro:
            1. `pip install pyro-ppl==0.4.1`
        5. numpyro:
            1. `python3 -m pip install --upgrade pip`
            2. `pip install https://files.pythonhosted.org/packages/24/bf/e181454464b866f30f09b5d74d1dd08e8b15e032716d8bcc531c659776ab/jaxlib-0.1.37-cp36-none-manylinux2010_x86_64.whl`
            3. `pip install numpyro==0.3.0`

## How PPL Bench works

1) Generate Data

The first step is to simulate data (train and test) given the generative model and model parameters. To do this, one can use Numpy or any other Python library that can be used to draws samples from probability distributions. Once this is defined, when benchmarking this model, PPL Bench will use the data generated from this function across all PPLs.

2) Implement Model in a PPL

Once we have simulated data for a given model, PPL Bench will go through the PPLs which have implemented the model in question. For every PPL that you want to benchmark against, you will need a corresponding model implementation in that PPL.

3) Evaluate Different PPLs

PPLBench automatically generates predictive log likelihood plots on the same test dataset across all PPLs.

We support multiple trials, which runs inference on the same training data, multiple times. Our plots use multiple trials to generate confidence bands in our predictive log likelihood plots.

We also show other important statistics such as effective sample size, inference time, and r_hat.

Let's dive right in with a benchmark run of Bayesian Logistic Regression. Please
install stan (see above) and then run the following command:

```
python -m pplbench.main example.json
```

This will create a benchmark run with two trials of Stan on the Bayesian Logistic Regression model. The results of the run are saved in the `outputs/` directory. Please see the `example.json` file to understand the schema for specifying benchmark runs.

The schema is documented in `pplbench/main.py` and can be printed by running the help command:

```
python -m pplbench.main -h
```

A number of models is available in the `models` directory and the PPL implementations are available in the `ppls` directory.

Please feel free to submit pull requests to modify an existing PPL implementation or to add a new PPL or model.

### How to add a new PPL?

All PPL implementations must inherit from `PPLBenchPPL` and implement the `obtain_posterior` method.
At a high level, this function should return samples from the posterior after inference is completed.

### How to add a new model?

To add a new model, you need to implement two methods — `generate_data` and `evaluate_posterior_predictive`.

`generate_data` should return the train and test data by simulating from the generative model.

`evaluate_posterior_predictive` should return the predictive log likelihood of the data given samples from the inferred parameters.

## Join the PPL Bench community

 For more information about PPLBench, refer to

1. Blog post: [LINK TO BLOG POST]
2. Paper: [LINK TO PAPER]

See the CONTRIBUTING.md file for how to help out.

## License

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
