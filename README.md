# Getting Started with PPLBench

## What is PPLBench?

PPLBench is a benchmarking framework for evaluating the performance of various PPLs on statistical models. It is designed to be modular so new models and PPL implementations of models can be added into this framework. The fundamental evaluation metric is the log predictive likelihood on a held out test dataset. We believe that log predictive likelihood is the common denominator across which PPL accuracy & convergence can be measured.

## Purpose of PPL Bench

The purpose of PPL Bench as a probabilistic programming benchmark is two-fold.

First, we want researchers as well as conference reviewers to be able to evaluate improvements in PPLs in a standardized setting. In essence, we’d like to create a similar standard to one like ImageNet, but for PPL inference measurement.

Second, we want end users to be able to pick the PPL that is most suited for their modeling application.

Typically, comparing different ML systems requires duplicating huge segments of work: generating data, running analysis, determining predictive performance, and comparing across implementations. PPL Bench automates nearly all of this workflow.


## License

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

## How to use PPLBench?

### Installation:

Following is the procedure to install PPLBench on Linux (Tested on Ubuntu 18.04):

1. Download/Clone PPLBench:
    `git clone https://github.com/facebookresearch/pplbench.git`

2. Installing dependencies:
    1. PPLBench core:
        `pip install -r requirements.txt`
    2. PPLs (Only need to install the ones which you want to benchmark):
        1. Stan
            1. `pip install pystan`
        2. Jags
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
            3. `pip install numpyro==0.2.0`

### Example:

Let us go through an example to check if the installation is working. From the PPLBench directory, run the following command:

```
python PPLBench.py -m robust_regression -l jags,stan -k 5 -n 2000 -s 500 --trials 2
```

To see supported models, PPL implementations and command line arguments, type:

`python PPLBench.py -h`

To get started, there are reference models and PPL implementations. Please submit pull requests to modify an existing PPL implementation or to add a new PPL or model.

### Adding a new PPL:

Given the modularity of the framework, to add a new PPL implementation to PPLBench, you need to implement one function — `obtain_posterior`.

At a high level, this function should return samples from the posterior after inference is completed.

### Adding a new Model:

To add a new model, you need to implement two methods — `generate_data` and `evaluate_posterior_predictive`.

`generate_data` should return the train and test data by simulating from the generative model.

`evaluate_posterior_predictive` should return the predictive log likelihood of the data given samples from the inferred parameters.

### Next steps:

 For more information about PPLBench, refer to

1. Blog post: [LINK TO BLOG POST]
2. Paper: [LINK TO PAPER]
