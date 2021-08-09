# Contributing to PPL Bench

We welcome any contributions - whether it is adding a new model, adding a new PPL  implementation, or modifying an existing PPL implementation.

### How to add a new PPL?

To add a new PPL, first, create a new directory in `pplbench/ppls`. Then, create implementations of `BasePPLInference` specific to the added PPL and chosen inference algorithms available for that system. For example, `pplbench/ppls/stan/inference.py` implements `BaseStanInference` which serves as a basis for two inference algorithms offered by the Stan system: `MCMC` and `VI`.

These inference classes must be able to solve a model implementation object provided to them as an argument. This object must be of a type derived from `BasePPLImplementation`. In the Stan example, the class `BaseStanImplementation` (in `pplbench/ppls/stan/base_stan_impl.py`) provides a Stan-specific interface for  information about Stan models for the inference algorithms to use, and specific model classes are sub-classes of `BaseStanImplementation` (for example, `RobustRegression` in `pplbench/ppls/stan/robust_regression.py`).

### How to add a new model?

To add a new model, you need to write code to generate and test data according to this new model (this data will be shared by all PPLs for evaluation), as well as code for encoding and running inference for this new model for each PPL that will be applied to it.

#### Generating and testing data for a New Model

To write the code to generate and test the data for the model, create a new file in `pplbench/models`. The model will need to inherit from `BaseModel` and implement two methods — `generate_data` and `evaluate_posterior_predictive`.

`generate_data` should return the train and test data by simulating from the generative model.

`evaluate_posterior_predictive` should return the predictive log likelihood of the data given samples from the inferred parameters.

#### Encoding the model in a particular PPL

To encode a new model with a PPL, you need to add a sub-class of `BasePPLImplementation` corresponding to it. One example is the aforementioned `RobustRegression` in `pplbench/ppls/stan/robust_regression.py`, but you can define your own new models by writing classes like that.

Note that, for convenience, a `BaseStanImplementation` class was derived from `BasePPLImplementation` and defines a shared interface for all Stan model implementations. A similar pattern applies to the other PPLs. Therefore you will very likely want to reuse such base classes for convenience when defining new models for these systems, or define such an analogous base class when introducing a completely new PPL.

## Pull Requests
We welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to pplbench, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
