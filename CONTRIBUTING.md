# Contributing to PPL Bench

We welcome any contributions - whether it is adding a new model, adding a new PPL  implementation, or modifying an existing PPL implementation.

### How to add a new PPL?

To add a new PPL, first, create a new directory in `pplbench/ppl`. Then, create a base implementation that future implementations of the PPL can use. You can refer to `pplbench/ppls/stan/base_stan_impl.py` as an example.

This base PPL implementation must inherit from `BasePPLImplementation`.


### How to add a new model?

To add a new model, create a new file in `pplbench/models`. The model will need to inherit from `BaseModel` and implement two methods — `generate_data` and `evaluate_posterior_predictive`.

`generate_data` should return the train and test data by simulating from the generative model.

`evaluate_posterior_predictive` should return the predictive log likelihood of the data given samples from the inferred parameters.


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
