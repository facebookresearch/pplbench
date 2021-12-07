---
id: getting_started
title: Getting Started
---
This document outlines how to get started with PPL Bench.

Before jumping into the project, we recommend you read ["Why Bean Machine?"](why_bean_machine.md) and [System Overview](system_overview.md) documents.

## Installation

1. Enter a virtual (or conda) environment
2. Install PPL Bench core via pip:

```
pip install pplbench
```

3. Install PPLs that you wish to benchmark. For PPL-specific instructions, see [Installing PPLs](working_with_ppls.md).
You could also run the following command to install all PPLs that are currently supported by PPL Bench (except for Jags):

```
pip install pplbench[ppls]
```

Alternatively, you could also install PPL Bench from source. Please refer to [Installing PPLs](working_with_ppls.md)
for instructions.

## Launching PPL Bench

Let's dive right in with a benchmark run of Bayesian Logistic Regression. To run this, you'll need to install
PyStan (if you haven't already):

```
pip install pystan==2.19.1.1
```

Then, run PPL Bench with example config:

```
pplbench examples/example.json
```

This will create a benchmark run with two trials of Stan on the Bayesian Logistic Regression model. The results of the run are saved in the `outputs/` directory.

This is what the Predictive Log Likelihood (PLL) plot should look like:

![PLL plot of example run](../website/static/img/example_pystan_pll.svg)
![PLL half plot of example run](../website/static/img/example_pystan_pll_half.svg)

Please see the [examples/example.json](https://github.com/facebookresearch/pplbench/blob/main/examples/example.json) file to understand the schema for specifying benchmark runs. The schema is documented in [pplbench/main.py](https://github.com/facebookresearch/pplbench/blob/main/pplbench/main.py) and can be printed by running the help command:

```
pplbench -h
```

A number of models is available in the `pplbench/models` directory and the PPL implementations are available in the `pplbench/ppls` directory.

Please feel free to submit pull requests to modify an existing PPL implementation or to add a new PPL or model.


<!-- ## API References

For an in-depth reference of the various PPL Bench internals, see our [API Reference](ToADD). -->

## Contributing

You'd like to contribute to PPL Bench? Great! Please see [here](https://github.com/facebookresearch/pplbench/blob/main/CONTRIBUTING.md) for how to help out.


## Join the PPL Bench community

 For more information about PPL Bench, refer to

1. Website: [link](https://facebookresearch.github.io/pplbench/)
2. Blog post: [link](https://ai.facebook.com/blog/ppl-bench-creating-a-standard-for-benchmarking-probabilistic-programming-languages)
3. Paper: [link](https://arxiv.org/abs/2010.08886)
