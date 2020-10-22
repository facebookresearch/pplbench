---
id: models
title: Models
---

PPL Bench currently has the following models.

## Bayesian Logistic Regression

* Simple model; baseline
* Log-concave posterior, easy convergence
* For a detailed description of this model, go [here](https://github.com/facebookresearch/pplbench/blob/master/pplbench/models/logistic_regression.py).

## Robust Regression

* Increased robustness to outliers
* Uses a Bayesian regression model with Student-T errors
* For a detailed description of this model, go [here](https://github.com/facebookresearch/pplbench/blob/master/pplbench/models/robust_regression.py).

## Noisy-Or Topic Model

* Inferring topics from words in a document
* Bayesian Netrowk structure with topics and words as nodes
* Supports hierarchical topics
* For a detailed description of this model, go [here](https://github.com/facebookresearch/pplbench/blob/master/pplbench/models/noisy_or_topic.py).


## Crowdsourced Annotation

* Inferring true label of an object given multiple labeler's label assignments
* Maintain confusion matrix of each labeler
* Includes inferring the unknown prevalence of labels
* For a detailed description of this model, go [here](https://github.com/facebookresearch/pplbench/blob/master/pplbench/models/crowd_sourced_annotation.py).


## Adding New Models
PPL Bench supports adding new models. Please refer to [CONTRIBUTING.md](https://github.com/facebookresearch/pplbench/blob/master/CONTRIBUTING.md) for details on how to do so!
