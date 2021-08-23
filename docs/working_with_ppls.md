---
id: working_with_ppls
title: Working with PPLs
---
For the PPL Bench to benchmark a probabilistic programming language, you need to first install the PPLs of interest.

## Stan

```
pip install pystan==2.19.1.1
```

## Jags

```
sudo apt-get install jags
sudo apt install pkg-config
pip install pyjags
```

 If you are installing in a conda environment:

```
conda install jags pkg-config
pip install pyjags
```

## PyMC3
```
pip install pymc3==3.9.0
```

## Pyro

```
pip install pyro-ppl==0.4.1
```

## NumPyro

```
pip install numpyro==0.3.0
```

## To Install PPL Bench from Source

```
git clone https://github.com/facebookresearch/pplbench.git
cd pplbench
pip install .
```
