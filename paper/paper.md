---
title: Likelihood-free Inference through Sample Summary Statistics Learning
author: Pablo de Castro, Tommaso Dorigo
abstract: >-
  Complex computer simulations are commonly required for accurate
  data modelling in many scientific disciplines, making statistical
  inference challenging due to the intractability of the likelihood
  evaluation for the observed data.
  Furthermore, sometimes we are interested only on inference over a subset
  of the generative model parameters while taking into account model
  uncertainty or misspecification on the resulting interval estimation
  or hypothesis testing.
bibliography: bibliography.bib
---

# Introduction

Simulator-based inference is currently at the core of many scientific
fields, such as population genetics, epidemiology or experimental
particle physics.
In many cases, the implicit generative procedure defined in the simulation is
stochastic and/or lacks a tractable probability density
$p(\boldsymbol{x}| \boldsymbol{\theta})$. Given some experimental
observations $\mathcal{D} = \{\boldsymbol{x}_0,...,\boldsymbol{x}_n\}$,
a problem of special relevance for many of these
disciplines is statistical inference on a subset of model parameters
$\boldsymbol{\theta}$, which can be approached via likelihood-free inference
algorithms.

Because the relation between the parameters of the model and the data is
only available via forward simulation, most likelihood-free inference algorithms
tend to be computationally expensive due to the need of repeated simulations
required to cover the parameter space. If data is high-dimensional, likelihood-free
inference rapidly become inefficient, so low-dimensional summary statistics
$\boldsymbol{S}(\mathcal{D})$ are used instead of the raw data
for tractability. The choice of summary statistics is quite important because
will likely lead to a loss of information relevant for statistical inference.


# Problem statement

# Method

# Experiments

# Related Work

See [@Kingma2013-qd; @Louppe2017-br].

# Conclusions

# References
