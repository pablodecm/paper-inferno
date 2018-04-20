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
observations $D = \{\boldsymbol{x}_0,...,\boldsymbol{x}_n\}$,
a problem of special relevance for many of these
disciplines is statistical inference on a subset of model parameters
$\boldsymbol{\theta}$, which can be approached via likelihood-free inference
algorithms such as Approximate Bayesian Computation (ABC), simplified
synthetic likelihoods or density estimation-by-comparison approaches.
<!--- TODO: add references-->

Because the relation between the parameters of the model and the data is
only available via forward simulation, most likelihood-free inference algorithms
tend to be computationally expensive due to the need of repeated simulations
required to cover the parameter space. Furthermore, when data is high-dimensional, likelihood-free
inference can rapidly become inefficient, so low-dimensional summary statistics
$\boldsymbol{s}(D)$ are used instead of the raw data
for tractability. The choice of summary statistics is quite important because
will likely lead to a loss of information relevant for statistical inference.

We can consider data analyses at the Large Hadron Collider (LHC) as a concrete
example.
<!--- TODO: maybe this can go in experiments--->


In this work, we present a new machine-learning method to
learn non-linear sample summary statistics that directly
optimize the expected amount of information about the subset of
parameters of interest using simulated samples.
In addition, the learned
summary statistics can be trivially used to build a synthetic
sample-based likelihood and perform robust and efficient classical or
Bayesian inference from the observed data.


# Problem statement

Let us consider a set of i.i.d. observations $D =
\{\boldsymbol{x}_0,...,\boldsymbol{x}_n\}$ where $\boldsymbol{x} \in \mathcal{X}
\subseteq \mathbb{R}^d$ and a generative model
which implicitly defines a
probability density $p(\boldsymbol{x} | \boldsymbol{\theta})$
parametrized by $\boldsymbol{\theta} \in \mathcal{\Theta} \subseteq
\mathbb{R}^p$ used to model the data. We want to learn a function
$\boldsymbol{s} : \mathcal{D} \subseteq \mathbb{R}^{d\times n} \rightarrow
\mathcal{S} \subseteq \mathbb{R}^{b}$ that computes a summary statistic
of the dataset and reduces its dimensionality so likelihood-free inference
methods can be applied efficiently.
<!---TODO: mention hierarchical model--->
<!---TODO: talk about statistical efficiency--->


# Method
Let us assume we already have or can create on demand a large simulated dataset $G_0=\{(\boldsymbol{x}_0,\boldsymbol{z}_0,
w_0), ..., (\boldsymbol{x}_g,\boldsymbol{z}_g,w_g)\}$ generated
for a certain instantiation of the simulator parameters
$\boldsymbol{\theta}_0$, where $\boldsymbol{z} \in \mathcal{Z}$ are
known latent variables per observation known in the simulation and $w \in \mathcal{W} \subseteq \mathbb{R}$ are
frequency weights, which are commonly produced during the
simulation in many scientific
disciplines.
For applying the method described
in this work, we will need a differentiable transformation
$\boldsymbol{t}_{\boldsymbol{\theta}}:
(\mathcal{X},\mathcal{Z},\mathcal{W})\rightarrow
(\mathcal{X},\mathcal{Z},\mathcal{W})$
that when applied over each
observation of $G_0$ the produced set of observations $G_1$
that approximates a sample of the simulator under a new
parameter instantiation $\boldsymbol{\theta}_1$.


# Related Work

See [@Kingma2013-qd; @Louppe2017-br].

# Experiments


# Conclusions

# References
