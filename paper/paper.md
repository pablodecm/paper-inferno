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

As a motivating example, we can consider data analyses at the Large
Hadron Collider (LHC), like those carried out to establish the
discovery of the Higgs boson.
In this case, the ultimate aim is to extract information
about Nature from the large amounts of high-dimensional
data are acquired by complex detectors setup around the collision points.
Accurate data modelling is only available via stochastic simulation
of the underlying physical processes, particle interactions and detector
readout so $p(\boldsymbol{x}| \boldsymbol{\theta})$ cannot be analytically
computed.

The inference problem in particle physics is commonly posed as hypothesis
testing based on the acquired data. An alternate hypothesis $H_1$ (e.g. a
new theory that predicts that a new fundamental particle) is tested against
a null hypothesis $H_0$ (e.g. existing theory that explain previous
observed phenomena). The aim is checking wether the null can be rejected
in favour of the alternate hypothesis at a certain confidence level $\alpha$,
($\alpha=3\times10^{7}$ is commonly required
for claiming discovery) also known as Type I error rate. Because $\alpha$ is
fixed, the sensibility of an analysis is determined by the power of the test
which corresponds to $1-\beta$, where $\beta$ is the Type II error rate or the
probability of reject a false null hypothesis.

Due to the high-dimensionality of the observed data, a low-dimensional summary
statistic has to be constructed in order to perform inference. A
classical statistical results establishes that the likelihood-ratio
$\Lambda(\boldsymbol{x})=p(\boldsymbol{x}| H_0)/p(\boldsymbol{x}| H_1)$ is
the most powerful test at a fixed confidence level for two simple hypotheses
[@NeymanPearson1933]. While $p(\boldsymbol{x}| H_0)$ and
$p(\boldsymbol{x}| H_1)$ are not available, simulated samples can be used to
obtain an approximation of this ratio by casting the problem as supervised
learning classification. While this approach can be effective and
increase the discovery sensibility, simulations often depend on additional
uncertain parameters that are not of immediate interest but have to be accounted
for. Classification-based summary statistics cannot easily account for this
effects, so the inference power is degraded when these additional parameters
are taken into account.

In this work, we present a new machine-learning method to
learn non-linear sample summary statistics that directly
optimise the expected amount of information about the subset of
parameters of interest using simulated samples, taking into account
the effect of nuisance parameters.
In addition, the learned
summary statistics can be used to build a synthetic
sample-based likelihood and perform robust and efficient classical or
Bayesian inference from the observed data, so they can be readily applied
in place of current classification-based or domain-motivated summary statistics
in current scientific analyses.


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
<!---TODO: talk about statistical sufficiency--->


# Method

In this section a general procedure to learn non-linear
sample summary statistics based on minimising the expected variance of
the parameters of interest obtained via a non-parametric
simulation-based synthetic likelihood is described.

The family of summary statistics $\boldsymbol{s}(D)$ considered in this
work will
be partially composed by a neural network model applied over each dataset
observation $\boldsymbol{f}(\boldsymbol{x}; \boldsymbol{\phi})$
whose parameters $\boldsymbol{\phi}$ will be learned during training.

\[
s_i (D) = \sum_{\boldsymbol{x} \in D }
\]

\[
\mathcal{L}(B; \boldsymbol{\theta},\boldsymbol{\phi})=\prod_{i=0 }^b
             \textrm{Pois}(n_\textrm{c}| \mu \cdot s_\textrm{c} + b_\textrm{c})
\]
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
