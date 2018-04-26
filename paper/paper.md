---
title: Likelihood-free Inference through Sample Summary Statistics Learning
author: |
 Pablo de Castro \
 INFN - Sezione di Padova \
 \texttt{pablo.de.castro@cern.ch}
 \And
 Tommaso Dorigo \
 INFN - Sezione di Padova \
 \texttt{tommaso.dorigo@cern.ch}
numbersections: true
abstract: >-
  Complex computer simulations are commonly required for accurate
  data modelling in many scientific disciplines, making statistical
  inference challenging due to the intractability of the likelihood
  evaluation for the observed data.
  Furthermore, sometimes we are interested only on inference over a subset
  of the generative model parameters while taking into account model
  uncertainty or misspecification on the resulting interval estimation
  or hypothesis testing.
header-includes: |
  \usepackage[nonatbib,preprint]{nips_2018}
  \usepackage{lineno}
  \linenumbers
  \DeclareMathSymbol{\Gamma}{\mathord}{operators}{"00}
  \DeclareMathSymbol{\Delta}{\mathord}{operators}{"01}
  \DeclareMathSymbol{\Theta}{\mathord}{operators}{"02}
  \DeclareMathSymbol{\Lambda}{\mathord}{operators}{"03}
  \DeclareMathSymbol{\Xi}{\mathord}{operators}{"04}
  \DeclareMathSymbol{\Pi}{\mathord}{operators}{"05}
  \DeclareMathSymbol{\Sigma}{\mathord}{operators}{"06}
  \DeclareMathSymbol{\Upsilon}{\mathord}{operators}{"07}
  \DeclareMathSymbol{\Phi}{\mathord}{operators}{"08}
  \DeclareMathSymbol{\Psi}{\mathord}{operators}{"09}
  \DeclareMathSymbol{\Omega}{\mathord}{operators}{"0A}
  \usepackage{algorithm}
  \usepackage{algpseudocode}
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
which corresponds to $1-\beta$, where $\beta$ is the
probability of reject a false null hypothesis, also known as Type II error rate.

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

In this work, we present a new machine learning method to
learn non-linear sample summary statistics that directly
optimise the expected amount of information about the subset of
parameters of interest using simulated samples, taking into account
the effect of nuisance parameters.
In addition, the learned
summary statistics can be used to build a synthetic
sample-based likelihood and perform robust and efficient classical or
Bayesian inference from the observed data, so they can be readily applied
in place of current classification-based or domain-motivated summary statistics
in current scientific data analysis workflows.


# Problem Statement

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

<!---TODO (maybe): mention hierarchical model--->

While there might be infinite ways to construct $\boldsymbol{s} (D)$, we are interested in those summary statistics that are informative
about the subset of interest
$\boldsymbol{\omega} \in \mathcal{\Omega} \subseteq \mathcal{\Theta}$
of the model parameters. The concept of statistical
sufficiency is specially useful to evaluate wether
summary statistics are informative,
which can be characterised by means of the factorisation
criterion:
$$
p(D|\boldsymbol{\omega}) = h(D) g(\boldsymbol{s}(D) | \boldsymbol{\omega} )
$$
where $h$ and $g$ are non-negative functions. In the case of sufficiency, the
summary statistic will yield the same inference about the parameters of
interest $\boldsymbol{\omega}$ than the full set of observations $D$. However,
because the probability density is not even tractable in our problem,
the general task of finding a sufficient summary statistic cannot be tackled
directly, so alternative evaluation metrics have to be specified.

An alternative metric can be specified via a unbiased interval estimation
rule or an approximation of it, which is the path taken in this work.
<!--TODO: expand more on this-->

# Method

In this section a machine learning method to learn non-linear
sample summary statistics based on minimising the expected variance of
the parameters of interest obtained via a non-parametric
simulation-based synthetic likelihood is described.

The family of summary statistics $\boldsymbol{s}(D)$ considered in this
work will composed by a neural network model applied over each dataset
observation $\boldsymbol{f}(\boldsymbol{x}; \boldsymbol{\phi}) :
\mathcal{X} \subseteq \mathbb{R}^{d} \rightarrow
\mathcal{Y} \subseteq \mathbb{R}^{b}$
whose parameters $\boldsymbol{\phi}$ will be learned during training. Therefore,
using set-builder notation the family of summary statistics considered
can be denoted as:
$$
\boldsymbol{s} (D, \boldsymbol{\phi})
 = \boldsymbol{s} \left ( \: \{ \:  \boldsymbol{f}(\boldsymbol{x}_i; \boldsymbol{\phi}) \:
  | \: \forall \: \boldsymbol{x}_i \in D \: \} \: \right )
$$
where $\boldsymbol{f}(\boldsymbol{x}_i; \boldsymbol{\phi})$
reduce will the dimensionality from the input observations space
$\mathcal{X}$ to a lower-dimensional space $\mathcal{Y}$.
The next step is to map observation outputs to a sample
sample summary statistics via a non-parametric likelihood
$\mathcal{L}(D; \boldsymbol{\theta},\boldsymbol{\phi})$
created using a set of simulated observations $G_s=
\{\boldsymbol{x}_0,...,\boldsymbol{x}_g\}$ generated at
a certain instantiation of the simulator parameters
$\boldsymbol{\theta}_s$.

In experimental high energy physics experiments, which is the scientific
context which initially motivated this work, histograms are the most
common non-parametric density estimator because the resulting likelihoods
can be expressed as the product of Poisson counts for each of the bins. A naive
sample summary statistic can be built from the output of the neural network
simply by assigning each observation $\boldsymbol{x}$ to a bin corresponding
to the cardinality of the maximum element of
$\boldsymbol{f}(\boldsymbol{x}; \boldsymbol{\phi})$ so each element of the
sample summary will correspond to the following sum:
$$
s_i(D;\boldsymbol{\phi})=\sum_{x \in D}
\begin{cases}
      1 & i = {argmax}_{j=\{0,...,b\}}
        (f_j(\boldsymbol{x}; \boldsymbol{\phi})) \\
      0 & i \neq {argmax}_{j=\{0,...,b\}}
        (f_j(\boldsymbol{x}; \boldsymbol{\phi})) \\
   \end{cases}
$$
which can in turn be used to build the following likelihood, where the
expectation for each bin is taken from the simulated sample $G_s$:
$$
\mathcal{L}(D; \boldsymbol{\theta},\boldsymbol{\phi})=\prod_{i=0 }^b
             \textrm{Pois}(s_i (D; \boldsymbol{\phi}) \:  | \: \frac{n}{g} \times s_i (G_s;\boldsymbol{\phi}))
$$
where the $n/g$ factor is to account for the different number of
observations in the simulated samples. In cases where the number of
observations is in itself a random variable providing information about
the parameters of interest or the simulated samples are weighted the
choice of normalisation factor can be a bit more involved. The chosen
family of summary statistics is however non-differentiable due to
the $argmax$ operation, so for training a differentiable
approximation is considered $\hat{\boldsymbol{s}}(D; \boldsymbol{\phi})$
by means of a $softmax$ operator:
$$
\hat{s}_i(D;\boldsymbol{\phi})=\sum_{x \in D}
  \frac{e^{f_i(\boldsymbol{x}; \boldsymbol{\phi})/\tau}}
  {\sum_{j=0}^{b} e^{f_j(\boldsymbol{x}; \boldsymbol{\phi})/\tau}}
$$
where the temperature $\tau$ will regulate the softness of the operator.
In the limit of $\tau \rightarrow 0^{+}$, the probability of the highest
component will tend to 1 while others to 0 and therefore
$\hat{\boldsymbol{s}}(D ; \boldsymbol{\phi})
\rightarrow \boldsymbol{s}(D; \boldsymbol{\phi})$. Similarly, let us
denote the differentiable approximation of the non-parametric likelihood
as $\hat{\mathcal{L}}(D; \boldsymbol{\theta}, \boldsymbol{\phi})$. Instead
of using the observed data $D$, the value of $\hat{\mathcal{L}}$
can also be computed
when the observation for each bins is equal to the expectation based on
the simulated sample $G_s$, which will denote as the
Asimov likelihood $\hat{\mathcal{L}}_A$:
$$
\hat{\mathcal{L}}_A(\boldsymbol{\theta}; \boldsymbol{\phi})=\prod_{i=0 }^b
             \textrm{Pois} (\frac{n}{g} \times \hat{s}_i (G_s;\boldsymbol{\phi}) \:  | \: \frac{n}{g} \times \hat{s}_i (G_s;\boldsymbol{\phi}))
$$
for which it can be easily proven that
$argmax_{\boldsymbol{\theta} \in \mathcal{\theta}} (\hat{\mathcal{L}}_A(
\boldsymbol{\theta; \boldsymbol{\phi}})) = \boldsymbol{\theta}_s$, so
the maximum likelihood (or the MAP if priors are flat) for the
Asimov likelihood are the parameter use to generate the simulated
dataset $G_s$. By means taking the minus logarithm and expanding in
$\boldsymbol{\theta}$ around $\boldsymbol{\theta}_s$, we can compute
the Fisher information matrix [@fisher_1925] for the
Asimov likelihood:
$$
{\boldsymbol{I}(\boldsymbol{\theta})}_{ij}
= \frac{\partial^2}{\partial {\theta_i} \partial {\theta_j}} - \log \mathcal{\hat{L}}_A(\boldsymbol{\theta};
 \boldsymbol{\phi})
$$
which can be computed via automatic differentiation
if the simulation is differentiable and included in
the computation graph or alternatively if the effect
of varying $\boldsymbol{\theta}$ over the simulated
dataset $G_s$ can be approximated. While this
requirement does constrain the application of this
technique to a subset of likelihood-free inference
problems, it is quite common in scientific domains
that the effect of the parameters of interest and the
main nuisance parameters over a sample can be
approximated such as the change of mixture coefficients
for mixture models, translations of a subset of features
or conditional density ratio re-weighting.

From the Fisher information, if $\hat{\boldsymbol{\theta}}$
is an unbiased estimator of the values of $\boldsymbol{\theta}$,
the covariance matrix using the Cramér-Rao lower bound
[@cramer2016mathematical; @rao1992information]:
$$
\textrm{cov}_{\boldsymbol{\theta}}(\hat{\boldsymbol{\theta}}) \geq
I(\boldsymbol{\theta})^{-1}
$$
so the inverse of the Fisher information can be used as an estimator
of the expected variance. If some of the parameters
$\boldsymbol{\theta}$ are constrained by independent measurements
characterised by their likelihoods
$\{\mathcal{L_C^0}(\boldsymbol{\theta}), ...,
\mathcal{L_C^c}(\boldsymbol{\theta})\}$,
those constraints can also be easily included in the covariance
estimation simply by considering the product of likelihoods
instead $\mathcal{L}_A'(\boldsymbol{\theta} ; \boldsymbol{\phi}) =
\mathcal{L}_A(\boldsymbol{\theta} ; \boldsymbol{\phi})
\prod_{i=0}^{c}\mathcal{L_C^i}(\boldsymbol{\theta})$. In Bayesian
terminology, this approach is referred as the Laplace approximation
[@laplace1986memoir] where the log joint density (including the priors)
is expanded around the MAP to an normal approximation of the posterior
density:
$$
p(\boldsymbol{\theta}|D) \approx \textrm{Normal}(
\boldsymbol{\theta} ; \hat{\boldsymbol{\theta}},
I(\boldsymbol{\theta})^{-1} )
$$
which has already been already approached by automatic differentiation in
probabilistic programming frameworks [@tran2016edward]. While a
Poisson count likelihood based on a histogram has been used in the
previous derivation, other non-parametric density estimation techniques
can be used to construct a likelihood based the neural network
output  $\boldsymbol{f}(\boldsymbol{x}; \boldsymbol{\phi})$ instead, kernel density estimation (KDE) being specially
promising because it is intrinsically differentiable.

<!-- algorithm -->
\begin{algorithm}[H]
  \caption{Sample Summary Statistics Learning.}
  \begin{flushleft}
    {\it Inputs:} \\
    {\it Outputs:} \\
    {\it Hyper-parameters:}
    \end{flushleft}
 \begin{algorithmic}[1]
 \For{$i=1$ to $n_{steps}$}
 \EndFor
 \end{algorithmic}
\end{algorithm}


# Related Work

See [@Kingma2013-qd; @Louppe2017-br].

# Experiments

# Conclusions

## Acknowledgments {.unnumbered}

# References
