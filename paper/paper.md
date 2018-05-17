---
title: |
  INFERNO: Inference-Aware Neural Optimisation
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
  Furthermore, sometimes we are interested on inference drawn over
  a subset of the generative model parameters while taking into account model
  uncertainty or misspecification on the remaining nuisance parameters.
  In this work, we show how non-linear
  summary statistics can be constructed by minimising
  inference-motivated losses via stochastic gradient descent.
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
  \PassOptionsToPackage{sorting=none}{biblatex}
bibliography: bibliography.bib
---

# Introduction

Simulator-based inference is currently at the core of many scientific
fields, such as population genetics, epidemiology or experimental
particle physics.
In many cases, the implicit generative procedure defined in the simulation is
stochastic and/or lacks a tractable probability density
$p(\boldsymbol{x}| \boldsymbol{\theta})$, where
$\boldsymbol{\theta} \in \mathcal{\Theta}$
is the vector of model parameters. Given some experimental
observations $D = \{\boldsymbol{x}_0,...,\boldsymbol{x}_n\}$,
a problem of special relevance for many of these
disciplines is statistical inference on a subset of model parameter
$\boldsymbol{\omega} \in \mathcal{\Omega} \subseteq \mathcal{\Theta}$.
This can be approached via likelihood-free inference
algorithms such as Approximate Bayesian Computation (ABC) [@beaumont2002approximate],
simplified synthetic likelihoods [@wood2010statistical]
or density estimation-by-comparison approaches
[@cranmer2015approximating].

Because the relation between the parameters of the model and the data is
only available via forward simulation, most likelihood-free inference algorithms
tend to be computationally expensive due to the need of repeated simulations
required to cover the parameter space. Furthermore, when data are
high-dimensional, likelihood-free
inference can rapidly become inefficient, so low-dimensional summary statistics
$\boldsymbol{s}(D)$ are used instead of the raw data
for tractability. The choice of summary statistics for such cases becomes
of the utmost importance,
given that naive choices might cause loss of
relevant information and a corresponding degradation of the power
of resulting statistical inference.

As a motivating example, we can consider data analyses at the Large
Hadron Collider (LHC), such as those carried out to establish the
discovery of the Higgs boson [@higgs2012cms; @higgs2012atlas].
In this framework, the ultimate aim is to extract information
about Nature from the large amounts of high-dimensional
data on the subatomic particles produced by energetic collision of protons,
and acquired by highly complex detectors built around the collision
point.
Accurate data modelling is only available via stochastic simulation
of a complicated chain of physical processes, from the underlying
fundament interaction to the subsequent particle interactions with
the detector elements and their readout.
As a result, the density $p(\boldsymbol{x}| \boldsymbol{\theta})$
cannot be analytically computed.

The inference problem in particle physics is commonly posed as hypothesis
testing based on the acquired data. An alternate hypothesis $H_1$ (e.g. a
new theory that predicts that a new fundamental particle) is tested against
a null hypothesis $H_0$ (e.g. an existing theory, which explain previous
observed phenomena). The aim is checking whether the null hypothesis can be rejected
in favour of the alternate hypothesis at a certain confidence level $\alpha$
($\alpha=3\times10^{-7}$ is commonly required
for claiming discovery), also known as Type I error rate. Because $\alpha$ is
fixed, the sensitivity of an analysis is determined by the power $1-\beta$ of
the test, where $\beta$ is the probability of rejecting
a false null hypothesis, also known as Type II error rate.

Due to the high dimensionality of the observed data, a low-dimensional summary
statistic has to be constructed in order to perform inference. A
well-known result of classical statistics,
the Neyman-Pearson lemma[@NeymanPearson1933],
establishes that the likelihood-ratio
$\Lambda(\boldsymbol{x})=p(\boldsymbol{x}| H_0)/p(\boldsymbol{x}| H_1)$ is
the most powerful test  when two simple hypotheses are considered.
 As $p(\boldsymbol{x}| H_0)$ and
$p(\boldsymbol{x}| H_1)$ are not available, simulated samples are used in
practice to obtain an approximation of the likelihood ratio by casting
the problem as supervised learning classification.

In many cases,
the mixture structure of
the generative model allows the treatment of the problem as
signal (S) vs background (B) classification [@adam2015higgs],
effectively estimating
an approximation of $p_{S}(\boldsymbol{x})/p_{B}(\boldsymbol{x})$ which will
vary monotonically with the likelihood ratio.
While the use of classifiers to learn a summary statistic can be effective and
increase the discovery sensitivity, the simulations used to generate
the samples for the classifier often depend on additional
uncertain parameters. These parameters are not of immediate interest but
have to be accounted for in order to make quantitative statements about the
model parameters based on the data. Classification-based summary statistics
cannot easily account for these effects, so the inference power is degraded
when these additional parameters are taken into account.

In this work, we present a new machine learning method to
construct non-linear sample summary statistics that directly
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

Let us consider a set of $n$ i.i.d. observations $D =
\{\boldsymbol{x}_0,...,\boldsymbol{x}_n\}$ where $\boldsymbol{x} \in \mathcal{X}
\subseteq \mathbb{R}^d$ and a generative model
which implicitly defines a
probability density $p(\boldsymbol{x} | \boldsymbol{\theta})$
used to model the data. The generative model is a function of
the vector of parameters $\boldsymbol{\theta} \in \mathcal{\Theta} \subseteq
\mathbb{R}^p$, which includes both interest and nuisance parameters.
We want to learn a function
$\boldsymbol{s} : \mathcal{D} \subseteq \mathbb{R}^{d\times n} \rightarrow
\mathcal{S} \subseteq \mathbb{R}^{b}$ that computes a summary statistic
of the dataset and reduces its dimensionality so likelihood-free inference
methods can be applied effectively. From here onwards, $b$ will be used to
denote the dimensionality of the summary statistic $\boldsymbol{s}(D)$.

While there might be infinite ways to construct $\boldsymbol{s} (D)$, we are
only interested in those summary statistics that are informative
about the subset of interest
$\boldsymbol{\omega} \in \mathcal{\Omega} \subseteq \mathcal{\Theta}$
of the model parameters. The concept of statistical
sufficiency is especially useful to evaluate whether
summary statistics are informative. Classical sufficiency
can be characterised by means of the factorisation criterion:
$$
p(D|\boldsymbol{\omega}) = h(D) g(\boldsymbol{s}(D) | \boldsymbol{\omega} )
$${#eq:sufficiency}
where $h$ and $g$ are non-negative functions. If $p(D | \boldsymbol{\omega})$
can be factorised as indicated, the
summary statistic $\boldsymbol{s}(D)$ will yield the same inference about the parameters of
interest $\boldsymbol{\omega}$ as the full set of observations $D$. For the problems
of interest of this work, the probability density is not explicit so
the general task of finding a sufficient summary statistic cannot be tackled
directly. Hence, alternative methods to build summary statistics have
to be specified.

For simplicity, let us consider a problem where we are only interested on
statistical inference on a
single one-dimensional parameter $\boldsymbol{\omega} = \{ \omega_0\}$ of
the model given some observed data.
Be given a summary statistic $\boldsymbol{s}$ and a statistical procedure
to obtain an unbiased interval estimate of the parameter of interest
which accounts for the effect of nuisance parameters. The resulting interval
can be characterised by its width
$\Delta \omega_0 = \hat{\omega}^{+}_0- \hat{\omega}^{-}_0$. The
expected magnitude of the interval depends
on the summary statistic $\boldsymbol{s}$ chosen: in general,
summary statistics that are
more informative about the parameters of interest will provide narrower
confidence or credible intervals.
Under this figure of merit, the problem
of choosing an optimal summary statistic
can be formally expressed as finding a summary statistic $\boldsymbol{s}^{\ast}$
that minimises the interval width:
$$
\boldsymbol{s}^{\ast} = \textrm{argmin}_{\boldsymbol{s}}  \Delta \omega_0.
$${#eq:general_task}
The above construction can be extended to several parameters of
interest by considering the
interval volume or any other function of the resulting
confidence or credible regions.

# Method

In this section, a machine learning technique to learn non-linear
sample summary statistics is described in detail.
The method is based on minimising the expected variance
of the parameters of interest obtained via a non-parametric
simulation-based synthetic likelihood. A graphical description of the
technique  is depicted on [@Fig:diagram].
The parameters of a neural network are
optimised by stochastic gradient descent within an automatic differentiation
framework, where the considered loss function accounts for the details of
the statistical model as well as the expected effect of nuisance parameters.

![**Learning inference-aware summary statistics.**](diagram.pdf){#fig:diagram}

The family of summary statistics $\boldsymbol{s}(D)$ considered in this
work is composed by a neural network model applied to each dataset
observation $\boldsymbol{f}(\boldsymbol{x}; \boldsymbol{\phi}) :
\mathcal{X} \subseteq \mathbb{R}^{d} \rightarrow
\mathcal{Y} \subseteq \mathbb{R}^{b}$,
whose parameters $\boldsymbol{\phi}$ will be learned during training by means of
stochastic gradient descent, as will be discussed later. Therefore,
using set-builder notation the family of summary statistics considered
can be denoted as:
$$
\boldsymbol{s} (D, \boldsymbol{\phi})
 = \boldsymbol{s} \left ( \: \{ \:  \boldsymbol{f}(\boldsymbol{x}_i; \boldsymbol{\phi}) \:
  | \: \forall \: \boldsymbol{x}_i \in D \: \} \: \right )
$${#eq:summary}
where $\boldsymbol{f}(\boldsymbol{x}_i; \boldsymbol{\phi})$
will reduce the dimensionality from the input observations space
$\mathcal{X}$ to a lower-dimensional space $\mathcal{Y}$.
The next step is to map observation outputs to a
dataset summary statistic, which will in turn be calibrated
and optimised via a non-parametric likelihood
$\mathcal{L}(D; \boldsymbol{\theta},\boldsymbol{\phi})$
created using a set of simulated observations $G_s=
\{\boldsymbol{x}_0,...,\boldsymbol{x}_g\}$, generated at
a certain instantiation of the simulator parameters
$\boldsymbol{\theta}_s$.

In experimental high energy physics experiments, which are the scientific
context that initially motivated this work, histograms of
observation counts are the most
common non-parametric density estimator because the resulting likelihoods
can be expressed as the product of Poisson counts in each of the bins. A naive
sample summary statistic can be built from the output of the neural network
by simply assigning each observation $\boldsymbol{x}$ to a bin corresponding
to the cardinality of the maximum element of
$\boldsymbol{f}(\boldsymbol{x}; \boldsymbol{\phi})$, so each element of the
sample summary will correspond to the following sum:
$$
s_i(D;\boldsymbol{\phi})=\sum_{\boldsymbol{x} \in D}
\begin{cases}
      1 & i = {argmax}_{j=\{0,...,b\}}
        (f_j(\boldsymbol{x}; \boldsymbol{\phi})) \\
      0 & i \neq {argmax}_{j=\{0,...,b\}}
        (f_j(\boldsymbol{x}; \boldsymbol{\phi})) \\
   \end{cases}
$${#eq:argmax}
which can in turn be used to build the following likelihood, where the
expectation for each bin is taken from the simulated sample $G_s$:
$$
\mathcal{L}(D; \boldsymbol{\theta},\boldsymbol{\phi})=\prod_{i=0 }^b
             \textrm{Pois} \left ( s_i (D; \boldsymbol{\phi}) \:  |
             \: \left ( \frac{n}{g} \right ) s_i (G_s;\boldsymbol{\phi}) \right )
$${#eq:likelihood}
where the $n/g$ factor accounts for the different number of
observations in the simulated samples. In cases where the number of
observations is itself a random variable providing information about
the parameters of interest, or where the simulated observation are weighted, the
choice of normalisation of $\mathcal{L}$ may be slightly more involved and
problem specific, but nevertheless amenable.
In the above construction, the chosen
family of summary statistics is non-differentiable due to
the $argmax$ operator, so gradient-based updates for the parameters
cannot be computed. To work around this problem, a differentiable
approximation $\hat{\boldsymbol{s}}(D ; \boldsymbol{\phi})$ is considered.
This function is defined by means of a $softmax$ operator:
$$
\hat{s}_i(D;\boldsymbol{\phi})=\sum_{x \in D}
  \frac{e^{f_i(\boldsymbol{x}; \boldsymbol{\phi})/\tau}}
  {\sum_{j=0}^{b} e^{f_j(\boldsymbol{x}; \boldsymbol{\phi})/\tau}}
$${#eq:soft_summary}
where the temperature hyper-parameter
$\tau$ will regulate the softness of the operator.
In the limit of $\tau \rightarrow 0^{+}$, the probability of the largest
component will tend to 1 while others to 0, and therefore
$\hat{\boldsymbol{s}}(D ; \boldsymbol{\phi})
\rightarrow \boldsymbol{s}(D; \boldsymbol{\phi})$. Similarly, let us
denote by $\hat{\mathcal{L}}(D; \boldsymbol{\theta}, \boldsymbol{\phi})$
the differentiable approximation of the non-parametric likelihood
obtained by substituting $\boldsymbol{s}(D ; \boldsymbol{\phi})$ with
$\hat{\boldsymbol{s}}(D ; \boldsymbol{\phi})$. Instead
of using the observed data $D$, the value of $\hat{\mathcal{L}}$
may be computed
when the observation for each bin is equal its corresponding
the expectation based on
the simulated sample $G_s$, which is commonly denoted as the
Asimov likelihood $\hat{\mathcal{L}}_A$:
$$
\hat{\mathcal{L}}_A(\boldsymbol{\theta}; \boldsymbol{\phi})=\prod_{i=0 }^b
             \textrm{Pois} \left ( \left ( \frac{n}{g} \right )
            \hat{s}_i (G_s;\boldsymbol{\phi}) \:  | \: \left ( \frac{n}{g} \right )
             \hat{s}_i (G_s;\boldsymbol{\phi}) \right )
$${#eq:likelihood_asimov}
for which it can be easily proven that
$argmax_{\boldsymbol{\theta} \in \mathcal{\theta}} (\hat{\mathcal{L}}_A(
\boldsymbol{\theta; \boldsymbol{\phi}})) = \boldsymbol{\theta}_s$ [@cowan2011asymptotic],
so the maximum likelihood estimator (MLE)
for the Asimov likelihood is the parameter vector $\boldsymbol{\theta}_s$
used to generate
the simulated dataset $G_s$. In Bayesian terms, if the prior over the parameters
is flat in the chosen metric,
then $\boldsymbol{\theta}_s$ is also the maximum a posteriori
(MAP) estimator.
By taking the negative logarithm and expanding in
$\boldsymbol{\theta}$ around $\boldsymbol{\theta}_s$, we can obtain
the Fisher information matrix [@fisher_1925] for the
Asimov likelihood:
$$
{\boldsymbol{I}(\boldsymbol{\theta})}_{ij}
= \frac{\partial^2}{\partial {\theta_i} \partial {\theta_j}}
 \left ( - \log \mathcal{\hat{L}}_A(\boldsymbol{\theta};
 \boldsymbol{\phi}) \right )
$${#eq:fisher_info}
which can be computed via automatic differentiation
if the simulation is differentiable and
included in
the computation graph or if the effect
of varying $\boldsymbol{\theta}$ over the simulated
dataset $G_s$ can be effectively approximated. While this
requirement does constrain the applicability of the proposed
technique to a subset of inference
problems, it is quite common for e.g. physical sciences
that the effect of the parameters of interest and the
main nuisance parameters over a sample can be
approximated by the changes of mixture coefficients
of mixture models, translations of a subset of features,
or conditional density ratio re-weighting.

From the Fisher information, if $\hat{\boldsymbol{\theta}}$
is an unbiased estimator of the values of $\boldsymbol{\theta}$,
the covariance matrix using the Cramér-Rao lower bound
[@cramer2016mathematical; @rao1992information]:
$$
\textrm{cov}_{\boldsymbol{\theta}}(\hat{\boldsymbol{\theta}}) \geq
I(\boldsymbol{\theta})^{-1}
$${#eq:CRB}
so the inverse of the Fisher information can be used as an estimator
of the expected variance. If some of the parameters
$\boldsymbol{\theta}$ are constrained by independent measurements
characterised by their likelihoods
$\{\mathcal{L}_C^{0}(\boldsymbol{\theta}), ...,
\mathcal{L}_{C}^{c}(\boldsymbol{\theta})\}$,
those constraints can also be easily included in the covariance
estimation, simply by considering the augmented likelihood
$\hat{\mathcal{L}}_A'$ instead of $\hat{\mathcal{L}}_A$ in [@Eq:fisher_info]:
$$\hat{\mathcal{L}}_A'(\boldsymbol{\theta} ; \boldsymbol{\phi}) =
\hat{\mathcal{L}}_A(\boldsymbol{\theta} ; \boldsymbol{\phi})
\prod_{i=0}^{c}\mathcal{L}_C^i(\boldsymbol{\theta}).$$ {#eq:add_constraint}
In Bayesian
terminology, this approach is referred as the Laplace approximation
[@laplace1986memoir] where the log joint density (including the priors)
is expanded around the MAP to a multi-dimensional normal
approximation of the posterior
density:
$$
p(\boldsymbol{\theta}|D) \approx \textrm{Normal}(
\boldsymbol{\theta} ; \hat{\boldsymbol{\theta}},
I(\hat{\boldsymbol{\theta})}^{-1} )
$${#eq:normal_approx}
which has already been already approached by automatic differentiation in
probabilistic programming frameworks [@tran2016edward]. While a
histogram has been used to construct a Poisson count sample likelihood,
non-parametric density estimation techniques
can be used in its place to construct a
product of observation likelihoods based on the neural network
output  $\boldsymbol{f}(\boldsymbol{x}; \boldsymbol{\phi})$ instead.
For example, an extension of this technique to use
kernel density estimation (KDE) should be straightforward, given its
intrinsic differentiability.

The loss function used for stochastic optimisation of the neural network
parameters $\boldsymbol{\phi}$ can be any function of the inverse
of the Fisher information matrix at $\boldsymbol{\theta}_s$, depending on the
ultimate inference aim. The diagonal
elements $I_{ii}^{-1}(\boldsymbol{\theta}_s)$ correspond to the expected
variance of each of the $\phi_i$ under the normal approximation mentioned
before, so if the aim is efficient inference about one of the parameters
$\omega_0 = \theta_k$ a candidate loss function is:
$$
U = I_{kk}^{-1}(\boldsymbol{\theta}_s)
$$ {#eq:example_loss}
which corresponds to the expected width of the confidence interval
for $\omega_0$ accounting also for the effect of the other nuisance
parameters in $\boldsymbol{\theta}$. This approach can also be extended
when the goal is inference over several parameters of interest
$\boldsymbol{\omega} \subseteq \boldsymbol{\theta}$ (e.g. when
considering a weighted sum of the relevant variances). A simple version
of the approach just described to learn a neural-network based summary statistic
based on an inference-aware loss is summarised in Algorithm
\autoref{alg:simple_algorithm}.

<!-- algorithm -->
\begin{algorithm}[H]
  \caption{Inference-Aware Neural Optimisation.}
  \begin{flushleft}
    {\it Input 1:} differentiable simulator or variational
    approximation $g(\boldsymbol{\theta})$. \\
    {\it Input 2:} initial parameter values $\boldsymbol{\theta}_s.$ \\
    {\it Input 3:} parameter of interest $\omega_0=\theta_k$.
     \\
    {\it Output:} learned summary statistic
      $\boldsymbol{s}(D; \boldsymbol{\phi})$.\\
 \end{flushleft}
 \begin{algorithmic}[1]
 \For{$i=1$ to $N$}
  \State{Sample a representative mini-batch $G_s$ from
  $g(\boldsymbol{\theta}_s)$.}
  \State{Compute differentiable summary statistic
    $\hat{\boldsymbol{s}}(G_s;\boldsymbol{\phi})$.}
  \State{Construct Asimov likelihood
    $\mathcal{L}_A(\boldsymbol{\theta}, \boldsymbol{\phi})$.}
  \State{Get information matrix inverse $I(\boldsymbol{\theta})^{-1}
  = \boldsymbol{H}_{\boldsymbol{\theta}}^{-1}(\log
  \mathcal{L}_A(\boldsymbol{\theta}, \boldsymbol{\phi}))$.}
  \State{Obtain loss
    $U= I_{kk}^{-1}(\boldsymbol{\theta}_s)$.}
  \State{Update network parameters $\boldsymbol{\phi} \rightarrow
  \textrm{SGD}(\nabla_{\boldsymbol{\phi}} U)$.}  
 \EndFor
 \end{algorithmic}
 \label{alg:simple_algorithm}
\end{algorithm}

# Related Work

Classification or regression models have been implicitly used
to construct summary statistics for inference in several
scientific disciplines. For example, in experimental particle physics, the
mixture model structure of the problem makes it amenable to supervised
classification based on simulated datasets
[@hocker2007tmva; @baldi2014searching]. While a classification objective
can be used to learn powerful feature representations and increase
the sensitivity of an analysis, it does not take into account the
details of the inference procedure or the effect of nuisance
parameters like the solution proposed in this work.

The first known effort to include the effect of nuisance parameters
in classification and explain the relation between classification
and the likelihood ratio was by Neal [@neal2008computing]. In the mentioned
work, Neal proposes training of classifier including a function of
nuisance parameter as additional input together with a per-observation
regression model of the expectation value for inference. Cranmer et al.
[@cranmer2015approximating] improved on this concept
by using a parametrised classifier to approximate the
likelihood ratio which is then calibrated to perform statistical inference.
At variance with the mentioned works, we do not consider a classification
objective at all and
the neural network is directly optimised based on an inference-loss.
Additionally, once the summary statistic has been learnt the likelihood can
be trivially constructed and used for classical or Bayesian inference
without a dedicated calibration step. Furthermore, the approach presented
in this work can also be extended as done by Baldi et al.
[@baldi2016parameterized] by a subset of the inference parameters
to obtain a parametrised family of summary statistics with a single model.

Recently, Brehmer et al. [@brehmer2018constraining; @brehmer2018guide] further
extended the approach of parametrised classifiers to better exploit the
latent-space space structure of generative models from particle physics
experiments. Additionally they propose a family of approaches that include
a direct regression of the likelihood ratio
and/or likelihood score in the training losses.
While extremely promising, the most performing solutions are designed for
a subset inference problems at the LHC and require considerable changes
in the way the inference is carried out. The aim of this work is different,
we try to learn sample summary statistics that may act as a plug-in replacement of
classifier-based dimensionality reduction and can be applied to general
likelihood-free problems where the effect of the parameters can be
modelled or approximated.

Within the field of Approximate Bayesian Computation (ABC), there have been
some attempts to use neural network as a dimensionality reduction step to
generate summary statistics. For example, Jiang et al. [@jiang2015learning]
successfully employ a summary statistic by directly regressing the parameters of
interest and therefore approximating the posterior mean given the data, which
then can be used directly as a summary statistic.

A different path is taken by Louppe et al. [@louppe2017learning],
where the authors present a adversarial training procedure to enforce a
pivotal property on a predictive model. The main concern of this
approach is that a classifier which is pivotal with respect
to nuisance parameters might not be optimal, neither for classification
nor for statistical inference. Instead of aiming for being pivotal, the
summary statistics learnt by our algorithm attempts to find a transformation
that directly reduces the expected effect of nuisance parameters
over the parameters of interest.

# Experiments

In this section, we first study the effectiveness of the inference-aware
optimisation in a synthetic problem where the likelihood is known and compare
with the results against classification-based summary statistics.

## 2D Mixture of Gaussians

In order to exemplify the usage of the proposed approach, evaluate its
viability and compare against using a classification model proxy,
a two-dimensional
Gaussian mixture example with two-components is considered. One component
will be
referred as background $b(\boldsymbol{x} | \lambda)$ and the other signal
$s(\boldsymbol{x})$, whose probability densities correspond respectively to:
$$
f_b(\boldsymbol{x} | \lambda) =
\mathcal{N} \left ( (2+\lambda, 0),
  \begin{bmatrix}
    5 & 0 \\
    0 & 9 \\
   \end{bmatrix}
\right)
$${#eq:bkg_toy_pdf}
$$
f_s(\boldsymbol{x}) =
\mathcal{N} \left ( (1,1),
  \begin{bmatrix}
    1 & 0 \\
    0 & 1 \\
   \end{bmatrix}
\right)
$${#eq:sig_toy_pdf}
where $\lambda$, a nuisance parameter that
shifts the mean of the background, is unknown. Hence, the probability density
function of observations as the following the mixture:
$$
p(\boldsymbol{x}| \mu, \lambda) = (1-\mu) f_b(\boldsymbol{x} | \lambda) + \mu f_s(\boldsymbol{x})
$${#eq:mixture_eq}
where $\mu$ is parameter corresponding to the mixture weight
for the signal and consequently $(1-\mu)$ is the mixture weight for the
background. Let us assume that we want to carry out inference based
on $n$ i.i.d. observations, so $\mathbb{E}[n_s]=\mu n$ observations of signal
and $\mathbb{E}[n_b] = (1-\mu)n$ observations of background
are expected respectively. While the mixture model
parametrisation shown in [@Eq:mixture_eq] is correct, the underlying model
could also give information on the expected number of observations as a function
of the model parameters. In this example, we will assume that the underlying model
predicts that number of background and signal observations are Poisson distributed
with means $b$ and $s$, so the following parametrisation will be
more convenient for creating sample likelihoods:
$$
p(\boldsymbol{x}| \nu, \lambda) = \frac{b}{\nu s+b} f_b(\boldsymbol{x} | \lambda) +
 \frac{\nu s}{\nu s+b} f_s(\boldsymbol{x})
$${#eq:mixture_alt}
where $\nu$ is the amount of signal relative to the model expectation. This
parametrisation is common for the most common for physics analyses at the LHC,
because theoretical calculations provide information about the expected number
of observations.

# Conclusions

<!-- ## Acknowledgments {.unnumbered} -->
