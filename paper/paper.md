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
  Furthermore, sometimes one is interested on inference drawn over
  a subset of the generative model parameters while taking into account model
  uncertainty or misspecification on the remaining nuisance parameters.
  In this work, we show how non-linear
  summary statistics can be constructed by minimising
  inference-motivated losses via stochastic gradient descent such that they
  provide the smallest uncertainty for the parameters of interest. As a
  use case, the problem of confidence interval estimation for
  the mixture coefficient in a multi-dimensional two-component mixture
  model (i.e. signal vs background) is considered, where the proposed technique
  clearly outperforms summary statistics based on probabilistic
  classification,
  a commonly used alternative which does not account for the presence
  of nuisance
  parameters.
bibliography: bibliography.bib
---

# Introduction

Simulator-based inference is currently at the core of many scientific
fields, such as population genetics, epidemiology, and experimental
particle physics.
In these situations the generative procedure implicitly defined
in the simulation may involve a large succession of stochastic steps,
and/or lacks a tractable probability density
$p(\boldsymbol{x}| \boldsymbol{\theta})$, where
$\boldsymbol{\theta}$
is the vector of model parameters. Given some experimental
observations $D = \{\boldsymbol{x}_0,...,\boldsymbol{x}_n\}$,
a problem of special relevance for these
disciplines is statistical inference on a subset of model parameters
$\boldsymbol{\omega}$.
This can be approached via likelihood-free inference
algorithms such as Approximate Bayesian Computation (ABC) [@beaumont2002approximate],
simplified synthetic likelihoods [@wood2010statistical]
or density estimation-by-comparison approaches
[@cranmer2015approximating].

Because the relation between the parameters of the model and the data is
only available via forward simulation, most likelihood-free inference algorithms
tend to be computationally expensive due to the need of repeated simulations
to cover the parameter space. When data are
high-dimensional, likelihood-free
inference can rapidly become inefficient, so low-dimensional summary statistics
$\boldsymbol{t}(D)$ are used instead of the raw data
for tractability. The choice of summary statistics for such cases becomes
critical,
given that naive choices might cause loss of
relevant information and a corresponding degradation of the power
of resulting statistical inference.

As a motivating example we consider data analyses at the Large
Hadron Collider (LHC), such as those carried out to establish the
discovery of the Higgs boson [@higgs2012cms; @higgs2012atlas].
In that framework, the ultimate aim is to extract information
about Nature from the large amounts of high-dimensional
data on the subatomic particles produced by energetic collision of protons,
and acquired by highly complex detectors built around the collision
point.
Accurate data modelling is only available via Monte Carlo simulation
of a complicated chain of physical processes, from the underlying
fundamental interaction to the subsequent particle interactions with
the detector elements and their readout.
As a result, the density $p(\boldsymbol{x}| \boldsymbol{\theta})$
cannot be analytically computed.

Due to the high dimensionality of the observed data, a low-dimensional summary
statistic has to be constructed in order to perform inference. A
well-known result of classical statistics,
the Neyman-Pearson lemma [@NeymanPearson1933],
establishes that the likelihood-ratio
$\Lambda(\boldsymbol{x})=p(\boldsymbol{x}| H_0)/p(\boldsymbol{x}| H_1)$ is
the most powerful test  when two simple hypotheses $H_0$ (null) and
$H_1$ (alternate) are considered.
As $p(\boldsymbol{x}| H_0)$ and
$p(\boldsymbol{x}| H_1)$ are not available, simulated samples are used in
practice to obtain an approximation of the likelihood ratio by casting
the problem as supervised learning classification.
In high-energy physics, a one-dimensional
summary - such as the output of a classifier - is often used for statistical
inference on the parameters of interest through the construction of a
binned likelihood function. Given the already mentioned
intractable form of the generative process
$p(\boldsymbol{x}| \boldsymbol{\theta})$, the resulting
likelihood does not correspond to the generative model but it is
a non-parametric approximation based on the
distribution of the summary statistic for simulated observations.

In many cases,
the nature of the generative model (a mixture of different processes)
allows the treatment of the problem as
signal (s) vs background (b) classification [@adam2015higgs],
when the task becomes one of effectively estimating
an approximation of $t_B(\boldsymbol{x}) = p_{s}(\boldsymbol{x})/(p_{s}(\boldsymbol{x})+
p_{b}(\boldsymbol{x}))$, which will
vary monotonically with the likelihood ratio.
While the use of classifiers to learn a summary statistic can be effective and
increase the discovery sensitivity, the simulations used to generate
the samples which are needed to train the classifier often depend on additional
uncertain parameters (commonly referred as nuisance parameters).
These nuisance parameters are not of immediate interest but
have to be accounted for in order to make quantitative statements about the
model parameters based on the available data.
Classification-based summary statistics
cannot easily account for those effects, so their inference power is degraded
when nuisance parameters are finally taken into account.

In this work, we present a new machine learning method to
construct non-linear sample summary statistics that directly
optimises the expected amount of information about the subset of
parameters of interest using simulated samples, by explicitly
and directly taking into account
the effect of nuisance parameters.
The optimisation procedure is carried out iteratively by stochastic gradient
descent (SDG) [@DBLP:journals/corr/Ruder16] using small subsets
of available simulated data.
In addition, the learned
summary statistics can be used to build synthetic
sample-based likelihoods and perform robust and efficient classical or
Bayesian inference from the observed data, so they can be readily applied
in place of current classification-based or domain-motivated summary statistics
in current scientific data analysis workflows.


# Problem Statement

Let us consider a set of $n$ i.i.d. $d$-dimensional observations $D =
\{\boldsymbol{x}_0,...,\boldsymbol{x}_n\}$ where
$\boldsymbol{x} \in \mathbb{R}^d$, and a generative model
which implicitly defines a
probability density $p(\boldsymbol{x} | \boldsymbol{\theta})$
used to model the data. The generative model is a function of
the $p$-dimensional vector of parameters
$\boldsymbol{\theta} \in \mathbb{R}^p$, which includes both relevant and nuisance parameters.
We want to learn a function
$\boldsymbol{t} : \mathbb{R}^{d\times n} \rightarrow
\mathbb{R}^{m}$ that computes a low-dimensional
summary statistic
of the dataset and greatly reduces its dimensionality (i.e. $m \ll d \times n$),
so likelihood-free inference
methods can be applied effectively.
From here onwards, $m$
will be used to denote the dimensionality of the summary
statistic $\boldsymbol{t}(D)$.


While there might be infinite ways to construct a summary statistic
$\boldsymbol{t} (D)$, we are
only interested in those that are informative
about the subset of interest of the model parameters,
which will be referred as $\boldsymbol{\omega}$.
The concept of statistical
sufficiency is especially useful to evaluate whether
summary statistics are informative. In the absence of nuisance
parameters, classical sufficiency
can be characterised by means of the factorisation
criterion:
$$
p(D|\boldsymbol{\omega}) = q(D) r(\boldsymbol{t}(D) | \boldsymbol{\omega} )
$$ {#eq:sufficiency}
where $q$ and $r$ are non-negative functions. If $p(D | \boldsymbol{\omega})$
can be factorised as indicated, the
summary statistic $\boldsymbol{t}(D)$ will yield the same inference about
the parameters $\boldsymbol{\omega}$ as the full set of observations $D$.
When nuisance parameters have to be accounted in the inference procedure,
alternative notions of sufficiency are commonly used, such as partial
or marginal sufficiency [@basu2011partial;@sprott1975marginal].
Nonetheless, for the problems
of relevance in this work, the probability density is not available in
closed form so
the general task of finding a low-dimensional
sufficient summary statistic cannot be tackled directly. Furthermore,
a low-dimensional summary statistic will not exist in general
for a given problem. The concept of sufficiency is important to understand
why probabilistic classification has been used as proxy task to obtain
summary statistics for mixture-model problems when nuisance parameters
are neglected,
as shown in [appendix @sec:sufficiency]. In this work, we instead propose
an alternative method to build summary statistics.

For simplicity, let us consider a problem where we are only interested on
statistical inference on a
single one-dimensional model parameter $\boldsymbol{\omega} = \{ \omega_0\}$
given some observed data.
Let there be given a summary statistic $\boldsymbol{t}$ and a statistical procedure
to obtain an unbiased interval estimate of the parameter of interest
which accounts for the effect of nuisance parameters. The resulting interval
can be characterised by its width
$\Delta \omega_0 = \hat{\omega}^{+}_0- \hat{\omega}^{-}_0$, defined by some
criterion so as to contain on average, upon repeated sampling,
a given fraction of the probability
density, e.g. a central $68.3\%$ interval. The
expected size of the interval depends
on the summary statistic $\boldsymbol{t}$ chosen: in general,
summary statistics that are
more informative about the parameters of interest will provide narrower
confidence or credible intervals on their value.
Under this figure of merit, the problem
of choosing an optimal summary statistic
can be formally expressed as finding a summary statistic $\boldsymbol{t}^{\ast}$
that minimises the interval width:
$$
\boldsymbol{t}^{\ast} = \textrm{argmin}_{\boldsymbol{t}}  \Delta \omega_0.
$$ {#eq:general_task}
The above construction can be extended to several parameters of
interest by considering the
interval volume or any other function of the resulting
confidence or credible regions.

# Method {#sec:method}

In this section, a machine learning technique to learn non-linear
sample summary statistics is described in detail.
The method seeks to minimise the expected variance
of the parameters of interest obtained via a non-parametric
simulation-based synthetic likelihood. A graphical description of the
technique  is depicted on [@Fig:diagram].
The parameters of a neural network are
optimised by stochastic gradient descent within an automatic differentiation
framework, where the considered loss function accounts for the details of
the statistical model as well as the expected effect of nuisance parameters.

![Learning inference-aware summary statistics (see text for
  details).](gfx/figure1.pdf){#fig:diagram}

The family of summary statistics $\boldsymbol{t}(D)$ considered in this
work is composed by a neural network model applied to each dataset
observation $\boldsymbol{h}(\boldsymbol{x}; \boldsymbol{\phi}) : \mathbb{R}^{d} \rightarrow
\mathbb{R}^{m}$,
whose parameters $\boldsymbol{\phi}$ will be learned during training by means of
stochastic gradient descent, as will be discussed later. Therefore,
using set-builder notation the family of summary statistics considered
can be denoted as:
$$
\boldsymbol{t} (D, \boldsymbol{\phi})
 = \boldsymbol{t} \left ( \: \{ \:  \boldsymbol{h}(\boldsymbol{x}_i; \boldsymbol{\phi}) \:
  | \: \forall \: \boldsymbol{x}_i \in D \: \} \: \right )
$$ {#eq:summary}
where the neural network $\boldsymbol{h}(\boldsymbol{x}_i; \boldsymbol{\phi})$
will reduce the dimensionality from the $d$-dimensional inputs to
the $m$-dimensional outputs and will effectively define the summary
statistic transformation.
The next step is to map observation outputs to a
dataset summary statistic, which will in turn be calibrated
and optimised via a non-parametric likelihood
$\mathcal{L}(D; \boldsymbol{\theta},\boldsymbol{\phi})$
created using a set of $l$ simulated observations $G_\textrm{MC}=
\{\boldsymbol{x}_0,...,\boldsymbol{x}_{l}\}$, generated at
a certain instantiation of the simulator parameters
$\boldsymbol{\theta}_\textrm{MC}$.

In experimental high energy physics experiments, which are the scientific
context that initially motivated this work, histograms of
observation counts are the most
commonly used non-parametric density estimator because the
resulting likelihoods
can be expressed as the product of Poisson factors, one for each of
the considered bins. A naive
sample summary statistic can be built from the output of the neural network
by simply assigning each observation $\boldsymbol{x}$ to a bin corresponding
to the cardinality of the maximum element of
$\boldsymbol{h}(\boldsymbol{x}; \boldsymbol{\phi})$, so each element of the
sample summary will correspond to the following sum:
$$
t_i(D;\boldsymbol{\phi})=\sum_{\boldsymbol{x} \in D}
\begin{cases}
      1 & i = {argmax}_{j=\{0,...,b\}}
        (f_j(\boldsymbol{x}; \boldsymbol{\phi})) \\
      0 & i \neq {argmax}_{j=\{0,...,b\}}
        (f_j(\boldsymbol{x}; \boldsymbol{\phi})) \\
   \end{cases}
$$ {#eq:argmax}
which can in turn be used to build the following likelihood, where the
expectation for each bin is taken from the simulated sample $G_\textrm{MC}$:
$$
\mathcal{L}(D; \boldsymbol{\theta},\boldsymbol{\phi})=\prod_{i=0 }^m
             \textrm{Pois} \left ( t_i (D; \boldsymbol{\phi}) \:  |
             \: \left ( \frac{n}{l} \right ) t_i (G_\textrm{MC};\boldsymbol{\phi}) \right )
$$ {#eq:likelihood}
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
approximation $\hat{\boldsymbol{t}}(D ; \boldsymbol{\phi})$ is considered.
This function is defined by means of a $softmax$ operator:
$$
\hat{t}_i(D;\boldsymbol{\phi})=\sum_{x \in D}
  \frac{e^{f_i(\boldsymbol{x}; \boldsymbol{\phi})/\tau}}
  {\sum_{j=0}^{m} e^{f_j(\boldsymbol{x}; \boldsymbol{\phi})/\tau}}
$$ {#eq:soft_summary}
where the temperature hyper-parameter
$\tau$ will regulate the softness of the operator.
In the limit of $\tau \rightarrow 0^{+}$, the probability of the largest
component will tend to 1 while others to 0, and therefore
$\hat{\boldsymbol{t}}(D ; \boldsymbol{\phi})
\rightarrow \boldsymbol{t}(D; \boldsymbol{\phi})$. Similarly, let us
denote by $\hat{\mathcal{L}}(D; \boldsymbol{\theta}, \boldsymbol{\phi})$
the differentiable approximation of the non-parametric likelihood
obtained by substituting $\boldsymbol{t}(D ; \boldsymbol{\phi})$ with
$\hat{\boldsymbol{t}}(D ; \boldsymbol{\phi})$. Instead
of using the observed data $D$, the value of $\hat{\mathcal{L}}$
may be computed
when the observation for each bin is equal to its corresponding
expectation based on
the simulated sample $G_\textrm{MC}$, which is commonly denoted as the
Asimov likelihood [@cowan2011asymptotic] $\hat{\mathcal{L}}_A$:
$$
\hat{\mathcal{L}}_A(\boldsymbol{\theta}; \boldsymbol{\phi})=\prod_{i=0 }^m
             \textrm{Pois} \left ( \left ( \frac{n}{l} \right )
            \hat{t}_i (G_\textrm{MC};\boldsymbol{\phi}) \:  | \: \left ( \frac{n}{l} \right )
             \hat{t}_i (G_\textrm{MC};\boldsymbol{\phi}) \right )
$$ {#eq:likelihood_asimov}
for which it can be easily proven that
$argmax_{\boldsymbol{\theta} \in \mathcal{\theta}} (\hat{\mathcal{L}}_A(
\boldsymbol{\theta; \boldsymbol{\phi}})) = \boldsymbol{\theta}_\textrm{MC}$,
so the maximum likelihood estimator (MLE)
for the Asimov likelihood is the parameter vector $\boldsymbol{\theta}_\textrm{MC}$
used to generate
the simulated dataset $G_\textrm{MC}$. In Bayesian terms, if the prior over the parameters
is flat in the chosen metric,
then $\boldsymbol{\theta}_\textrm{MC}$ is also the maximum a posteriori
(MAP) estimator.
By taking the negative logarithm and expanding in
$\boldsymbol{\theta}$ around $\boldsymbol{\theta}_\textrm{MC}$, we can obtain
the Fisher information matrix [@fisher_1925] for the
Asimov likelihood:
$$
{I(\boldsymbol{\theta})}_{ij}
= \mathop{\mathbb{E}} \left [
\frac{\partial^2}{\partial {\theta_i} \partial {\theta_j}}
 \left ( - \log \mathcal{\hat{L}}_A(\boldsymbol{\theta};
 \boldsymbol{\phi}) \right ) \right ]
$$ {#eq:fisher_info}
which can be computed via automatic differentiation
if the simulation function $g(\boldsymbol{\theta}_\textrm{MC})$ is
differentiable or if the effect
of varying $\boldsymbol{\theta}$ over the simulated
dataset $G_\textrm{MC}$ can be effectively approximated. While this
requirement does constrain the applicability of the proposed
technique to a subset of likelihood-free inference
problems, it is quite common for e.g. physical sciences
that the effect of the parameters of interest and the
main nuisance parameters over a sample can be
approximated by the changes of mixture coefficients
of mixture models, translations of a subset of features,
or conditional density ratio re-weighting.

If $\hat{\boldsymbol{\theta}}$
is an unbiased estimator of the values of $\boldsymbol{\theta}$,
the covariance matrix fulfils the Cramér-Rao lower bound
[@cramer2016mathematical; @rao1992information]:
$$
\textrm{cov}_{\boldsymbol{\theta}}(\hat{\boldsymbol{\theta}}) \geq
I(\boldsymbol{\theta})^{-1}
$$ {#eq:CRB}
and the inverse of the Fisher information can be used as an
approximate estimator of the expected variance, given that
the bound would become
an equality in the asymptotic limit for MLE.
If some of the parameters
$\boldsymbol{\theta}$ are constrained by independent measurements
characterised by their likelihoods
$\{\mathcal{L}_C^{0}(\boldsymbol{\theta}), ...,
\mathcal{L}_{C}^{c}(\boldsymbol{\theta})\}$,
those constraints can also be easily included in the covariance
estimation, simply by considering the augmented likelihood
$\hat{\mathcal{L}}_A'$ instead of $\hat{\mathcal{L}}_A$ in [@Eq:fisher_info]:
$$\hat{\mathcal{L}}_A'(\boldsymbol{\theta} ; \boldsymbol{\phi}) =
\hat{\mathcal{L}}_A(\boldsymbol{\theta} ; \boldsymbol{\phi})
\prod_{i=0}^{c}\mathcal{L}_C^i(\boldsymbol{\theta}).
$$ {#eq:add_constraint}
In Bayesian
terminology, this approach is referred to as the Laplace approximation
[@laplace1986memoir] where the logarithm of the joint density (including the priors)
is expanded around the MAP to a multi-dimensional normal
approximation of the posterior
density:
$$
p(\boldsymbol{\theta}|D) \approx \textrm{Normal}(
\boldsymbol{\theta} ; \hat{\boldsymbol{\theta}},
I(\hat{\boldsymbol{\theta})}^{-1} )
$$ {#eq:normal_approx}
which has already been approached by automatic differentiation in
probabilistic programming frameworks [@tran2016edward]. While a
histogram has been used to construct a Poisson count sample likelihood,
non-parametric density estimation techniques
can be used in its place to construct a
product of observation likelihoods based on the neural network
output  $\boldsymbol{h}(\boldsymbol{x}; \boldsymbol{\phi})$ instead.
For example, an extension of this technique to use
kernel density estimation (KDE) should be straightforward, given its
intrinsic differentiability.

The loss function used for stochastic optimisation of the neural network
parameters $\boldsymbol{\phi}$ can be any function of the inverse
of the Fisher information matrix at $\boldsymbol{\theta}_\textrm{MC}$, depending on the
ultimate inference aim. The diagonal
elements $I_{ii}^{-1}(\boldsymbol{\theta}_\textrm{MC})$ correspond to the expected
variance of each of the $\phi_i$ under the normal approximation mentioned
before, so if the aim is efficient inference about one of the parameters
$\omega_0 = \theta_k$ a candidate loss function is:
$$
U = I_{kk}^{-1}(\boldsymbol{\theta}_\textrm{MC})
$$ {#eq:example_loss}
which corresponds to the expected width of the confidence interval
for $\omega_0$ accounting also for the effect of the other nuisance
parameters in $\boldsymbol{\theta}$. This approach can also be extended
when the goal is inference over several parameters of interest
$\boldsymbol{\omega} \subseteq \boldsymbol{\theta}$, e.g.
considering a weighted sum of the relevant variances
or using the sub-determinant
of the covariance matrix for the parameters of interest:
$$
\tilde{U} = |\tilde{\Sigma}|
$$ {#eq:multivariate_case}
where $\tilde{\Sigma}$ is the determinant of the covariance
matrix $\Sigma=I(\boldsymbol{\theta})^{-1}$ restricted to the
elements corresponding to the set of parameters of interest.
A simple version
of the approach just described to learn a neural-network based summary statistic
employing an inference-aware loss is summarised in Algorithm
\ref{alg:simple_algorithm}. It is worth highlighting that the purpose of the
algorithm is to use simulated examples to directly optimise the expected
uncertainty on parameters of interest by constructing an appropriate summary
statistic, in view of its later use on real data.
In the current implementation the parameters of the model
($\boldsymbol{\theta}_\textrm{MC}$
in the algorithm description below) are kept fixed at some assumed initial
values, and only their differential variation is accounted for in the
training process.

<!-- algorithm -->
\begin{algorithm}[H]
  \caption{Inference-Aware Neural Optimisation.}
  \begin{flushleft}
    {\it Input 1:} differentiable simulator or variational
    approximation $g(\boldsymbol{\theta})$. \\
    {\it Input 2:} initial parameter values $\boldsymbol{\theta}_\textrm{MC}.$ \\
    {\it Input 3:} parameter of interest $\omega_0=\theta_k$.
     \\
    {\it Output:} learned summary statistic
      $\boldsymbol{t}(D; \boldsymbol{\phi})$.\\
 \end{flushleft}
 \begin{algorithmic}[1]
 \For{$i=1$ to $N$ (number of SGD iterations)}
  \State{Sample a representative mini-batch $G_\textrm{MC}$ from
  $g(\boldsymbol{\theta}_\textrm{MC})$.}
  \State{Compute differentiable summary statistic
    $\hat{\boldsymbol{t}}(G_\textrm{MC};\boldsymbol{\phi})$.}
  \State{Construct Asimov likelihood
    $\mathcal{L}_A(\boldsymbol{\theta}, \boldsymbol{\phi})$.}
  \State{Get information matrix inverse $I(\boldsymbol{\theta})^{-1}
  = \boldsymbol{H}_{\boldsymbol{\theta}}^{-1}(\log
  \mathcal{L}_A(\boldsymbol{\theta}, \boldsymbol{\phi}))$.}
  \State{Obtain loss
    $U= I_{kk}^{-1}(\boldsymbol{\theta}_\textrm{MC})$.}
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
the neural network is directly optimised based on an inference-aware loss.
Additionally, once the summary statistic has been learnt the likelihood can
be trivially constructed and used for classical or Bayesian inference
without a dedicated calibration step. Furthermore, the approach presented
in this work can also be extended, similarly to what was
done by Baldi et al.
[@baldi2016parameterized] considering a subset of the inference parameters
to obtain a parametrised family of summary statistics with a single model.

Recently, Brehmer et
al. [@Brehmer:2018hga; @brehmer2018constraining; @brehmer2018guide] further
extended the approach of parametrised classifiers to better exploit the
latent-space space structure of generative models from complex scientific
simulators. Additionally they proposed a family of approaches that include
a direct regression of the likelihood ratio
and/or likelihood score in the training losses.
While extremely promising, the best performing solutions are designed for
a subset of the inference problems at the LHC and they require considerable changes
in the way the inference is carried out. The aim of this work is different,
as we try to learn sample summary statistics that may act as a
plug-in replacement of
classifier-based dimensionality reduction and can be applied to general
likelihood-free problems where the effect of the parameters can be
modelled or approximated.

Within the field of Approximate Bayesian Computation (ABC), there have been
some attempts to use neural network as a dimensionality reduction step to
generate summary statistics. For example, Jiang et al. [@jiang2015learning]
successfully employ a summary statistic by directly regressing the parameters of
interest and therefore approximating the posterior mean given the data, which
then can be used directly as a summary statistic. With the purpose
of improving ABC-based inference for astrophysical observations,
Charnock et al. [@charnock2018automatic] have developed information maximising neural
networks (IMNNs) whose goal is also to learn non-linear transformations
by directly optimising a function of the Fisher information. While IMNNs
are the closest approach found in the literature to the one
presented in this work, our approach is instead focussed on directly optimising
the expected uncertainty on the parameter of interest and addressing directly
the problem of nuisance parameters. Other differences between IMNNs
and the way the Fisher information is approximated and the
consideration a Poisson binned likelihood instead of a
Gaussian approximation, which is motivated by the the differences between
the inference problems in astrophysical observations with those
within experimental particle physics.

A different path is taken by Louppe et al. [@louppe2017learning],
where the authors present a adversarial training procedure to enforce a
pivotal property on a predictive model. The main concern with this
approach is that a classifier which is pivotal with respect
to nuisance parameters might not be optimal, neither for classification
nor for statistical inference. Instead of aiming for being pivotal, the
summary statistics learnt by our algorithm attempt to find a transformation
that directly reduces the expected effect of nuisance parameters
over the parameters of interest.

Within High Energy Physics, there has been an effort
to develop techniques that can learn directly from data instead of using
simulated observations. In the simplest case, the are based in using
data from a control region for modelling the background component
in a supervised classification setting, but there have been also
recent efforts exploiting weakly-supervised
[@Dery:2017fap; @Metodiev:2017vrx; @Komiske:2018oaa; @Cohen:2017exh] or
even fully unsupervised attempts
[@DeSimone:2018efk; @Cerri:2018anq; @Hajer:2018kqm] to learn from directly from data.
In our view, all
these data-driven approaches can be useful to obtain good summaries
when the simulated observations do not provide adequate modelling
but cannot circumvent the problem of nuisance parameters.
A statistical model has to be built for making quantitative statements
about the model parameters given that data, in which nuisance parameters
represent the modelling limitations, independently of the method used to
construct the summary statistic.

The limitation of the previously mentioned data-driven approaches
is better shown using
an example: let us suppose that we were able to use a weakly-supervised
technique to obtain a classifier that approximates the quantity
$t_B(\boldsymbol{x} | \boldsymbol{\theta}_t) = p_{s}(\boldsymbol{x} | \boldsymbol{\theta}_t )/(p_{s}(\boldsymbol{x}| \boldsymbol{\theta}_t )+ p_{b}(\boldsymbol{x}| \boldsymbol{\theta}_t))$,
where $\boldsymbol{\theta}_t$ are the true value of the nuisance parameters.
In order to use this summary for inference, a
non-parametric likelihood based on simulated observations (or data from
a control region) has to be built, and the effect of nuisance parameter
has to be accounted for in the calibration of the statistical model even
if the transformation is able to obtain the optimal classifier
$t_B(\boldsymbol{x} | \boldsymbol{\theta}_t)$. Similar concerns apply to
summary statistics obtained using unsupervised techniques
or anomaly detection algorithms if they are to be used for statistical
inference of model parameters.





# Experiments {#sec:d-synthetic-mixture}

In this section, we first study the effectiveness of the inference-aware
optimisation in a synthetic mixture problem where the likelihood is known. We then
compare our results with those obtained by standard classification-based
summary statistics. All the code needed to reproduce the results
presented the results presented here is available in
an online repository [@code_repository], extensively using \textsc{TensorFlow}
[@tensorflow2015-whitepaper]
and \textsc{TensorFlow Probability} [@tran2016edward;@dillon2017tensorflow] software libraries.

In order to demonstrate the usage of the proposed approach, evaluate its
viability and test its performance by comparing to the use of
a classification model proxy, a three-dimensional
mixture example with two components is considered.
One component will be referred as background $f_b(\boldsymbol{x} | \lambda)$ and
the other as signal $f_s(\boldsymbol{x})$; their probability density functions
are taken to correspond respectively to:
$$
f_b(\boldsymbol{x} | r, \lambda) =
\mathcal{N} \left (
  (x_0, x_1) \, \middle | \,
  (2+r, 0),
  \begin{bmatrix}
    5 & 0 \\
    0 & 9 \\
   \end{bmatrix}
\right)
Exp (x_2 | \lambda)
$$ {#eq:bkg_toy_pdf}
$$
f_s(\boldsymbol{x}) =
\mathcal{N} \left (
  (x_0, x_1) \, \middle | \,
  (1,1),
  \begin{bmatrix}
    1 & 0 \\
    0 & 1 \\
   \end{bmatrix}
\right)
Exp (x_2 | 2)
$$ {#eq:sig_toy_pdf}
so that $(x_0,x_1)$ are distributed according to a multivariate normal
distribution while $x_2$ follows an independent exponential distribution
both for background and signal, as shown in [@Fig:subfigure_a].
The signal distribution is fully specified while
the background distribution depends on $r$, a parameter which
shifts the mean of the background density, and a parameter $\lambda$ which
specifies
the exponential rate in the third dimension. The $r$ and $\lambda$
parameters will be the
treated as nuisance parameters when benchmarking different methods.
The functional form and parameter values of the problem chosen
as a basis of the benchmarks have been set to arbitrarily values.
Hence, the probability density
function of observations has the following mixture structure:
$$
p(\boldsymbol{x}| \mu, r, \lambda) = (1-\mu) f_b(\boldsymbol{x} | r, \lambda) 
                                      + \mu f_s(\boldsymbol{x})
$$ {#eq:mixture_eq}
where $\mu$ is the parameter corresponding to the mixture weight
for the signal and consequently $(1-\mu)$ is the mixture weight for the
background. The low-dimensional projections from samples from
the mixture distribution for a small $\mu=50/1050$ is shown in
[@Fig:subfigure_b].

::: {#fig:subfigs_distributions .subfigures}
![signal (red) and background (blue)
 ](gfx/figure2a.pdf){#fig:subfigure_a width=49%}
![mixture distribution (black)
 ](gfx/figure2b.pdf){#fig:subfigure_b width=49%}

Projection in 1D and 2D dimensions of 50,000 samples from
the synthetic problem considered. The background distribution
nuisance parameters used for generating data correspond to
$r=0$ and $\lambda=3$. For samples the mixture distribution,
$s=50$ and $b=1000$ were used, hence the mixture coefficient is $\mu=50/1050$.
:::

Let us assume that we want to carry out inference based
on $n$ i.i.d. observations, such that $\mathbb{E}[n_s]=\mu n$ observations
of signal
and $\mathbb{E}[n_b] = (1-\mu)n$ observations of background
are expected, respectively.
While the mixture model
parametrisation shown in [@Eq:mixture_eq] is correct as is, the underlying model
could also give information on the expected number of observations as a function
of the model parameters.
In this toy problem, we consider a case where the underlying model
predicts that the total number of observations are Poisson distributed with
a mean $s+b$, where $s$ and $b$ are the expected number of signal
and background observations. Thus the following parametrisation will be
more convenient for building sample-based likelihoods:
$$
p(\boldsymbol{x}| s, r, \lambda, b) = \frac{b}{ s+b}
 f_b(\boldsymbol{x} | r, \lambda) +
 \frac{s}{s+b} f_s(\boldsymbol{x}).
$$ {#eq:mixture_alt}
This parametrisation is common for physics analyses at the LHC,
because theoretical calculations provide information about the expected number
of observations. If the probability density is known, but the expectation for
the number of observed events depends on the model parameters, the likelihood
can be extended [@barlow1990extended] with a Poisson count term as:
$$
\mathcal{L}(s, r, \lambda, b) = \textrm{Pois}(n | s+b) \prod^{n}
p(\boldsymbol{x}| s,r, \lambda, b)
$$ {#eq:ext_ll}
which will be used to provide an optimal inference baseline when benchmarking
the different approaches. Another quantity of relevance is the conditional
density ratio, which would correspond to the optimal classifier (in the
Bayes risk sense) separating
signal and background events in a balanced dataset (equal priors):
$$
t_B(\boldsymbol{x} | r, \lambda) =
\frac{f_s(\boldsymbol{x})}{
f_s(\boldsymbol{x}) + f_b(\boldsymbol{x} | r, \lambda) }
$$ {#eq:opt_clf}
noting that this quantity depends on the parameters that define the background
distribution $r$ and $\lambda$, but not on $s$  or $b$ that are a function of
the mixture coefficients. It can be proven (see [appendix @sec:sufficiency] )
that $t_B(\boldsymbol{x} | r, \lambda)$ is a sufficient summary statistic with
respect to an
arbitrary two-component mixture model if the only unknown parameter
is the signal mixture fraction $\mu$ (or alternatively $s$ in the
parametrisation of [@Eq:mixture_alt]).
In practice, the probability density functions of signal and
background are not known analytically, and only forward samples are available
through simulation, so alternative approaches are required.

While the synthetic nature of this example allows one to rapidly generate
training data
on demand,
a training dataset of 200,000 simulated observations has been considered,
in order
to study how the proposed method performs when training data is limited.
Half of the simulated
observations correspond to the signal component and half to the background
component. The latter has been generated using $r=0.0$ and
$\lambda=3.0$.
A validation holdout from the training dataset of 200,000
observations is only used for computing relevant metrics during
training and
to control over-fitting. The final figures of merit that allow to
compare different approaches are computed
using a larger dataset of 1,000,000 observations.
For simplicity, mini-batches for each training step are balanced so the same
number of events from each component is taken both when using
standard classification or inference-aware losses.

Another option would be to pose the problem as one of classification
based on a simulated dataset. A supervised machine learning model such a
neural network can be trained to discriminate signal and background
observations, considering parameters $r$ and $\lambda$ fixed.
The output of such a model typically consists in class probabilities
$c_s$ and $c_b$ given an observation $\boldsymbol{x}$, which will tend
asymptotically to the optimal classifier from [@Eq:opt_clf] given
enough data, a flexible enough model and a powerful learning rule.
The conditional class probabilities (or alternatively the likelihood ratio 
$f_s(\boldsymbol{x})/f_b(\boldsymbol{x})$) are powerful
learned features that can be used as summary statistics; however
their construction
ignores the effect of the nuisance parameters $r$ and $\lambda$ on the
background distribution. Furthermore, some kind  of non-parametric density estimation
(e.g. a histogram) has to be considered in order to build a calibrated statistical
model using the classification-based learned features, which  will in turn
smooth and reduce the information available for inference.

To demonstrate the use of this family of classification-based summary statistics,
a histogram of a deep neural network classifier output trained on simulated data
and its variation computed for different values of $r$ and $\lambda$
are shown in [@Fig:train_clf]. The details of the training procedure
will be provided later in this document. The classifier output can be directly
compared with $t_B(\boldsymbol{x} | r = 0.0, \lambda = 3.0)$ evaluated using the
analytical distribution function of signal and background 
according to [@Eq:opt_clf],
which is shown in [@Fig:opt_clf] and corresponds to the optimal classifier. The
trained classifier approximates very well the optimal classifier. The
summary statistic distribution for the background component
shows a marked dependence on the value
of the nuisance parameters both for the trained and the optimal classifier,
which will
in turn cause an important degradation on the subsequent statistical inference.

::: {#fig:subfigs_clf_hists .subfigures}
![classifier trained on simulated samples
   ](gfx/figure3a.pdf){#fig:train_clf width=48%}
![optimal classifier $s(\boldsymbol{x} | r = 0.0, \lambda = 3.0)$
 ](gfx/figure3b.pdf){#fig:opt_clf width=48%}

Histograms of the classifier-based
summary statistics for signal and background (top) and
variation for different values of nuisance parameters compared
with the expected signal relative to the nominal background magnitude (bottom).
The
classifier was trained using signal and background samples generated
for $r = 0.0$ and $\lambda = 3.0$.
:::

The statistical model described above has up to four unknown parameters: the
expected number of signal observations $s$,
the background mean shift $r$,
the background exponential rate in the third dimension $\lambda$, and
the expected number of background observations. The effect of the
expected number of signal and background observations $s$ and $b$
can be easily included in the computation graph by 
weighting the signal and background observations.
This is equivalent to scaling the resulting vector of Poisson counts (or its
differentiable approximation)
if a non-parametric counting model as the one described in [@Sec:method] is used.
Instead the effect of $r$ and $\lambda$, both nuisance parameters
that will define the background distribution, is more easily modelled
as a transformation of the input data $\boldsymbol{x}$. In particular,
$r$ is a nuisance parameter that causes a shift on the background
along the first dimension  and its effect can accounted for
in a differentiable manner by simply adding $(r,0.0,0.0)$ to
each observation
in the mini-batch generated from the background distribution.
Similarly, the effect of $\lambda$ can be modelled by multiplying
$x_2$ by the ratio between the $\lambda_0$ used for generation and the
one being modelled.
These transformations are specific for this example,
but alternative transformations depending on
parameters could also be accounted for as long as they are
differentiable or substituted by a differentiable approximation.

For this problem, we are interested in carrying out statistical
inference on the parameter of interest $s$. In fact,
the performance of inference-aware optimisation as described in
[@Sec:method] will be compared with classification-based summary statistics for
a series of inference benchmarks based on the synthetic problem described above
that vary in the number of nuisance parameters considered and their constraints,
as shown in \autoref{tab:benchmark_table}. For Benchmark 0 no nuisance
parameters are considered, thus the inference problem is that of
determining the
number of signal events $s$ given some data and a perfectly known signal
and background model, so the classification approach is expected
to provide sufficient summary statistics. The rest of the benchmarks
correspond to the presence of nuisance parameters,
differing among them in their number and constrains. The main figure of merit
used to compared the different techniques will be the expected
uncertainty in the parameter of interest $s$ for the inference problem
defined for each benchmark and conditioned
on the values for the true value of the parameters of $s=50$, $r=0.0$,
$\lambda=3.0$ and $b=1000$.

\begin{table}
  \caption{Definition of the different statistical
  inference benchmark problems that will be
  considered when comparing different techniques to obtain
  summary statistics. The number of signal events $s$ will be the parameter
  of interest in all cases, where each benchmark differs on the nuisance
  parameters considered and their constraints.}
  \label{tab:benchmark_table}
  \centering
  \footnotesize
  \input{table_syst.tex}
\end{table}

When using classification-based summary statistics, the construction of
a summary statistic does depend on the presence of nuisance parameters, so the same
model is trained independently of the benchmark considered. In real-world
inference scenarios, nuisance parameters have often to be accounted for and
typically are constrained by prior information or auxiliary measurements.
For the approach
presented in this work, inference-aware neural optimisation, the effect of the
nuisance parameters and their constraints can be taken into account during training.
Hence, 5 different training procedures for \textsc{INFERNO} will be considered,
one for each of the benchmarks, denoted by the corresponding benchmark number. 

The same basic network architecture is used both for cross-entropy and
inference-aware training: two hidden layers of 100 nodes followed by
rectified linear unit (ReLU) [@Goodfellow-et-al-2016]
activations. Because we are using the multi-class formulation of
the cross-entropy loss $L_\textrm{CE}=\sum_i y_i \log \hat{y}_i$,
the number of nodes on the output
layer is two when
classification proxies are used, matching the number of mixture classes
in the problem considered. Instead, for inference-aware classification
the number of output nodes can be arbitrary and will be denoted with $m$,
corresponding to the dimensionality of the sample summary statistics. For
the experiments shown in this work using the INFERNO technique,
an output size of $m=10$ has been used.
The final layer is followed by a softmax activation function and
a temperature $\tau = 0.1$ for inference-aware learning to ensure
that the differentiable approximations are close to the true values
(see approximation from [@Eq:soft_summary]). Standard
mini-batch stochastic gradient descent (SGD) is used for training and
the optimal learning rate is fixed and decided by means of a
simple scan; the best choice found is specified together with the results.

::: {#fig:subfigs_training .subfigures}
![inference-aware training loss
](gfx/figure4a.pdf){#fig:training_dynamics width=48%}
![profile-likelihood comparison
](gfx/figure4b.pdf){#fig:profile_likelihood width=48%}
 
Dynamics and results of inference-aware optimisation: (a) square root of
inference-loss (i.e. approximated standard deviation of the parameter
of interest) as a function
of the training step for 10 different random initialisations of the neural
network parameters; (b) profiled likelihood around the expectation value
for the parameter of interest $s$ (i.e. number of signal events
in the sample) of 10 trained inference-aware models and 10
trained cross-entropy loss based models. The latter are constructed by
building a uniformly binned Poisson count likelihood of the conditional
signal probability output.  All results
correspond to Benchmark 2.
:::


In [@Fig:training_dynamics], the dynamics of inference-aware optimisation
are shown by the validation loss, which corresponds
to the approximate expected variance
of parameter $s$, as a function of the training step for 10 random-initialised
instances of the \textsc{INFERNO} model corresponding to Benchmark 2.
All inference-aware models were trained during 200 epochs with SGD using
mini-batches of 2000 observations
and a learning rate $\gamma=10^{-6}$. All the model initialisations
converge to summary statistics that provide low variance for the estimator of
$s$ when the nuisance parameters are accounted for.

To compare with alternative approaches and verify the validity of the results,
the profiled likelihood [@tanabashi2018review] obtained both with
the INFERNO-based and
classifier-based
summary statistics, accounting for the
effect of nuisance parameters defined in Benchmark 2, are shown
in [@Fig:profile_likelihood].
The expected uncertainty of the trained models are
used for subsequent inference on the value of $s$
can be estimated from the profile width when $\Delta \mathcal{L} = 0.5$. Hence,
the average width for the profile likelihood using inference-aware training,
$16.97\pm0.11$, can be
compared with the corresponding one obtained by uniformly binning the output of
classification-based models in 10 bins, $24.01\pm0.36$.
The number of bins is the same as that of [@Fig:subfigs_clf_hists],
and is chosen to match
the dimensionality adopted for the summary statistic of INFERNO.
The models based on
cross-entropy loss were
trained during 200 epochs using a mini-batch size of 64 and a fixed learning
rate of $\gamma=0.001$.

A more complete study of the improvement provided by the INFERNO
training procedure is provided in \autoref{tab:results_table},
where the median and 1-sigma
percentiles on the expected absolute uncertainty on $s$ are provided for 100
random-initialised instances of each model. In addition, results for 100
random-initialised cross-entropy neural network models trained as previously
indicated, the optimal (Bayes) classifier
$t_B(\boldsymbol{x} | r = 0.0 , \lambda = 3.0)$
from [@Eq:opt_clf],
and the analytical likelihood-based inference are also
included for comparison.
The expected uncertainties shown in
\autoref{tab:results_table},
with the exception of the analytical likelihood which was based on the
extended likelihood of the generative model from [@Eq:ext_ll],
where obtained by building a binned likelihood by interpolating
the signal and background histograms when the nuisance parameters
are varied. In all cases, the uncertainties quoted are in correspondence
with those obtained from the covariance matrix obtained using 
the Hessian of the negative logarithm the log likelihood, which were found
to match very closely with those obtained by computing the profile
likelihood width.

Except for Benchmark 0, the confidence intervals obtained using
INFERNO-based
summary statistics are considerably narrower than those using
classification and tend to be much closer to those expected when using
the true model likelihood for inference. The results for Benchmark 0,
when no nuisance parameters are considered and thus the mixture components
are perfectly known, show that classification-based summaries in this simplified
setting can outperform the INFERNO technique, which was anticipated
based on the fact that classification will be very efficiently approximating
a sufficient summary statistic (the optimal classifier $t_B$)
as discussed in [appendix @sec:sufficiency]. This factor also
explains why for Benchmark 0 the optimal classifier $t_B$ outperforms the
trained model approximation when it is a sufficient statistic, 
while it does not provide better inference that the approximation
when nuisance parameters are important and thus the sufficiency condition
is not guaranteed.

The analytical likelihood, which amounts to use the true generative likelihood
for inference can be thought of an upper bound for the likelihood-free
setting (i.e. both classification and INFERNO based trained summaries),
because it most effectively uses all the information of the data to constrain
all the model parameters. Much smaller
fluctuations between initialisations are observed for the INFERNO-based
cases than for classification-based summary statistics when nuisance
parameters are relevant. The relative improvement over classification
increases when more nuisance parameters are considered.
As shown in \autoref{tab:results_table}, the constraining power of the summary
statistic generated by INFERNO is stronger when it is constructed to solve
the corresponding inference question, i.e. the training based on
other benchmarks is sub-optimal. Thus the results also
seem to suggest the inclusion of the detailed information about the inference
problem in the INFERNO technique leads to comparable or better results than
its omission.

\begin{table}
  \caption{Expected uncertainty on the parameter of interest $s$
    for each of the inference benchmarks considered using a cross-entropy
    trained neural network model, a INFERNO model customised for each problem,
    the optimal classifier $t_B(\boldsymbol{x} | r = 0.0, \lambda = 3.0)$
    from Eq.~\ref{eq:opt_clf} and the likelihood based results using
    the analytical probability density function from Eq.~\ref{eq:mixture_alt}
    at the parameter values used for training. The last two rows are
    would not be available in a likelihood-free
    setting. The results
    for INFERNO matching each problem are shown with bold characters.}
  \label{tab:results_table}
  \centering
  \footnotesize
  \input{table.tex}
\end{table}


::: {#fig:validity_range .subfigures}
![different $r$ value
](gfx/figure5a.pdf){#fig:range_r_dist width=48%}
![different $\lambda$ value
](gfx/figure5b.pdf){#fig:range_b_rate width=48%}

Expected uncertainty when the true value of the nuisance
parameters $r$ and $\lambda$ is different for those used for training
the model (i.e. $r=0.0$ and $\lambda=3.0$).
The results for 10 randomly initialised 10 learnt summary statistics
(different random initialisation) based either on cross-entropy
classification and the inference-aware technique are shown. Results
correspond to Benchmark 2.
:::

Given that a fixed value of the parameters $\boldsymbol{\theta}_\textrm{MC}$
was used to learn the summary statistics as described in
Algorithm \ref{alg:simple_algorithm} while their true
value is unknown in practical applications, the expected uncertainty
on $s$ has also been
computed for cases when the true value of the parameters
$\boldsymbol{\theta}_{\textrm{true}}$ differs. The
variation of the expected uncertainty on $s$ when either $r$ or $\lambda$
is varied for classification and inference-aware summary statistics is
shown in [@Fig:validity_range] for Benchmark 2. The inference-aware summary
statistics learnt for $\boldsymbol{\theta}_\textrm{MC}$ work
well when  $\boldsymbol{\theta}_{\textrm{true}} \neq \boldsymbol{\theta}_\textrm{MC}$
in the range of variation explored. 

This synthetic example demonstrates that the direct optimisation of inference-aware
losses as those described in the [@Sec:method] is effective.
The summary statistics
learnt accounting for the effect of nuisance parameters compare very favourably
to those obtained by using a classification proxy to approximate the
likelihood ratio.  Of course, more experiments are needed to benchmark the
usefulness of this technique for real-world inference problems as those
found in High Energy Physics analyses at the LHC.

# Conclusions

Classification-based summary statistics for mixture models
often suffer from the need of specifying
a fixed model of the data, thus neglecting the effect of nuisance parameters
in the learning process. The effect of nuisance parameters is only considered
downstream of the learning phase, resulting in sub-optimal inference on
the parameters of interest.

In this work we have described a new approach for building
non-linear summary statistics for
likelihood-free inference that directly minimises the expected
variance of the parameters of interest, which is considerably more
effective than the use of classification surrogates when nuisance
parameters are present.

The results obtained for the synthetic experiment considered clearly
demonstrate that machine learning techniques, in particular neural networks,
can be adapted for learning summary statistics that match the particularities
of the inference problem  at hand, greatly increasing the information available
for subsequent inference. The application of INFERNO to non-synthetic examples
where nuisance parameters are relevant, such as the systematic-extended
Higgs dataset [@estrade2017adversarial], are left for future studies.



Furthermore, the technique presented can be applied to arbitrary
likelihood-free problems as long as the effect of parameters
over the simulated data can be implemented as a differentiable
transformations. As a possible extension,
alternative non-parametric density estimation
techniques such as kernel density could very well be used in
place Poisson count models.

# Acknowledgments {.unnumbered}

Pablo de Castro would like to thank Daniel Whiteson, Peter Sadowski and
the other members of the ML for HEP group at UCI for the initial feedback
and support of the idea presented in this paper, as well as Edward Goul for
his interest when the project was in early stages. The authors would also
like to acknowledge Gilles Louppe and Joeri Hermans for some useful discussions
directly related to this work.

This work is part of a more general effort to develop new statistical
and machine learning techniques
to use in High Energy Physics analyses within within the  AMVA4NewPhysics
project, which is supported by the European Union's Horizon 2020 research
and innovation programme under Grant Agreement number 675440. CloudVeneto
is also acknowledged for the use of computing and storage facilities provided.
