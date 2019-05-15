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
  data modelling in many scientific disciplines, including experimental
  High Energy Physics, making statistical
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
in the simulation often lacks a tractable probability density
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

In many cases, such as particle collisions at the Large Hadron Collider (LHC),
the nature of the generative model (i.e. a mixture of different
processes)
allows the treatment of the problem as
signal (s) vs background (b) classification [@adam2015higgs],
when the task becomes one of effectively estimating
an approximation of $t_B(\boldsymbol{x}) = p_{s}(\boldsymbol{x})/(p_{s}(\boldsymbol{x})+
p_{b}(\boldsymbol{x}))$ by means of probabilistic classification.
While the use of classifiers to learn a summary statistic can
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
descent (SGD) [@DBLP:journals/corr/Ruder16] using small subsets
of available simulated data.
The learned
summary statistics can be used to
perform robust and efficient classical or
Bayesian inference from the observed data, so they can be readily applied
in place of current classification-based or domain-motivated summary statistics
in current scientific data analysis workflows.

# Method {#sec:method}

In this section, a machine learning technique to learn non-linear
sample summary statistics is described
The method seeks to minimise the expected variance
of the parameters of interest obtained via a non-parametric
simulation-based synthetic likelihood. A graphical description of the
technique  is depicted on [@Fig:diagram].
The parameters of a neural network are
optimised by SGD within an automatic differentiation
framework, where the considered loss function accounts for the details of
the statistical model as well as the expected effect of nuisance parameters.

![Learning inference-aware summary statistics (see text for
  details).](gfx/figure1.pdf){#fig:diagram width=80%}

The family of summary statistics $\boldsymbol{t}(D)$ considered in this
work is composed by a neural network model applied to each dataset
observation $\boldsymbol{h}(\boldsymbol{x}; \boldsymbol{\phi}) : \mathbb{R}^{d} \rightarrow
\mathbb{R}^{m}$,
whose parameters $\boldsymbol{\phi}$ will be learned during the
training phase.
The neural network $\boldsymbol{h}(\boldsymbol{x}_i; \boldsymbol{\phi})$
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

In experimental high-energy physics experiments, which are the scientific
context that initially motivated this work, histograms of
observation counts are the most
commonly used non-parametric density estimator because the
resulting likelihoods
can be expressed as the product of Poisson factors, one for each of
the considered bins. A naive
sample summary statistic can be built from the output of the neural network
by simply assigning each observation $\boldsymbol{x}$ to a bin corresponding
to the cardinality of the maximum element of
$\boldsymbol{h}(\boldsymbol{x}; \boldsymbol{\phi})$,
which can in turn be used to build the following likelihood, where the
expectation for each bin is taken from the simulated sample $G_\textrm{MC}$:
$$
\mathcal{L}(D; \boldsymbol{\theta},\boldsymbol{\phi})=\prod_{i=0 }^m
             \textrm{Pois} \left ( t_i (D; \boldsymbol{\phi}) \:  |
             \: \left ( \frac{n}{l} \right ) t_i (G_\textrm{MC};\boldsymbol{\phi}) \right )
$$ {#eq:likelihood}
where $t_i (D; \boldsymbol{\phi})$ are the sum of observation
for which the maximum is at the bin $j$ and
the $n/l$ factor accounts for the different number of
observations in the simulated samples.
In the above construction, the chosen
family of summary statistics is non-differentiable due to
the $argmax$ operator, so gradient-based updates for the parameters
cannot be computed. To work around this problem, a differentiable
approximation $\hat{\boldsymbol{t}}(D ; \boldsymbol{\phi})$ is considered.
This function is defined by means of a *softmax* operator
$\hat{t}_i(D;\boldsymbol{\phi})=\sum_{x \in D} e^{f_i(\boldsymbol{x}; \boldsymbol{\phi})/\tau}  \ \sum_{j=0}^{m} e^{f_j(\boldsymbol{x}; \boldsymbol{\phi})/\tau}$,
where the temperature hyper-parameter
$\tau$ will regulate the softness of the operator.
Similarly, let us
denote by $\hat{\mathcal{L}}(D; \boldsymbol{\theta}, \boldsymbol{\phi})$
the differentiable approximation of the non-parametric likelihood
obtained by substituting $\boldsymbol{t}(D ; \boldsymbol{\phi})$ with
$\hat{\boldsymbol{t}}(D ; \boldsymbol{\phi})$. Instead
of using the observed data $D$, the value of $\hat{\mathcal{L}}$
may be computed
when the observation for each bin is equal to its corresponding
expectation based on
the simulated sample $G_\textrm{MC}$, which is commonly denoted as the
Asimov likelihood [@cowan2011asymptotic] $\hat{\mathcal{L}}_A$.
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
if the simulation function, which
we denote here and below as $g(\boldsymbol{\theta}_\textrm{MC})$, is
differentiable or if the effect
of varying $\boldsymbol{\theta}$ over the simulated
dataset $G_\textrm{MC}$ can be effectively approximated.

The inverse of the Fisher information can be used as an
approximate estimator of the expected covariance matrix of the
parameters $\boldsymbol{\theta}$ for an unbiased estimator.
In Bayesian
terminology, this approach is referred to as the Laplace approximation
[@laplace1986memoir].
The loss function used for stochastic optimisation of the neural network
parameters $\boldsymbol{\phi}$ can be any function of the inverse
of the Fisher information matrix at $\boldsymbol{\theta}_\textrm{MC}$, depending on the
ultimate inference aim. The diagonal
elements $I_{ii}^{-1}(\boldsymbol{\theta}_\textrm{MC})$ correspond to the expected
variance of each of the $\phi_i$ under the approximation mentioned
before, so if the aim is efficient inference about one of the parameters
$\omega_0 = \theta_k$ a candidate loss function is $U = I_{kk}^{-1}(\boldsymbol{\theta}_\textrm{MC})$
which corresponds to the expected width of the confidence interval
for $\omega_0$ accounting also for the effect of the other nuisance
parameters in $\boldsymbol{\theta}$.

# Experiments {#sec:d-synthetic-mixture}

In this section, we first study the effectiveness of the inference-aware
optimisation in a synthetic mixture problem where the likelihood is known. We then
compare our results with those obtained by standard classification-based
summary statistics. All the code needed to reproduce the results
presented the results presented here is available in
an online repository [@code_repository], extensively using \textsc{TensorFlow}
[@tensorflow2015-whitepaper]
and \textsc{TensorFlow Probability} [@tran2016edward;@dillon2017tensorflow] software libraries.

To demonstrate the usage of the proposed approach a three-dimensional
mixture example with two components
$p(\boldsymbol{x}| \mu, r, \lambda) = (1-\mu) f_b(\boldsymbol{x} | r, \lambda) 
+ \mu f_s(\boldsymbol{x})$
is considered.
One component will be referred as background $f_b(\boldsymbol{x} | r, \lambda)$ and
the other as signal $f_s(\boldsymbol{x})$; their probability density functions
are taken to correspond respectively to:
$$
\scriptstyle f_b(\boldsymbol{x} | r, \lambda) =
\mathcal{N} \left (
  (x_0, x_1) \, \middle | \,
  (2+r, 0),
  \begin{bmatrix}
    5 & 0 \\
    0 & 9 \\
   \end{bmatrix}
\right)
Exp (x_2 | \lambda)
\quad
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
$$ {#eq:sig_bkg_den}
so that $(x_0,x_1)$ are distributed according to a multivariate normal
distribution while $x_2$ follows an independent exponential distribution
both for background and signal.

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
If the probability density is known, but the expectation for
the number of observed events depends on the model parameters, the likelihood
can be extended [@barlow1990extended] with a Poisson count term as:
$\mathcal{L}(s, r, \lambda, b) = \textrm{Pois}(n | s+b) \prod^{n} p(\boldsymbol{x}| s,r, \lambda, b)$
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
the mixture coefficients.

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

Another option would be to pose the problem as one of classification.
A supervised machine learning model such a
neural network can be trained to discriminate signal and background
observations, considering parameters $r$ and $\lambda$ fixed.
The output of such a model typically consists in class probabilities
$c_b$ and $c_s$ given an observation $\boldsymbol{x}$, the latter will
asymptotically tend to the optimal classifier from [@Eq:opt_clf] given
enough data and a flexible enough model.
The classification output is a powerful
learned feature that can be used as summary statistics; however
their construction
ignores the effect of the nuisance parameters.

The statistical model described above has up to four unknown parameters: the
expected number of signal observations $s$,
the background mean shift $r$,
the background exponential rate in the third dimension $\lambda$, and
the expected number of background observations. The effect of the
expected number of signal and background observations $s$ and $b$
can be easily included in the computation graph by 
weighting the signal and background observations.
Instead the effect of $r$ and $\lambda$, both nuisance parameters
that will define the background distribution, is more easily modelled
as a transformation of the input data $\boldsymbol{x}$. In particular,
$r$ is a nuisance parameter that causes a shift on the background
along the first dimension  adn
the effect of $\lambda$ can be modelled by multiplying
$x_2$ by the ratio between the $\lambda_0$ used for generation and the
one being modelled.

For this problem, we are interested in carrying out statistical
inference on the parameter of interest $s$. In fact,
the performance of inference-aware optimisation as described in
[@Sec:method] will be compared with classification-based summary statistics for
a series of inference benchmarks based on the synthetic problem described above
that vary in the number of nuisance parameters considered and their constraints,
as shown in \autoref{tab:benchmark_table}. For Benchmark 0 no nuisance
parameters are considered, so the classification approach is expected
to provide near optimal summary statistics. The rest of the benchmarks
correspond to the presence of nuisance parameters,
differing among them in their number and constrains. The main figure of merit
will be the expected
uncertainty in the parameter of interest $s$ for the inference problem
defined for each benchmark and conditioned
on the true value of the parameters of $s=50$, $r=0.0$,
$\lambda=3.0$ and $b=1000$.

\begin{table}
  \caption{Definition of the different statistical
  inference benchmark problems that will be
  considered when comparing different techniques to obtain
  summary statistics.}
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
that the differentiable approximations are close to the true values.
Standard
mini-batch stochastic gradient descent is used for training and
the optimal learning rate is fixed and decided by means of a
simple scan; the best choice found is specified together with the results.

::: {#fig:subfigs_training .subfigures}
![inference-aware training loss
](gfx/figure4a.pdf){#fig:training_dynamics width=37%}
![profile-likelihood comparison
](gfx/figure4b.pdf){#fig:profile_likelihood width=38%}
 
Dynamics and results of inference-aware optimisation: (a) square root of
inference-loss (i.e. approximated uncertainty of the parameter
of interest) as a function
of the training step for 10 different random initialisations of the neural
network parameters; (b) profiled likelihood around the expectation value
for the parameter of interest $s$ of 10 trained inference-aware models and 10
trained cross-entropy loss based models. All results
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
extended analytical likelihood,
were obtained by building a binned likelihood by interpolating
the signal and background histograms when the nuisance parameters
are varied. In all cases, the uncertainties quoted are in correspondence
with those obtained from the covariance matrix obtained using 
the Hessian of the negative logarithm of the log likelihood, which were found
to match very closely with those obtained by computing the profile
likelihood width.

Except for Benchmark 0, the confidence intervals obtained using
INFERNO-based
summary statistics are considerably narrower than those using
classification and tend to be much closer to those expected when using
the true model likelihood for inference. The results for Benchmark 0,
when no nuisance parameters are considered and thus the mixture components
are perfectly known, show that classification-based summaries in this simplified
setting can outperform the INFERNO technique. This factor also
explains why for Benchmark 0 the optimal classifier $t_B$ outperforms the
trained model approximation when it is a sufficient statistic, 
while it does not provide better inference that the approximation
when nuisance parameters are important and thus the sufficiency condition
is not guaranteed.

The analytical likelihood, which amounts to use the true generative likelihood
for inference, can be thought of an upper bound for the likelihood-free
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
    from Eq.~\ref{eq:opt_clf} and the analytical likelihood.
    The results
    for INFERNO matching each problem are shown in bold.}
  \label{tab:results_table}
  \centering
  \footnotesize
  \input{table.tex}
\end{table}

# Conclusions

In this work we have described a new approach for building
non-linear summary statistics for
likelihood-free inference that directly minimises the expected
variance of the parameters of interest, which is considerably more
effective than the use of classification surrogates when nuisance
parameters are present. The application of INFERNO to non-synthetic examples
where nuisance parameters are relevant, such as the systematic-extended
Higgs dataset [@estrade2017adversarial], are left for future studies.


# Acknowledgments {.unnumbered}

Pablo de Castro would like to thank Daniel Whiteson, Peter Sadowski and
the other members of the ML for HEP group at UCI for the initial feedback
and support of the idea presented in this paper, as well as Edward Goul for
his interest when the project was in early stages. The authors would also
like to acknowledge Gilles Louppe and Joeri Hermans for some useful discussions
directly related to this work.

This work is part of a more general effort to develop new statistical
and machine learning techniques
to use in high-energy Physics analyses within within the  AMVA4NewPhysics
project, which is supported by the European Union's Horizon 2020 research
and innovation programme under Grant Agreement number 675440. CloudVeneto
is also acknowledged for the use of computing and storage facilities provided.
