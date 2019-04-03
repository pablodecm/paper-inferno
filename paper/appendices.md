

\appendix
\renewcommand{\thesection}{\Alph{section}}

# Sufficient Statistics for Mixture Models  {#sec:sufficiency}

Let us consider the general problem of inference for a two-component
mixture problem, which is very common in scientific disciplines such
as High Energy Physics.
While their functional form will not be explicitly specified to keep
the formulation general, one of the components will be denoted as signal
$f_s(\boldsymbol{x}| \boldsymbol{\theta})$ and the other as background
$f_b(\boldsymbol{x} | \boldsymbol{\theta})$, where  $\boldsymbol{\theta}$ is
are of all parameters the distributions might depend on. The probability
distribution function of the mixture can then be expressed as:
$$
p(\boldsymbol{x}| \mu, \boldsymbol{\theta} ) = (1-\mu) f_b(\boldsymbol{x} | \boldsymbol{\theta}) 
                                                + \mu f_s(\boldsymbol{x} | \boldsymbol{\theta})
$$ {#eq:mixture_general}
where $\mu$ is a parameter corresponding to the signal mixture fraction.
Dividing and multiplying by $f_b(\boldsymbol{x} | \boldsymbol{\theta})$ we
have:
$$
p(\boldsymbol{x}| \mu, \boldsymbol{\theta} ) = f_b(\boldsymbol{x} | \boldsymbol{\theta})   \left ( 1-\mu
                    + \mu \frac{f_s(\boldsymbol{x} | \boldsymbol{\theta})}{f_b(\boldsymbol{x} | \boldsymbol{\theta})}
                    \right )  
$$ {#eq:mixture_div}
from which we can already prove that the density ratio
$s_{s/ b}= f_s(\boldsymbol{x} | \boldsymbol{\theta}) / f_b(\boldsymbol{x} | \boldsymbol{\theta})$
(or alternatively its inverse) is a sufficient summary statistic for the
mixture coefficient parameter $\mu$. This would also be the case for
the parametrization using $s$ and $b$ if the alternative $\mu=s/(s+b)$
formulation presented for the synthetic problem in Sec. \ref{sec:d-synthetic-mixture}.

However, previously in this work (as well as for most studies using
classifiers to construct summary statistics) we have been using the
summary statistic $s_{s/(s+b)}= f_s(\boldsymbol{x} | \boldsymbol{\theta}) /(
  f_s(\boldsymbol{x} | \boldsymbol{\theta}) + f_b(\boldsymbol{x} | \boldsymbol{\theta}))$
instead of $s_{s/ b}$. The advantage of $s_{s/(s+b)}$ is that it represents
the conditional probability of one observation $\boldsymbol{x}$ coming
from the signal assuming a balanced mixture, and hence is bounded between
zero and one. This greatly simplifies its visualisation and non-parametetric
likelihood estimation. Taking [@Eq:mixture_div] and manipulating the
subexpression depending on $\mu$ by adding and subtracting $2\mu$  we have:
$$
p(\boldsymbol{x}| \mu, \boldsymbol{\theta} ) = f_b(\boldsymbol{x} | \boldsymbol{\theta})   \left ( 1-3\mu
                    + \mu \frac{f_s(\boldsymbol{x} | \boldsymbol{\theta}) + f_b(\boldsymbol{x} | \boldsymbol{\theta})}{f_b(\boldsymbol{x} | \boldsymbol{\theta})}
                    \right )  
$$ {#eq:mixture_sub}
which can in turn can be expressed as:
$$
p(\boldsymbol{x}| \mu, \boldsymbol{\theta} ) = f_b(\boldsymbol{x} | \boldsymbol{\theta})   \left ( 1-3\mu
                    + \mu \left ( 1- \frac{f_s(\boldsymbol{x} | \boldsymbol{\theta})}{f_s(\boldsymbol{x} | \boldsymbol{\theta})
                  +f_b(\boldsymbol{x} | \boldsymbol{\theta})} \right )^{-1}
                    \right )  
$$ {#eq:mixture_suff}
hence proving that $s_{s/(s+b)}$ is also a sufficient statistic and theoretically
justifying its use for inference about $\mu$. The advantage of both $s_{s/(s+b)}$ 
and $s_{s/b}$ is they are one-dimensional and do not depend on the
dimensionality of $\boldsymbol{x}$ hence allowing much more efficient
non-parametric density estimation from simulated samples. Note that
we have been only discussing sufficiency with respect to the mixture
coefficients and not the additional distribution parameters
$\boldsymbol{\theta}$. In fact, if a subset of $\boldsymbol{\theta}$ 
parameters are also relevant for inference (e.g. they are nuisance
parameters) then $s_{s/(s+b)}$ and $s_{s/b}$ are not sufficient statistics
unless the $f_s(\boldsymbol{x}| \boldsymbol{\theta})$ and
$f_b(\boldsymbol{x}| \boldsymbol{\theta})$ have very specific functional
form that allows a similar factorisation. 

