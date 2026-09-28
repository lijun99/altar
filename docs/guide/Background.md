(background)=
# Bayesian Inference for Inverse Problems

## Inverse problems

An inverse problem in science is to infer a set of unknown parameters,
$\boldsymbol\theta = \{\theta_1, \theta_2, \ldots, \theta_m\}$, from observed data
$\mathbf d = \{d_1, d_2, \ldots, d_n\}$. For example, in seismology we infer the rupture of an
earthquake (parameterized as $\boldsymbol\theta$) from the ground motions ($\mathbf d$) measured
by seismometers, GPS, InSAR and other geodetic surveys. In most cases the forward problem
$\mathbf d = G(\boldsymbol\theta)$ is well defined, but the inverse
$\boldsymbol\theta = G^{-1}(\mathbf d)$ is not: a linear system
$G(\boldsymbol\theta) = \mathbf G \boldsymbol\theta$ may be ill-posed, and a nonlinear one may be
inherently hard to invert.

## The Bayesian approach

Bayesian inference offers a statistical solution: it treats the unknown parameters
$\boldsymbol\theta$ as random variables, with the conditional probability
$P(\boldsymbol\theta|\mathbf d)$ given by Bayes' theorem,

$$
\begin{aligned}
P(\boldsymbol\theta|\mathbf d) &= \frac{P(\boldsymbol\theta)\, P(\mathbf d|\boldsymbol\theta)}{P(\mathbf d)}, \\
P(\boldsymbol\theta|\mathbf d) &: \text{the posterior, the probability of $\boldsymbol\theta$ given the data $\mathbf d$}, \\
P(\boldsymbol\theta) &: \text{the prior, the probability of $\boldsymbol\theta$ without regard to $\mathbf d$}, \\
P(\mathbf d|\boldsymbol\theta) &: \text{the likelihood, the probability of observing $\mathbf d$ given $\boldsymbol\theta$}, \\
P(\mathbf d) &: \text{the evidence, the probability of observing $\mathbf d$, independent of $\boldsymbol\theta$}.
\end{aligned}
$$

The evidence $P(\mathbf d)$ is the same for all $\boldsymbol\theta$, so it only normalizes the
posterior. The likelihood follows from the forward model, e.g., for Gaussian errors,

$$
P(\mathbf d|\boldsymbol\theta) \propto
\exp\left\{-\frac{1}{2} \left[\mathbf d - G(\boldsymbol\theta)\right]^T C_\chi^{-1}
\left[\mathbf d - G(\boldsymbol\theta)\right]\right\},
$$

where the covariance matrix $C_\chi$ captures the errors in the observations $\mathbf d$ (the
data covariance $C_d$), and possibly the uncertainties of the forward model itself (a model
covariance $C_p$, see {doc}`StaticCp`).

The parameters with the maximum posterior probability (MAP), or the mean or median of the
posterior, estimate the solution of the inverse problem. Unlike a single point estimate, the full
posterior also shows when several sets of $\boldsymbol\theta$ explain the data comparably well, or
when the posterior is not unimodal, and it quantifies the uncertainty of the solution.

The price is computation: the forward model has to be evaluated for very many $\boldsymbol\theta$.
Efficient sampling algorithms and parallel hardware, GPUs in particular, make this feasible for
problems with many parameters and expensive forward models.

(catmip)=
## The CATMIP algorithm

Rather than evaluating $P(\boldsymbol\theta)\, P(\mathbf d|\boldsymbol\theta)$ over the entire
parameter space, Markov chain Monte Carlo (MCMC) methods draw samples distributed as the
posterior. CATMIP (Cascading Adaptive Transitional Metropolis in Parallel) belongs to the MCMC
methods that use *transitioning*: samples are first drawn from the prior, and then *annealed* to
the posterior through a series of intermediate distributions,

$$
P_m(\boldsymbol\theta|\mathbf d) \propto P(\boldsymbol\theta)\, P(\mathbf d|\boldsymbol\theta)^{\beta_m},
$$

where $\beta_m$, in analogy to an inverse temperature, increases from $\beta_0 = 0$ to
$\beta_M = 1$ in $M$ steps:

1. At $\beta_0 = 0$, draw $N_s$ samples from the prior $P_0(\boldsymbol\theta|\mathbf d) =
   P(\boldsymbol\theta)$, as the seeds of $N_s$ parallel chains.
2. Choose the next $\beta_{m+1}$ from the statistics of the current samples. CATMIP picks it so
   that the coefficient of variation (COV) of the importance weights
   $w_i = P(\mathbf d|\boldsymbol\theta_i)^{\beta_{m+1}-\beta_m}$ reaches a target, typically 1,
   i.e., an effective sample size of 50%.
3. Resample by the importance weights $\{w_i\}$: samples with small weights may be dropped and
   samples with large weights duplicated, keeping $N_s$ samples as the seeds of the chains at
   $\beta_{m+1}$.
4. Run each chain for a number of Metropolis–Hastings steps at $\beta_{m+1}$, with a Gaussian
   proposal whose covariance comes from the samples; the acceptance rate rescales the proposal for
   the next $\beta$ step.
5. Repeat steps 2–4 until $\beta_M = 1$.

Step 4 carries most of the computation. The chains are independent, so it is embarrassingly
parallel, which makes CATMIP a natural fit for parallel computers and GPUs.

(gradient-samplers)=
## Gradient-based samplers

Besides the Metropolis–Hastings random walk, AlTar can move the chains with samplers that use the
gradient of the posterior, $\nabla_{\boldsymbol\theta} \log P(\boldsymbol\theta|\mathbf d)$, which
explore high-dimensional posteriors more efficiently:

- **Hamiltonian Monte Carlo (HMC)** simulates the motion of a particle on the potential
  $U(\boldsymbol\theta) = -\log P(\boldsymbol\theta|\mathbf d)$ with leapfrog integration, and
  accepts or rejects the end of each trajectory with a Metropolis test. It can replace the random
  walk in step 4 of CATMIP, or sample the posterior directly at $\beta = 1$.
- **Stochastic gradient Langevin dynamics (SGLD)** follows the gradient with injected Gaussian
  noise and a decreasing step size, without an accept/reject step.

Both need the model to provide the gradient of its data likelihood, and priors that are defined on
the whole real line. Bounded priors, e.g., uniform ones, are sampled in an unbounded space instead,
by reparameterizing them, e.g., with a logit transform; see {doc}`Priors`.

## References

1. Albert Tarantola, *Inverse Problem Theory and Methods for Model Parameter Estimation*, SIAM,
   2005. ISBN: 978-0-89871-572-9.
2. Sarah E. Minson, Mark Simons, and James L. Beck, *Bayesian inversion for finite fault
   earthquake source models I — theory and algorithm*, Geophysical Journal International, Vol. 194,
   1701 (2013).
3. Radford M. Neal, *MCMC using Hamiltonian dynamics*, in *Handbook of Markov Chain Monte Carlo*,
   Chapman & Hall/CRC, 2011.
4. Max Welling and Yee Whye Teh, *Bayesian learning via stochastic gradient Langevin dynamics*,
   Proceedings of the 28th International Conference on Machine Learning (ICML), 2011.
