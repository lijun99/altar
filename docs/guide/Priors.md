(prior-distributions)=
# Prior Distributions

## Parameter sets

A model's parameters are grouped into *parameter sets*, each with its own prior distribution, e.g.
the strike and the dip slips of the patches of a fault. The sets are laid out one after another,
in the order of the model's `psets_list`:

```none
model:
    psets_list = [strikeslip, dipslip]
    psets:
        strikeslip = contiguous
        dipslip = contiguous

        strikeslip:
            count = 9
            prior = gaussian
            prior.mean = 0
            prior.sigma = 0.5

        dipslip:
            count = 9
            prior = uniform
            prior.support = (-0.5, 20)
```

A parameter set (`contiguous`) has

`count`
: the number of its parameters;

`prior`
: the prior distribution of its parameters;

`prep`
: the distribution to draw its initial samples from, when it should differ from the prior; e.g.
  samples consistent with a moment magnitude, or the posterior of an earlier run (see
  {ref}`Preset <preset>`). By default, the prior.

AlTar works with the logarithms of the densities. The chains are processed as a batch, so a
distribution works on its own columns of the (samples × parameters) matrix of samples. A proposal
that falls outside the support of a bounded prior is rejected by the sampler as *invalid*.

## Distributions

### Uniform

`uniform`: the uniform distribution on the interval $[a, b]$,

$$
f(x; a, b) = \frac{1}{b-a} \quad \text{for } x \in [a, b], \qquad 0 \text{ otherwise}.
$$

`support`
: the interval $(a, b)$; default `(0, 1)`.

`reparameterize`, `transform`
: sample in an unbounded space instead; see {ref}`below <reparameterization>`.

```none
prior = uniform
prior:
    support = (0, 1)
```

(softuniform)=
### Soft uniform

`softuniform`: a uniform distribution with logistic edges
([Minson, 2024](https://doi.org/10.1093/gji/ggae353)), the normalized difference of two logistic
functions of sharpness $k$,

$$
f(x; a, b, k) = \frac{1}{b-a} \left[\frac{1}{1 + e^{-k(x-a)}} - \frac{1}{1 + e^{-k(x-b)}}\right],
$$

which approaches the uniform distribution on $[a, b]$ as $k$ grows, but is smooth and positive
everywhere, and so changes shape when raised to a power, as {ref}`cross-fade CATMIP <cross-fade>`
does to the prior. Its initial samples are uniform on $[a, b]$.

`support`
: the interval $(a, b)$; default `(0, 1)`.

`sharpness`
: $k$, in inverse units of the parameter; by default $100 / (b - a)$, edges about a hundredth of
  the interval wide.

Outside $[a, b]$, its log density falls by $k$ per unit, so strongly informative data can pull a
parameter across an edge: a bound the data disagree with needs a larger $k$, or a `uniform` prior.

### Gaussian

`gaussian`: the normal distribution,

$$
f(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right].
$$

`mean`
: $\mu$; default 0.

`sigma`
: $\sigma$; default 1.

```none
prior = gaussian
prior:
    mean = 0
    sigma = 2
```

`ugaussian` is the unit Gaussian, $\mu = 0$ and $\sigma = 1$ (cpu only).

### Truncated Gaussian

`tgaussian`: the [truncated Gaussian](https://en.wikipedia.org/wiki/Truncated_normal_distribution),
a Gaussian restricted to, and renormalized on, the interval $[a, b]$.

`mean`, `sigma`
: $\mu$ and $\sigma$ of the Gaussian; defaults 0 and 1.

`support`
: the interval $(a, b)$; default `(0, 1)`.

`reparameterize`, `transform`
: sample in an unbounded space instead; see {ref}`below <reparameterization>`.

```none
prior = tgaussian
prior:
    support = (-1, 1)
    mean = 0
    sigma = 2
```

(correlated)=
### Correlated Gaussian

`correlated`: a Gaussian whose parameters are correlated in space, e.g. the slips of the patches of
a fault, so that neighboring patches tend to slip alike:

$$
\boldsymbol\theta \sim \mathcal N(\mu \mathbf 1, \mathbf C), \qquad
C_{ij} = \sigma^2 \left[ k(r_{ij}) + \epsilon\, \delta_{ij} \right], \qquad
r_{ij} = \left| \frac{\mathbf x_i - \mathbf x_j}{\boldsymbol\ell} \right|,
$$

with $\mathbf x_i$ the location of parameter $i$, $\boldsymbol\ell$ the correlation length, one
value or one per coordinate, and the kernel $k$ one of

| `kernel` | $k(r)$ |
|---|---|
| `exponential` | $e^{-r}$, rough |
| `matern32` | $(1 + \sqrt 3\, r)\, e^{-\sqrt 3\, r}$ |
| `gaussian` | $e^{-r^2/2}$, smooth |

`mean`, `sigma`
: $\mu$ and $\sigma$, the mean and the standard deviation of each parameter; defaults 0 and 1.

`kernel`
: the correlation with distance; default `exponential`.

`length`
: $\boldsymbol\ell$, in the units of the coordinates: one value, `(30,)`, or one per coordinate,
  e.g. `(40, 20)` along strike and down dip; default `(1,)`.

`coordinates`
: the locations of the parameters, (parameters × dimensions), in a text or an `.h5` file; looked
  for in the current directory, then among the model's input files.

`dataset`
: the dataset of `coordinates` in an `.h5` file; by default the first one.

`support`
: an interval the parameters must stay in, e.g. `(0, 15)` to keep the slips positive; unbounded by
  default.

`reparameterize`, `transform`
: sample a bounded `support` in an unbounded space instead; see {ref}`below <reparameterization>`.

`jitter`
: $\epsilon$, which keeps $\mathbf C$ positive definite; default $10^{-6}$.

`sweeps`, `normalization_draws`
: for a bounded `support`: the sweeps of elliptical slice sampling for the initial samples, default
  100, and the draws of the GHK estimate of its mass, default 10000.

Without a support, the initial samples are exact draws. Within one, the distribution is the
Gaussian restricted to it, and renormalized: the mass of the support under the Gaussian, a
multivariate normal integral, is estimated once by the GHK simulator (Geweke, Hajivassiliou and
Keane), with a fixed seed, so that the evidences of runs with the same prior can be compared. Its
log is in the journal. The initial samples come from elliptical slice sampling (Murray et al.,
2010), started inside the support.

For the slips of a fault, the coordinates are the centroids of its patches, in the coordinates of
the fault surface if it curves: distances along the fault, not through the earth. With CSI's
`TriangularPatchesKin`, for one:

```python
numpy.savetxt('centroids.txt', [fault.uv[face].mean(axis=0) for face in fault.Faces])
```

and, for the slips along the plate convergence, positive:

```none
parallel:
    count = 128
    prior = correlated
    prior:
        mean = 0
        sigma = 3
        kernel = exponential
        length = (20,)
        coordinates = centroids.txt
        support = (0, 15)
        reparameterize = True
```

The correlation is a preference, not a constraint: where the data resolve the slip, they override
it; where they don't, it fills the gaps from the neighbors, rather than from the prior mean, patch
by patch. The correlation length is the setting that matters. On a Pedernales-like synthetic, with
independent priors as the reference, the same marginals and lengths of 20 to 80 km:

- the error of the posterior mean slip fell for every true slip distribution tried: smooth
  asperities, a sharp-edged box, and checkerboards of 40 and 20 km;
- with waveforms, every prior recovered even the 20 km checkerboard, one patch cell per block: the
  prior doesn't erase what the data resolve;
- with static data alone, a length longer than the true features made the posterior overconfident:
  on the 40 km checkerboard, 55% of the patches had the truth in their 68% interval at 80 km,
  against 88% with independent priors;
- with static data alone, the evidence rose with the length whatever the truth, so it can't choose
  the length when the data don't resolve the slip; with waveforms, it stopped preferring 40 over
  20 km once the true features were 20 km.

So take the length from what is known of the source, e.g. scaling relations (Mai and Beroza, 2002),
or from the size of the features the data can resolve, rather than from the evidence alone; prefer
the shorter of the lengths in doubt; and check with a {ref}`recovery test <recovery-tests>` that
the posterior covers features of the size you report.

### Positive uniform

`positiveuniform`: the uniform distribution on $(0, 1)$, never exactly 0 (cpu only).

(preset)=
### Preset

`preset` draws the initial samples of a parameter set from a file, e.g. the posterior of an earlier
run; it serves only as a `prep`, never as a prior.

`input_file`
: an `.h5` file, e.g. an archived step; looked for in the current directory, then among the
  model's input files.

`dataset`
: the (samples × parameters) dataset; for an archived step, e.g.
  `ParameterSets/strikeslip`, its physical samples (`strikeslip_physical`) are used when it was
  reparameterized, else `strikeslip_sampling`.

Each chain takes the next row of the file; with more chains than rows, the rows are reused. Under
MPI, each process starts from its own rows.

```none
strikeslip:
    count = 9
    prep = preset
    prep:
        input_file = results/static/step_final.h5
        dataset = ParameterSets/strikeslip
    prior = gaussian
    prior.sigma = 0.5
```

### Moment magnitude

`altar.models.seismic.moment`, for the slips of earthquake source models, comes with the seismic
models; see {doc}`Static`.

(reparameterization)=
## Reparameterization

The gradient-based samplers, HMC, MALA and SGLD, need priors defined on the whole real line: a trajectory
has no way to respect the bounds of a uniform prior. A bounded prior can instead be sampled in an
unbounded *sampling space*, mapped to its *physical space* by a transform. With the logit
transform, the default, a sample $s \in \mathbb R$ maps to

$$
x = a + (b - a)\, \sigma(s), \qquad \sigma(s) = \frac{1}{1 + e^{-s}},
$$

which lies in $(a, b)$ for any $s$. To reparameterize a prior, set `reparameterize`:

```none
prior = uniform
prior:
    support = (-0.5, 20)
    reparameterize = True
```

The model still sees physical values, and the prior is still the density of the physical values:
the chains move in sampling space, and the log-Jacobian of the map,
$\log |\mathrm{d}x/\mathrm{d}s|$, which the posterior of the sampling-space values includes, is kept
apart. The archived steps then hold, for each such parameter set, both `<pset>_physical` and
`<pset>_sampling` samples, and the log-Jacobian of each sample in `jacobian`.

`uniform`, `tgaussian`, `correlated` and the seismic `moment` prior can be reparameterized, on the
cpu and the GPU. The gradient-based samplers, HMC, MALA and SGLD, need it: they refuse bounded priors that aren't
reparameterized, and name them. Metropolis (and so CATMIP) takes it as an option. Without it,
Metropolis walks the physical values and rejects the proposals outside the support, which slows
the chains down when the posterior presses against a bound. With it, Metropolis walks the
sampling-space values, so that every proposal is in the support, and accepts on the posterior plus
the log-Jacobian. The posterior of the physical values is the same either way.

`transform`
: the transform, `altar.distributions.logittransform` by default, the only one for now.

## More distributions

More distributions can be added, following the existing ones: see the {doc}`Programming`.
