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

`uniform`, `tgaussian` and the seismic `moment` prior can be reparameterized. Reparameterization is
for the gradient-based samplers, HMC, MALA and SGLD, on the cpu and the GPU; they refuse bounded priors
that aren't reparameterized, and name them. Metropolis ignores it: it walks the
physical values, rejects the proposals outside the support, and keeps the sampling-space values in
step, so its results are the same either way.

`transform`
: the transform, `altar.distributions.logittransform` by default, the only one for now.

## More distributions

More distributions can be added, following the existing ones: see the {doc}`Programming`.
