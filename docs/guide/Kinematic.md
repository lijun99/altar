(kinematic-inversion)=
# Kinematic Slip Inversion

## The kinematic source model

The kinematic source model infers how the slip evolved in time, not only its final value.

The fault, a rectangle, is divided into $N_{dd} \times N_{as}$ square patches (down dip, along
strike), and time into $N_t$ intervals. The slip function $\mathbf M_b(\vec\xi, t)$ ("big M") is

$$
\mathbf M_b(\vec\xi, t) = \mathbf D(\vec\xi)\, S\big(t - T_R(\vec\xi);\, T_r(\vec\xi)\big),
$$

where $\vec\xi$ labels the patch and

- $\mathbf D(\vec\xi)$ is the final slip, with its strike and dip components, the same as in the
  {doc}`static model <Static>`;
- $T_R(\vec\xi)$ is the time the rupture reaches the patch, from the solution of the eikonal
  equation by fast sweeping, given the hypocenter $\mathbf H_0$ and the rupture velocity
  $V_r(\vec\xi)$ (isotropic, and constant within each patch);
- $T_r(\vec\xi)$ is the rise time, the duration of the slip on the patch;
- $S(t; T_r)$ is the source time function, a triangle over $[0, T_r]$ with unit integral.

For smooth predictions, each patch is refined into an $N_{mesh} \times N_{mesh}$ grid to solve the
eikonal equation, and each time interval into $N_{pt}$ points; $\mathbf M_b$ is interpolated and
integrated from these finer meshes. The predicted data are then

$$
d^{pred}(\vec x, t) = \sum_{\vec\xi,\, t'} \mathcal G_b(\vec x, \vec\xi;\, t - t')\, \mathbf M_b(\vec\xi, t'),
$$

linear in $\mathbf M_b$ (though $\mathbf M_b$ itself is not linear in the parameters), where
$\mathcal G_b$ ("big G") are the kinematic Green's functions, from a source on a patch to an
observation at a time. They are computed beforehand, e.g. with the frequency-wavenumber
integration of [Zhu and Rivera](https://doi.org/10.1046/j.1365-246X.2002.01610.x), or with
[AXITRA](https://github.com/coutanto/axitra).

The parameters of the kinematic model are therefore

- the two components of the slip, $\mathbf D(\vec\xi)$: $2 N_{dd} N_{as}$ values;
- the rupture velocity $V_r(\vec\xi)$: $N_{dd} N_{as}$ values;
- the rise time $T_r(\vec\xi)$: $N_{dd} N_{as}$ values;
- the hypocenter $\mathbf H_0$: 2 values,

and the forward model takes two steps: $\mathbf M_b$ from the eikonal solver, then
$\mathbf d^{pred} = \mathcal G_b \mathbf M_b$.

```{note}
The kinematic model runs on the GPU (`job.gpus = 1`) or on the cpu (`job.gpus = 0`), where the
samples run in parallel threads (`OMP_NUM_THREADS`), with the same results to round-off; the GPU is
much faster for production runs. It provides the exact gradient of its data likelihood for HMC and
MALA, by the adjoint of its forward model: back through the source time functions, the
interpolation of the arrival times, and their fast sweeping and seeding.
```

## Joint static and kinematic inversion

The slips $\mathbf D(\vec\xi)$ are constrained by both the static and the kinematic data, so the
two models are usually solved together, as an *ensemble* of models sharing their parameters. The
joint posterior is

$$
P(\boldsymbol\theta_c, \boldsymbol\theta_s, \boldsymbol\theta_k | \mathbf d_s, \mathbf d_k)
\propto P(\boldsymbol\theta_c)\, P(\boldsymbol\theta_s)\, P(\boldsymbol\theta_k)\,
P(\mathbf d_s | \boldsymbol\theta_c, \boldsymbol\theta_s)\,
P(\mathbf d_k | \boldsymbol\theta_c, \boldsymbol\theta_k),
$$

where $\boldsymbol\theta_c$ are the parameters the models share, the slips;
$\boldsymbol\theta_s$ those of the static model only, e.g. InSAR ramps; and $\boldsymbol\theta_k$
those of the kinematic model only, the rise times, rupture velocities and hypocenter. Annealing
goes through the intermediate distributions

$$
P(\boldsymbol\theta_c)\, P(\boldsymbol\theta_s)\, P(\boldsymbol\theta_k)\,
P(\mathbf d_s | \boldsymbol\theta_c, \boldsymbol\theta_s)^{\beta_s}\,
P(\mathbf d_k | \boldsymbol\theta_c, \boldsymbol\theta_k)^{\beta_k},
$$

with two schemes:

- **non-cascading**: $\beta_s = \beta_k = \beta$ rise together from 0 to 1, and the COV scheduler
  weighs the samples by both likelihoods;
- **cascading**: first the static model alone is solved, for the posterior of
  $\boldsymbol\theta_c$ and $\boldsymbol\theta_s$; then the ensemble starts from those samples,
  with the static likelihood at full weight, $\beta_s = 1$, while $\beta_k$ rises from 0 to 1, the
  COV scheduler weighing the samples by the kinematic likelihood only.

The static inversion narrows the slips down, usually by a lot, so the cascading scheme converges
much faster, and is recommended whenever an expensive model, such as the kinematic one, is part of
the ensemble.

## The kinematic model

A configuration for the kinematic model alone, in the examples, `kinematic.pfg`:

```{literalinclude} ../../models/seismic/examples/kinematic.pfg
:language: none
:caption: kinematic.pfg (the model)
:start-at: "    ; the kinematic model"
:end-before: "    controller = altar.bayesian.catmip"
```

### Parameter sets

The kinematic model reads its parameters in this order: `strikeslip`, `dipslip`, `risetime`,
`rupturevelocity`, `hypocenter`, whatever the names of the parameter sets (`strikeslip` and
`dipslip` may be swapped, to match the Green's functions).

- `strikeslip` and `dipslip` are the slips, in m; they can start from the posterior of a static
  inversion, with a `preset` prep (see {doc}`Priors`), or from a distribution, as in the
  {doc}`static inversion <Static>`.
- `risetime` (in s) and `rupturevelocity` (in km/s) are positive, e.g. with uniform or truncated
  Gaussian priors.
- These four have one value per patch, the patches in the order
  $(as_0, dd_0), (as_0, dd_1), \ldots, (as_0, dd_{N_{dd}-1}), (as_1, dd_0), \ldots$: down dip first.
- `hypocenter` (in km) is the location of the hypocenter along strike, then down dip, measured from
  the **center** of patch $(as_0, dd_0)$, not from its corner. With different priors along the two
  directions, split it into two parameter sets, the along-strike one first.

`idx_map` can instead list the columns of the parameters, in the order above, when the model sees
more parameters than its own.

### Input files

`green`
: the kinematic Green's functions, of shape $(2 N_{dd} N_{as} N_t, N_{obs})$: for each time
  interval, component (strike, then dip), patch along strike and patch down dip, in this order,
  the Green's functions for all observations,

  ```none
  (t=0, strike, as_0, dd_0, obs_0), (t=0, strike, as_0, dd_0, obs_1), ..., (t=0, strike, as_0, dd_0, obs_{Nobs-1})
  (t=0, strike, as_0, dd_1, obs_0), ...
  ...
  (t=0, strike, as_{Nas-1}, dd_{Ndd-1}, obs_0), ...
  (t=0, dip, as_0, dd_0, obs_0), ...
  ...
  (t=1, strike, as_0, dd_0, obs_0), ...
  ...
  (t=Nt-1, dip, as_{Nas-1}, dd_{Ndd-1}, obs_0), ..., (t=Nt-1, dip, as_{Nas-1}, dd_{Ndd-1}, obs_{Nobs-1})
  ```

  which is the order of $\mathbf M_b$ in the forward model.

`dataobs.data_file`, `dataobs.cd_file` or `dataobs.cd_std`
: the observed data, and their covariance or a common standard deviation, as for the static
  model.

The files can be text, raw binary or HDF5, told by their suffix. The Green's functions of the
example, `9patch/kinematicG.gf.h5` (88 MB), are not in the repository; see `9patch/NOTE`.

### Settings

`Nas`, `Ndd`
: the number of patches along strike and down dip.

`Nmesh`
: the grid points along each side of a patch, for the eikonal equation.

`dsp`
: the length of a side of a patch, in km.

`Nt`, `dt`
: the number of time intervals, long enough to cover the rupture, and their length, in s.

`Npt`
: the points within each time interval.

`t0s`
: a start time for each patch, added to the arrival time of the rupture; chosen well, they reduce
  the number of time intervals needed. By default 0.

`cp`, `cmu_file`, `kmu_file`
: the uncertainty of the Green's functions (see {doc}`StaticCp`); the kinematic kernels have the
  shape of the Green's functions.

## The kinematic model on triangular meshes

`altar.models.seismic.kinematicmesh` is the same source model on a fault meshed with triangles,
planar or curved, e.g. by CSI's `TriangularPatchesKin`. The rupture front spreads from the
hypocenter by the fast iterative method, on a mesh refined from the patches, and each patch slips
at its source points, each from the moment the front reaches it. Its parameters are the two slips
and the rise time of each patch, a rupture velocity per patch or per group of patches, and the
hypocenter, along strike and down dip in the surface coordinates of the mesh; like the model on
rectangles, it runs on the GPU or the cpu, with the exact gradient.

`mesh`
: the patches, an `.h5` file as CSI's `saveKinematic` writes it: `vertices` (km, depth positive),
  their surface coordinates `uv`, the `faces`, and the source points of each patch,
  `sources/uv` and `sources/patch`, the same number for every patch.

`refine`
: the triangles of the mesh of the arrival times along each edge of a patch; default 10.

`velocity_groups`
: a dataset of `mesh` with the group of each patch, numbered from 0; the patches of a group share
  one rupture velocity. By default, a velocity per patch.

`green`, `Nt`, `dt`, `t0s`
: as for the model on rectangles; the rows of the Green's functions are the time intervals, then
  the two slips, then the patches.

### The source time function

Both kinematic models release the slip of each source point over its rise time $T_r$ with a source
time function of unit area, sampled at the start of each time interval:

`stf`
: `triangle`, the default, or `cosine`, a raised cosine, $(1 - \cos 2\pi s/T_r)/T_r$, smooth.

`peak`
: where the triangle peaks, as a fraction of the rise time: 0.5, the default, for a symmetric
  triangle; less for a fast rise and a slow decay, closer to the slip rates of dynamic ruptures.

With waveforms bandpassed at periods longer than the rise times, the data constrain the time of
each patch's slip better than its shape; the shape matters with broadband records.

### Time shifts of the stations

Errors of the velocity model shift the arrivals at each station, and a model without them would
absorb the shifts in the hypocenter, the rupture velocities and the slips. A time shift per
station, a parameter set of its own sampled with the others, takes them up instead:

`shifts`
: the number of stations; their shifts, in s, positive later, are the last `shifts` parameters
  of the model. 0, the default, for none.

`traces`
: the traces of each station, e.g. its three components, which share its shift; default 3.

`sampling`
: the sampling interval of the traces, in s; by default `dt`.

The observations must be the traces of the stations, station by station, each trace the same
length; each is shifted by cubic interpolation, with zeros beyond its ends. The data covariance
must be a constant, `dataobs.cd_std`. In an ensemble:

```none
psets_list = [across, parallel, risetime, rupturevelocity, hypo_strike, hypo_dip, shift]
psets:
    shift = contiguous
    shift:
        count = 18
        prior = gaussian
        prior.sigma = 3
models:
    kinematic:
        psets_list = [across, parallel, risetime, rupturevelocity, hypo_strike, hypo_dip, shift]
        shifts = 18
        traces = 3
```

On a synthetic whose records were delayed by up to 4.6 s, the inversion without shifts collapsed
on a wrong model; with them, it recovered the slip, the hypocenter and the velocities as well as
from the undelayed records, and the delays to 0.1 s.

### Arrival times

`altar.models.seismic.arrivals` fits the arrival times of a phase, e.g. P picks, from the
hypocenter on the mesh: it maps the hypocenter from the surface coordinates into space through its
triangle, and computes the travel times along straight rays in a homogeneous medium, or from a
table. In an ensemble with the kinematic model, the two share the hypocenter, which the picks pin
down apart from the rupture velocities, its common trade-off with them:

```none
models:
    arrivals = altar.models.seismic.arrivals
    arrivals:
        cascaded = True
        psets_list = [hypo_strike, hypo_dip]
        mesh = mesh.h5
        stations = stations.txt
        velocity = 6.5
        dataobs:
            observations = 18
            data_file = picks.h5
            cd_std = 0.2
```

`mesh`
: the mesh of the kinematic model.

`stations`
: the station of each pick, x and y in km in the frame of the mesh, and optionally its elevation;
  a text or an `.h5` file.

`velocity`
: the velocity of the phase along straight rays, in km/s; default 6.

`table`
: an `.h5` file with the travel times, `times`, on the grid of `depths` and `distances` (km), for a
  layered earth; it overrides `velocity`.

`origin`
: the time of the origin of the rupture in the time of the picks, in s; default 0.

## The cascaded inversion

The example, `cascaded.pfg`, sets up the ensemble:

```{literalinclude} ../../models/seismic/examples/cascaded.pfg
:language: none
:caption: cascaded.pfg (the model)
:start-at: "    ; an ensemble of models sharing one set of parameters"
:end-before: "    controller = altar.bayesian.catmip"
```

The model is an ensemble, `altar.models.ensemble`, which owns the parameter sets: it draws the
initial samples and evaluates the priors. Each of its `models` names, in its own `psets_list`, the
parameter sets it uses, in the order it expects them, and computes its data likelihood from those
columns and its own data. A model with `cascaded = True` contributes at $\beta = 1$ throughout;
the others are annealed. Otherwise each model is configured as when it runs alone.

For the cascading scheme:

1. Run the static inversion:

   ```bash
   slipmodel.plexus sample --config=static.pfg
   ```

   which saves its posterior to `results/static/step_final.h5`.
2. Point the `preset` preps of the slips to it (the example uses `9patch/theta_cascaded.h5`, a
   static posterior that comes with it):

   ```none
   strikeslip:
       prep = preset
       prep.input_file = results/static/step_final.h5
       prep.dataset = ParameterSets/strikeslip
   ```

   With fewer static samples than chains, the samples are reused.
3. Run the ensemble:

   ```bash
   slipmodel.plexus sample --config=cascaded.pfg
   ```

   which saves its results to `results/cascaded`.

For the non-cascading scheme, skip the static inversion: set `cascaded = False` for the static
model too, and draw the slips from distributions rather than `preset`, e.g. as in `static.pfg`. It
usually takes many more steps to converge.

### Slips as a magnitude and a rake

The two slips of each patch can be sampled as a magnitude and a rake instead, with the ensemble's
`polar`: its parameter sets of the magnitudes and of the rakes, in degrees from the first
direction, then the names the models use for the two components, which the ensemble computes,
magnitude $\cos$(rake) and magnitude $\sin$(rake):

```none
model = altar.models.ensemble
model:
    polar = [magnitude, rake, parallel, across]
    psets_list = [magnitude, rake, risetime, rupturevelocity, hypo_strike, hypo_dip]
    psets:
        magnitude:
            prior = uniform
            prior.support = (0, 15)
            prior.reparameterize = True
        rake:
            prior = gaussian
            prior.mean = 0
            prior.sigma = 20
    models:
        static:
            psets_list = [across, parallel]
        kinematic:
            psets_list = [across, parallel, risetime, rupturevelocity, hypo_strike, hypo_dip]
```

A prior on the rake ties the slip across the main direction to the magnitude, where independent
priors on the two components give every patch the same spread across it, however little it
slips; the magnitude, and the moment from it, are then no longer inflated by that spread. On a
synthetic of the Pedernales earthquake, the moment magnitude from the slips came out 0.06 too
large with independent components and 0.01 with polar slips, the error of the slips halved, and
the evidence favored the polar slips. A static model alone takes polar slips as the single model
of an ensemble.

```{note}
Run the cascaded example in double precision (`job.gpuprecision = float64`, as it is set). In
single precision, its acceptance rate drops to zero part way through the annealing, and the run
stops when the covariance of the samples becomes singular; the kinematic model alone runs fine in
single precision.
```

## The check against the data

```bash
slipmodel.plexus forward --config=cascaded.pfg
```

runs the {ref}`forward check <forward-check>` for each model of the ensemble, on its own data:
the report, and the groups of the output file, `static/` and `kinematic/`, are per model. For the
kinematic model, the output also holds the slip history `Mb`, for the mean model and as the mean
and the standard deviation over the samples, e.g. to animate the rupture; it is arranged as the
rows of the Green's functions, $[N_t][2][N_{as}][N_{dd}]$, the last index running fastest.

`utils/meanModelKinematic.py` prints and saves the mean and the standard deviation of the
parameters of `step_final.h5`, like its {doc}`static counterpart <Static>`.
