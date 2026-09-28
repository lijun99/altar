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
The kinematic model runs only on the GPU (`job.gpus = 1`): its fast sweeping is too expensive for
the cpu. It doesn't provide the gradient of its data likelihood, so it samples with Metropolis,
not HMC.
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
