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
eikonal equation, and holds $N_{pt} \times N_{pt}$ source points, each with its own rupture time;
the slip rate of the patch is the mean of the source time functions of its points, so that it
slips over the time the front takes to cross it (see {ref}`the forward model <kinematic-forward>`).
The predicted data are then

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
: the cells of the grid of the eikonal equation along each side of a patch.

`dsp`
: the length of a side of a patch, in km.

`Nt`, `dt`
: the number of time intervals, long enough to cover the rupture, and their length, in s.

`Npt`
: the source points along each side of a patch, $N_{pt} \times N_{pt}$ in all.

`t0s`
: a start time for each patch, added to the arrival time of the rupture; chosen well, they reduce
  the number of time intervals needed. By default 0.

`cp`, `cmu_file`, `kmu_file`
: the uncertainty of the Green's functions (see {doc}`StaticCp`); the kinematic kernels have the
  shape of the Green's functions.

(kinematic-forward)=
### The forward model

For each sample, the model computes $\mathbf M_b$ in four steps.

1. **Seeds.** The grid of the eikonal equation covers the fault padded by one patch on each side,
   $(N_{as}+2) N_{mesh} \times (N_{dd}+2) N_{mesh}$ cells of side $h = dsp/N_{mesh}$; the cells of
   the padding have the rupture velocity of the nearest patch. The cells within one cell of the
   hypocenter start at their straight-ray times, $r/V_r$, $r$ their distance to it; the others at
   times that grow steeply with the distance, $\big(r + 10 (r-h)^2/h\big)/V_r$, which the sweeps
   then lower. Unlike infinite times, these keep the arrival times differentiable in the
   hypocenter and the velocities.
2. **Fast sweeping.** One pass of four sweeps along the diagonals of the grid, one in each
   direction, updates each cell from the earlier of its two neighbors along strike, $a$, and of
   its two neighbors down dip, $b$, with $f = 1/V_r$ of its patch,

   $$
   T = \begin{cases}
   \min(a, b) + f h & \text{if } |a - b| \ge f h, \\
   \tfrac12 \big(a + b + \sqrt{2 f^2 h^2 - (a-b)^2}\big) & \text{otherwise},
   \end{cases}
   $$

   and keeps the earlier of the new and the old time. One pass is exact for a uniform velocity;
   strong contrasts, which make the front turn back, would need more.
3. **Source points.** The $N_{pt} \times N_{pt}$ source points of each patch are the centers of
   its sub-squares of side $dsp/N_{pt}$; their rupture times $T_q$ are interpolated bilinearly
   from the four nearest cells.
4. **Slip histories.** For patch $p$ and the time interval $i$, from $t_i = t_{0,p} + i\,dt$,

   $$
   c_i = \frac{dt}{N_{pt}^2} \sum_q S\big(t_i - T_q;\, T_{r,p}\big), \qquad
   \mathbf M_b(p, i) = \mathbf D_p\, c_i,
   $$

   with the source time function $S$ of unit area sampled at the start of the interval (see
   {ref}`below <kinematic-stf>`). $c_i$ is the share of the slip in the interval,
   $\sum_i c_i \approx 1$, and $\mathbf M_b(p, i)$ the slip of the patch in it, in m, as Green's
   functions of 1 m of slip in each interval, such as CSI's, expect.

   ```{note}
   Until November 2026, $\mathbf M_b$ held the slip rates, without the factor $dt$: with
   $dt = 1$ s the same, but with another $dt$, the slips of earlier runs are $dt$ times too large
   for Green's functions of 1 m of slip.
   ```

The gradient runs the same steps backward: the adjoints of the source time functions, of the
interpolation, of the sweeps, from the arrival times kept after each, and of the seeds.

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
  `sources/uv` and `sources/patch`, the same number for every patch: $N_{pt} \times N_{pt}$ with
  CSI's `setSourcePoints(Npt)`.

`Nmesh`
: the triangles of the mesh of the arrival times along each edge of a patch; default 10. Its
  former name, `refine`, still works.

`velocity_groups`
: a dataset of `mesh` with the group of each patch, numbered from 0; the patches of a group share
  one rupture velocity. By default, a velocity per patch.

`green`, `Nt`, `dt`, `t0s`
: as for the model on rectangles; the rows of the Green's functions are the time intervals, then
  the two slips, then the patches.

### The forward model on triangles

The same four steps, on a mesh of triangles. Each edge of a patch is cut into $N_{mesh}$ equal
segments, and the lines through their ends parallel to the edges cut the patch into $N_{mesh}^2$
equal triangles, $N_{mesh}(N_{mesh}+1)/2$ pointing as the patch and $N_{mesh}(N_{mesh}-1)/2$
upside down; the vertices on the edges that patches, or faults that meet, share are merged.

1. **Seeds.** The hypocenter is mapped from the surface coordinates onto the fault, through the
   affine map of the patch that holds it. The vertices near it start at their straight-ray
   times in space, with the fastest velocity of their patches, rising steeply past one spacing
   of the mesh as on rectangles; the others, at a cap.
2. **The fast iterative method.** The active vertices, at first the seeds, update from each of
   their triangles, with the velocity of its patch, and keep the earliest time; the neighbors of
   a vertex whose time drops become active, until a pass changes nothing.
3. **Source points.** CSI's `setSourcePoints(Npt)` cuts each patch the same way, into $N_{pt}^2$
   sub-triangles, and places a source point at the centroid of each; the sub-triangles have the
   same area, so the points share the slip of the patch equally. Their rupture times are
   interpolated linearly in the triangles of the mesh that hold them.
4. **Slip histories.** As on rectangles, the mean over the $N_{pt}^2$ source points of each patch.

### Choosing Nmesh, Npt and dt

With $L$ the size of a patch, $V_r$ the rupture velocity, $T_{r,min}$ the shortest rise time of the
prior, and $f_{max}$ the highest frequency of the data, for both models:

`dt`
: no longer than about $T_{r,min}/2$: the source time functions are sampled at the start of each
  interval, not averaged over it, and an interval as long as the rise time misses the shape of
  the function and its area. With Green's functions of a source time function $2\,dt$ long, as
  CSI's, $dt \approx 1/(4 f_{max})$ keeps the band of the data; shorter adds columns to
  $\mathcal G_b$, but no information. It must be a multiple of the sampling of the Green's
  functions.

`Nt`
: enough for $N_t\, dt$ to cover the latest rupture time, the longest rise time and the start
  times `t0s`; the cost of $\mathcal G_b \mathbf M_b$ grows with it.

`Npt`
: neighboring source points start about $L/(N_{pt} V_r)$ apart; when that is under about
  $T_{r,min}/2$, their source time functions overlap into a smooth slip rate of the patch,
  $N_{pt} \gtrsim 2L/(V_r T_{r,min})$. Once it is under $dt$ too, more points change little. The
  model costs little more with more points, but Green's functions that sum the point sources of
  each patch, as CSI's with `points=True`, cost $N_{pt}^2$ sources a patch.

`Nmesh`
: the error of the arrival times shrinks with the spacing of the mesh, $L/N_{mesh}$; the cost grows
  as $N_{mesh}^2$ on rectangles, closer to $N_{mesh}^3$ on triangles. At least $2 N_{pt}$, so that
  the times of neighboring source points differ by more than the error. To check it, compare the
  rupture times at $N_{mesh}$ and $2 N_{mesh}$, e.g. with `SlipHistory.arrivals` or CSI's
  `computeRuptureTimes`: if they differ by much less than $dt$, $N_{mesh}$ is enough. On the
  gmsh mesh of the Slab2 tutorial, those at 10 are 0.14 s later than at 40 on average, 0.30 s at
  most. Make the synthetics with the same `Nmesh` in CSI as in AlTar.

(kinematic-stf)=
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

### Slip histories outside of AlTar

`SlipHistory`, in `altar.models.seismic.history`, makes the slip histories of the model on
triangles without an application, e.g. for synthetics or least squares in other programs, such
as CSI's `TriangularPatchesKin.castbigM`. It runs the compiled model on the cpu:

```python
from altar.models.seismic.history import SlipHistory

history = SlipHistory(vertices, faces, uv, sources, patches, Nt=85, dt=1.0, Nmesh=10,
                      stf="triangle", peak=0.5)
Mb = history(slip, risetime, velocity, hypocenter)        # Nt x 2 x patches
onsets, times = history.arrivals(velocity, hypocenter)    # at the source points, at the vertices
```

The `slip` of each patch is along the two rakes of the Green's functions (patches x 2), the rise
times and rupture velocities one per patch, and the hypocenter in surface coordinates; with a
leading axis on each, it computes a batch of models at once. The data are then the big G times
`Mb`, raveled.

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

#### Choosing the rake's prior

Where its center comes from: the direction the fault is expected to slip.

- On a subduction interface, the plate convergence from a plate motion model (MORVEL, GSRM),
  projected onto each patch; the first direction of the Green's functions usually already follows
  it (csi's `convergenceRakes`), and the mean is then 0.
- On a strike-slip or crustal fault, pure strike slip along the trace, or the rake of the event's
  moment tensor (W phase, GCMT).

What sets its standard deviation: how far a patch may turn from that center.

- The scatter of the slip vectors of earlier events on the same fault, e.g. the interplate thrust
  mechanisms of GCMT around the rupture; on many megathrusts it is 10° to 20°.
- The obliquity of the convergence: where it is oblique, part of it may be taken up by a
  strike-slip fault in the forearc, and the interface slips off the convergence; shift the mean or
  widen the sd (30° or more).
- The event's own moment tensor: its rake against the center is a lower bound on the spread to
  allow.
- What the data resolve: where the fault slips a lot, the data set its rake and the sd matters
  little; where it barely slips, its rake is the prior's.

The slip across the center that the prior allows grows with the magnitude, about 0.34 m per meter
of slip at 20°, a few centimeters on a patch that slips 0.1 m. Run two or three sds, e.g. 10°, 20°
and 40°: with the same data, their evidences compare them, and a slip or rake that moves with them
is set by the prior, not by the data. The 20° above is generous for the Pedernales synthetic,
whose rakes stay within a few degrees of the convergence where it slips.

```{note}
Run the cascaded example in double precision (`job.gpuprecision = float64`, as it is set). In
single precision, its acceptance rate drops to zero part way through the annealing, and the run
stops when the covariance of the samples becomes singular; the kinematic model alone runs fine in
single precision.
```

### The noise scale of each data set

The covariance of each data set fixes its weight against the others. When it is uncertain, e.g.
the errors of the earth model in the waveforms against those of the InSAR, the ensemble can sample
a scale for it with `noise`: per model, a parameter set of one parameter $h$, used by no model,
that scales its covariance by $e^{2h}$, its standard deviations by $e^h$:

```none
model = altar.models.ensemble
model:
    psets_list = [strikeslip, dipslip, risetime, rupturevelocity, hypo_strike, hypo_dip, h_static, h_kinematic]
    noise:
        static = h_static
        kinematic = h_kinematic
    psets:
        h_static = contiguous
        h_kinematic = contiguous
        h_static:
            count = 1
            prior = gaussian
            prior.mean = 0
            prior.sigma = 1
        h_kinematic:
            count = 1
            prior = gaussian
            prior.mean = 0
            prior.sigma = 1
```

With $N$ observations and $\chi^2$ under its covariance, a model's data likelihood becomes
$c - N h - e^{-2h} \chi^2 / 2$, so $h$ settles near $\frac{1}{2} \log(\chi^2 / N)$: a data set
the model fits worse than its covariance says gets a larger $h$ and less weight. The scale
applies to the covariance in effect, $C_d$ or $C_\chi = C_d + C_p$ with a model uncertainty. A
gaussian prior on $h$ needs no reparameterization for HMC; a uniform one does. With HMC, the
gradient by $h$ takes a forward run of the model beyond its own gradient. The `optimize` action
takes the noise scales as fixed values: list them in its `fixed`.

On a synthetic of GPS and InSAR on a curved slab, with the InSAR noise three times the standard
deviation its covariance stated, the stated covariances let the InSAR warp the slips: their error
was 1.8 times that of a run with the true standard deviations, and only 78% of them were within
one standard deviation of the truth, against 99%. With a noise scale per data set, $h$ came out
$1.05 \pm 0.08$ for the InSAR, against $\log 3 = 1.10$, and $-0.07 \pm 0.10$ for the GPS, and
the slips were as good as with the true standard deviations. The evidence favored the scales over
the stated covariances by 117, and the true covariances over the scales by 23, the price of two
more parameters.

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
