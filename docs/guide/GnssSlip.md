(gnss-slip)=
# Time-dependent slip from GNSS time series

The model `altar.models.gnss.slip` infers how slip evolves on a fault from the daily position time
series of a GNSS network: slow slip events, afterslip, or any transient slip that a linear
combination of temporal basis functions can describe. It is linear and Gaussian, so the
{ref}`perturbation sampler <perturbation>` samples it exactly. The posterior has a parameter per
patch, rake and basis function, typically $10^5$ or more, so the sampler works on the
{ref}`matrix-free path <perturbation-matrix-free>`, on the CPU or the GPU.

## The model

The data are the time series of $B$ blocks, each one component (east, north or up) of one station,
on a common daily time axis. A block's trajectory is a sum of groups of basis functions, as in
the trajectory model of `altar.models.gnss`: an offset, a trend, seasonal terms, steps at known
epochs, and transients, e.g. I-splines (integrated B-splines, smooth steps) at several time
scales. The model splits the groups into two kinds:

- the **slip groups**, shared by all the blocks, are the temporal basis of the slip. Their
  coefficients at block $b$ are

  $$
  \boldsymbol\theta_b = (\mathbf G \mathbf X + \boldsymbol\Xi)_b,
  $$

  with $\mathbf X$, $(2N \times J)$, the slip of the $N$ patches (dip slip in the first $N$ rows,
  strike slip in the next $N$) on the $J$ basis functions; $\mathbf G$, $(B \times 2N)$, the
  elastic Green's functions; and $\boldsymbol\Xi$, $(B \times J)$, an optional noise correlated
  between stations (see below);
- the **nuisance groups**, everything else (offset, trend, seasonal, steps, ...), have Gaussian
  priors and are integrated out exactly, block by block.

The noise of a block is white, with the standard deviations of the case file, plus optional
exponentially correlated processes,
$C_b(t, t') = w_b^2 \sigma_b(t)^2 \delta_{tt'} + \sum_m a_{bm}^2 e^{-|t - t'| / \tau_m}$. With the
nuisance integrated out, each block contributes

$$
\log P(\mathbf d_b | \boldsymbol\theta_b) = -\tfrac12 \boldsymbol\theta_b^T \mathbf M_b \boldsymbol\theta_b
+ \mathbf h_b^T \boldsymbol\theta_b + c_b,
$$

with $\mathbf M_b$ $(J \times J)$, $\mathbf h_b$ and $c_b$ computed once, from the Cholesky factor
of $C_b$ over the block's observed days. The likelihood therefore costs $O(B J^2)$ per sample,
however long the time series are.

The prior of $\mathbf X$ is Gaussian and separable, with precision $\mathbf P \otimes \mathbf D$:

- in space, $\mathbf P = \mathrm{blockdiag}(\mathbf C^{-1}, \mathbf C^{-1} / \rho^2)$, with
  $C_{ij} = e^{-d_{ij} / \ell}$ between the patch centroids, so that strike slip has $\rho$ times
  the prior standard deviation of dip slip;
- in time, $\mathbf D = \mathrm{diag}(1 / \sigma_j^2)$, with
  $\sigma_j = \sigma_0 (s_j / s_{\rm ref})^{p}$ for a basis function of time scale $s_j$, so that
  longer transients may carry more slip.

The optional noise $\boldsymbol\Xi$ takes up the part of a transient that is correlated between
nearby stations but is not slip on the fault, e.g. common-mode or hydrological signals. Per
component $c$, it has the matrix-normal prior
$N(0, \mathbf S_c, c_{0c} (\bar{\mathbf M}_c + \mathbf I / \mathrm{cap}^2)^{-1})$, with
$S_{c,ij} = e^{-d_{ij} / \ell_c}$ between the stations and $\bar{\mathbf M}_c$ the mean
$\mathbf M_b$ of the component. The fraction $c_{0c}$ of the noise becomes correlated, and the
independent noise of the component is scaled by $1 / (1 - c_{0c})$.

The posterior at $\beta$ is $N(\mathbf A^{-1} \mathbf b, \mathbf A^{-1})$. The model applies
$\mathbf A$ without forming it, at the cost of products with $\mathbf G$, the $\mathbf M_b$ and
$\mathbf P$. It preconditions the conjugate gradients with a Kronecker approximation of
$\mathbf A$, which replaces every $\mathbf M_b$ by a multiple of their mean, so the solves take
tens of iterations.

## Input files

All the files are in the directory `case`.

`design_file`, the case file (hdf5, format `altar.gnss.trajectory/1`, see
`altar.models.gnss.design`)
: the common `time` axis (days); the blocks (`blocks/station`, `blocks/component`, `blocks/lon`,
  `blocks/lat`); the whitened observations `data` and their `mask`; `raw` and `sigma`; and one
  entry `groups/NAME` per group of basis functions, with its kind (`polynomial`, `sinusoid`,
  `isplines`, `steps`, `logarithmic`) and settings. `altar.models.gnss.design` builds and writes
  one from arrays.

`greens_file` (hdf5)
: `stations`, the $S$ station names; `gf`, $(S \times 3 \times 2 \times N)$, the displacement
  (east, north, up) at each station of unit strike slip and unit dip slip on each patch, in the
  units of the case; `centroids_km`, $(N \times 3)$, the patch centroids in km, for the
  distances of the prior. Any elastic model will do: half-space or layered, rectangles or
  triangles. Stations of the case without Green's functions are an error.

`noise_file` (json, optional)
: the correlated noise of the blocks, `{"taus_days": [...], "blocks": [{"station", "component",
  "w", "amps_mm": [...]}, ...]}`, one entry per block in the order of the case; a `null` entry
  takes its component's medians. Without it, the noise is white.

`spatial_file` (json, optional)
: `{"east": {"c0": ..., "ell_km": ...}, "north": ..., "up": ...}`, the noise correlated
  between stations. Without it, there is no $\boldsymbol\Xi$.

## Configuration

```none
gnss:
    model = altar.models.gnss.slip
    model:
        case = .
        design_file = trajectory.h5
        greens_file = greens.h5
        noise_file = noise.json      ; optional
        spatial_file = spatial.json  ; optional
        slip_groups = []             ; the slip's basis; empty: every I-spline group
        nuisance:                    ; prior sd of the other groups, by name, prefix or kind
            offset = 10
            trend = 1
            seasonal = 1
            steps = 2
        sigma0 = 10                  ; prior sd of the dip slip at scale scale_ref
        scale_ref = 16               ; days
        scale_power = 0.5
        length = 30                  ; km, correlation length of the slip prior
        rho = 0.3                    ; strike slip prior sd / dip slip prior sd
        cap = 10                     ; Xi: the prior sd where the data say nothing
        dataobs:
            observations = 223608    ; blocks x days of the case
            data_file = trajectory.h5
            datafile_dataset = data
            mask_dataset = mask
            cd_std = 1.0
        psets_list = [slip, noise]   ; noise only with spatial_file
        psets:
            slip = contiguous
            noise = contiguous
            slip:
                count = 167904       ; 2 N J
                prep = gaussian
                prior = gaussian     ; replaced by the structured prior above
            noise:
                count = 66528        ; B J
                prep = gaussian
                prior = gaussian
    controller = altar.bayesian.perturbation
    job.chains = 200
    job.gpus = 1                     ; 0: on the CPU
```

The model checks the counts against the case and stops with the expected values if they don't
match. The `prior` of the parameter sets is ignored: the prior is the structured one above.
The slip has the units that make $\mathbf G \mathbf X$ a displacement in the units of the case:
with the data in mm and the Green's functions in mm per mm of slip, the slip is in mm.

## Running

```bash
altar-gnss --config=slip.pfg
```

With `job.gpus = 1`, the conjugate gradients and the perturbations run on the GPU (cuBLAS and
kernels compiled at run time by NVRTC), in double precision, all the chains at once; the
densities of the samples and the setup stay on the host. The device needs about
$8\,(6 \cdot 2NJ + 7 \cdot BJ) \times$ `chains` bytes, 2.4 GB in the case below. The GPU pays
off on large cases; a small one, like the afterslip example, runs as fast on the CPU.

One walk gives `chains` independent, exact samples of the posterior; there is nothing to tune
or to check for convergence besides the residual of the solves, which the log reports. A case
of 924 blocks (308 stations, 242 days), 1166 triangles and 72 I-splines at 16, 32 and 64 days,
with noise correlated in time and between stations, has 234 432 parameters. Its mean takes 70
iterations, and 200 samples 65. The whole run takes 2 min 43 s on 6 CPU threads, and 37 s on one
GPU, where the solves of the samples take 13 s and the rest is on the host: the setup and the
densities of the samples.

The archive (`results/step_final.h5`) holds the samples of each parameter set,
`ParameterSets/slip_sampling`, (chains $\times\, 2NJ$, $\mathbf X$ row major), and
`ParameterSets/noise_sampling`. The slip of a sample at day $t$ is $\mathbf X \mathbf g(t)$,
with $\mathbf g(t)$ the values of the slip groups' basis functions at $t$ (the columns of their
design matrices, in the order of `slip_groups`).

## Choosing the basis

The slip groups are the slip's only freedom in time, so they decide what the model can find:

- the I-splines should start where slip may start. For slip that follows a known event, e.g.
  afterslip, start them at the event, with short scales early on. A basis spanning the whole
  window, before the event too, trades off against the trend and the coseismic step;
- the nuisance priors should be wide enough for the data to set the offsets, trends and steps,
  but not so wide that the trend absorbs a slow transient; a stretch of data before the
  transient pins them down;
- the time scales set the shortest event the model resolves: a scale of $s$ days cannot fit
  slip faster than about $s$ days.

The prior's `length`, `sigma0` and `rho` are hyperparameters: they have to be chosen, e.g. by
synthetic tests or by cross-validation, since the matrix-free path gives no evidence to compare
them.

## Example: afterslip on a strike-slip fault

`models/gnss/examples/afterslip` is a synthetic case unlike the subduction slow slip the model
was first used for. It has a vertical strike-slip fault of 150 triangles and 25 stations over
two years, with a 1 m coseismic rupture on day 365 followed by a year of logarithmic afterslip
below it:

```bash
cd models/gnss/examples/afterslip
python make_case.py          # trajectory.h5, greens.h5 (cutde), noise.json, truth.npz
altar-gnss --config=afterslip.pfg
python compare.py            # the samples against the truth
```

The 55 200 parameters sample in about 6 s on the CPU. From two weeks after the event, the
moment of the afterslip is within 25% of the truth and inside the 90% intervals; five days after
it, the early afterslip is underestimated. The 25 stations resolve the spatial pattern of the
afterslip only in part: the correlation of the mean final strike slip with the truth is about
0.5.

## Limits

- No bounds: the slip may take either sign on each rake. Positivity would need the bounded
  sampler, which factors $\mathbf A$.
- No evidence on the matrix-free path, hence no Bayesian comparison of hyperparameters.
- The slip groups must be shared by all the blocks (same basis functions everywhere).
- The preconditioner assumes that the $\mathbf M_b$ are alike up to scale; very uneven data
  (e.g. a few continuous stations among many short campaigns) need more iterations.
