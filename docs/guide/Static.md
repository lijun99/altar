(static-inversion)=
# Static Slip Inversion

## The static source model

A finite fault earthquake source model infers the slip on the fault from the displacements it
caused at the surface, or below it. The *static* model infers the final slip, not its evolution
in time.

The fault is divided into patches, each treated as a point source with two slip components, along
the strike and along the dip. Each slip causes a surface displacement, e.g. by the Okada model, a
Green's function solution of the elastic half-space problem, and the displacement observed at a
location is the sum of those caused by the slips of all patches. The forward model is linear:

$$
\mathbf d^{pred} = \mathbf G \boldsymbol\theta,
$$

where $\boldsymbol\theta$ (often $\mathbf m$ in the geophysics literature) holds the
$N_{param} = 2 N_{patch}$ slips, along the strike and the dip of each patch; $\mathbf d$ holds the
$N_{obs}$ observed displacements, e.g. vertical, east and north components at several locations;
and $\mathbf G$ is the $N_{obs} \times N_{param}$ matrix of Green's functions, computed
beforehand, connecting each slip to each observed displacement.

The model can include other linear parameters, e.g. InSAR ramps: the parameters $(a, b, c)$ of a
spurious ramp $a + bx + cy$ in an interferogram, where $x$ and $y$ are the local coordinates of the
data.

```{note}
For the static inversion, the patches can be of any shape and size, as long as each can be treated
as a point source. The {doc}`kinematic model <Kinematic>` needs a rectangular fault of
$N_{dd} \times N_{as}$ square patches: for a joint static-kinematic inversion, use these patches
for the static inversion too.
```

The static model is the {doc}`linear model <QuickStart>` with these parameters: it runs on the cpu
and on the GPU, and it provides the gradient of its data likelihood, for HMC. It can also account
for the uncertainty of its Green's functions, $C_p$ (see {doc}`StaticCp`).

## Input

The inversion needs three input files, all in the directory `model.case`:

the data
: the $N_{obs}$ observed displacements, a vector.

the data covariance
: the uncertainties of the data, an $N_{obs} \times N_{obs}$ matrix, with off-diagonal terms for
  correlated errors. For uncorrelated data with a common standard deviation, give it instead with
  `dataobs.cd_std`.

the Green's functions
: an $N_{obs} \times N_{param}$ matrix, stored row by row: for each observation, its Green's
  functions for each parameter, in the order of the parameters in $\boldsymbol\theta$,

  ```none
  G[obs1][param1] G[obs1][param2] ... G[obs1][paramN]
  G[obs2][param1] G[obs2][param2] ... G[obs2][paramN]
  ...
  ```

The files can be text (`.txt`), raw binary (`.bin` or `.dat`, in the precision of the
computation, reshaped as needed), or HDF5 (`.h5`, the recommended format, which carries the shape
and the precision); the suffix tells which. Their names are free: they are given in the
configuration. Software such as [CSI](http://www.geologie.ens.fr/~jolivet/csi/) computes the
Green's functions and prepares the input files for AlTar.

## Configuration

The example, in `models/seismic/examples`:

```{literalinclude} ../../models/seismic/examples/static.pfg
:language: none
:caption: static.pfg
:lines: 15-
```

### The application

The seismic models run with `slipmodel.plexus`, whose root in the configuration is `slipmodel`:

```bash
slipmodel.plexus sample --config=static.pfg
```

samples the posterior, and

```bash
slipmodel.plexus forward --config=static.pfg
```

checks it against the data (see {ref}`below <static-forward>`). Without `--config`, pyre reads
`slipmodel.pfg` from the current directory.

### The model

`model = altar.models.seismic.static` selects the static model. Its settings are

`case`
: the directory with the input files.

`patches`
: the number of patches; the parameter sets must hold `2 * patches` parameters.

`green`
: the file with the Green's functions.

`dataobs`
: the observed data, and their covariance:
  - `observations`: the number of observations;
  - `data_file`: the file with the observed data;
  - `cd_file`: the file with the data covariance, or
  - `cd_std`: a common standard deviation for all observations, instead of `cd_file`;
  - `cd_dtype`: the precision for factoring the covariance, if it should differ from
    `job.gpuprecision`, e.g. `float64` for a large or ill-conditioned covariance.

`psets_list`, `psets`
: the parameter sets, in the order of the columns of the Green's functions (see
  {ref}`below <static-parameter-sets>`).

`cp`, `cmu_file`, `kmu_file`
: the uncertainty of the Green's functions (see {doc}`StaticCp`).

(static-parameter-sets)=
### Parameter sets

The parameters are grouped in parameter sets, laid out in $\boldsymbol\theta$ in the order of
`psets_list`, which must match the order of the columns of the Green's functions: e.g., with 9
patches and the 3 parameters of an InSAR ramp,

```none
psets_list = [strikeslip, dipslip, ramp]
```

makes $\boldsymbol\theta$ the 9 strike slips, then the 9 dip slips, then the 3 ramp parameters. The
names are free. Each set has a `count`, a `prior`, and optionally a `prep` for its initial samples
(see {doc}`Priors`). Typical choices are a Gaussian prior centered at 0 for the strike slips of a
dip-slip fault,

```none
strikeslip:
    count = {slipmodel.model.patches}
    prior = gaussian
    prior.mean = 0
    prior.sigma = 0.5
```

the moment magnitude prior for the dip slips (see {ref}`below <moment-distribution>`), and a uniform or
Gaussian prior for ramps:

```none
ramp:
    count = 3
    prior = uniform
    prior.support = (-0.5, 0.5)
```

To give different patches different priors, e.g. narrower ranges far from the hypocenter, split a
slip component into several parameter sets, each with its own `count` and prior.

### Controller and job

The example anneals with CATMIP and Metropolis sampling, on the cpu; add `--job.gpus=1` to run it
on the GPU. `static_logit.pfg` samples with HMC instead, which needs the bounded dip slip prior
reparameterized (`reparameterize = True`, see {doc}`Priors`). See {doc}`AlTarFramework` for the
controllers, and for running on several processes or GPUs.

(moment-distribution)=
## The moment magnitude prior

The slips of an earthquake should add up to its seismic moment, which the moment magnitude
estimates:

$$
M_w = \frac{2}{3} \left(\log_{10} M_0 - 9.1\right), \qquad M_0 = \sum_{p=1}^{N_{patch}} \mu_p A_p D_p,
$$

where $M_0$ is the scalar seismic moment (in N·m), $\mu_p$ the shear modulus of the rocks, and
$A_p$ and $D_p$ the area and the slip of patch $p$.

`altar.models.seismic.moment` is a uniform prior on the slips, with two additions:

- its initial samples are consistent with a moment magnitude: it draws $M_w$ from a Gaussian
  $N(\bar M_w, \sigma_{M_w})$, spreads the corresponding $M_0 / \mu$ over the patches with a flat
  Dirichlet distribution, and divides by the patch areas to get slips; samples with a slip outside
  the support are drawn again;
- optionally, the moment constraint adds a Gaussian penalty on the moment magnitude of each sample
  to its log prior, $-f\,(M_w - \bar M_w)^2 / (2\sigma_{M_w}^2)$, which keeps the posterior close to
  the estimated magnitude.

`support`
: the range of the slips, in m, as for the uniform prior.

`Mw_mean`, `Mw_sigma`
: $\bar M_w$ and $\sigma_{M_w}$.

`Mu`
: the shear modulus of each patch, in GPa; one value for all patches, e.g. `[30]`.

`area`
: the area of each patch, in km²; one value for all patches, e.g. `[400]`, or one per patch. Only
  the products $\mu_p A_p$ matter, so e.g. `Mu = [1]` and the products as `area` work too.

`area_patch_file`
: a text file with the area of each patch, instead of `area`.

`slip_sign`
: `positive` (the default) or `negative`: the sign of the initial slips, along or against the
  strike or dip direction.

`moment_constraint`
: add the moment magnitude penalty to the prior; default `False`.

`moment_constraint_factor`
: the weight $f$ of the penalty; default 1.

`reparameterize`
: sample in an unbounded space, for HMC; see {doc}`Priors`.

```none
dipslip:
    count = {slipmodel.model.patches}
    prior = altar.models.seismic.moment
    prior:
        support = (-0.5, 20)
        Mw_mean = 7.3
        Mw_sigma = 0.2
        Mu = [30]
        area = [400]
        moment_constraint = True
```

It can also serve only as a `prep`, with another distribution as the prior.

(static-forward)=
## Output, and the check against the data

The results are saved by the archiver of the controller; the example writes HDF5 files to
`results/static` (see {doc}`QuickStart` for their layout). To see how well the posterior explains
the data, run

```bash
slipmodel.plexus forward --config=static.pfg
```

which runs the forward model on the posterior mean and on each posterior sample, reports how well
the predicted data fit the observed data, and saves the predictions and their spread to
`results/static/forward.h5` (see {ref}`the forward check <forward-check>`).

## Utilities

A few python scripts in `models/seismic/examples/utils` help to prepare the input files and to
look at the results.

### H5Converter

converts text or raw binary files to HDF5, the recommended input format; from the examples
directory,

```bash
utils/H5Converter --inputs=static.gf.txt
utils/H5Converter --inputs=kinematicG.gf.bin --precision=float32 --shape=[100,11000]
```

and merges several files into one, e.g. the sensitivity kernels for $C_p$:

```bash
utils/H5Converter --inputs=[static.kernel.pertL1.txt,static.kernel.pertL2.txt] --output=static.kernel.h5
```

See `utils/H5Converter --help` for all its options.

### plotBayesian

plots the histograms of the log prior, likelihood and posterior of an archived step, which tell
how well the sampling went; it needs `matplotlib`:

```bash
cd results/static
../../utils/plotBayesian                     # the final step
../../utils/plotBayesian --step=step_000.h5  # another step
../../utils/plotBayesian --bin=20            # with 20 bins
```

It saves the plots to `Bayesian_histograms.pdf`.

### meanModelStatic.py

prints the mean and the standard deviation of each parameter of `step_final.h5`, saves them to
`theta_mean.txt` and `theta_std.txt`, and converts the step to the format of AlTar 1.1,
`step_final_v1.h5`:

```bash
cd results/static
python3 ../../utils/meanModelStatic.py
```

For another step, or other parameter sets, edit the file names and `psets_list` in the script;
`meanModelKinematic.py` does the same for the kinematic model.
