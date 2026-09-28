(model-uncertainty-cp)=
# Model Uncertainty: $C_p$

## Epistemic uncertainties

Besides the errors of the observations, the forward model itself is uncertain: it rests on an
imperfect knowledge of the Earth, e.g. of its elastic structure (Duputel et al., 2014) or of the
geometry of the fault (Ragon et al., 2018, 2019). These *epistemic* uncertainties (Minson et al.,
2013) can be accounted for by assuming that the true displacements scatter around the predictions,
with a prediction covariance $C_p$ that depends on the source model. The data likelihood then uses
the combined covariance

$$
C_\chi = C_d + C_p
$$

in place of the data covariance $C_d$ alone.

For the static model, with uncertain inputs $\boldsymbol\mu$ (e.g. the elastic properties) of
covariance $C_\mu$, the prediction covariance for a source model $\boldsymbol\theta$ is

$$
C_p = K_p\, C_\mu\, K_p^T, \qquad
K_p[:, i] = \mathbf K_i \boldsymbol\theta, \qquad
\mathbf K_i = \frac{\partial \mathbf G}{\partial \mu_i},
$$

where the *sensitivity kernels* $\mathbf K_i$ are the derivatives of the Green's functions with
respect to each uncertain input. $C_p$ grows with the slip: it is largest where the model predicts
large displacements.

## Turning $C_p$ on

Any model can take a $C_p$; how it is used is chosen with the model's `cp`:

`altar.models.cp.none`
: no $C_p$: $C_\chi = C_d$. The default.

`altar.models.cp.fixed`
: a $C_p$ computed beforehand, added to $C_d$ once, before the sampling starts:
  - `cp_file`: the $N_{obs} \times N_{obs}$ matrix, among the model's input files;
  - `dataset`: its dataset in an `.h5` file; by default the first one.

`altar.models.cp.adaptive`
: a $C_p$ the model estimates again at each $\beta$ step, from the mean of the current samples, as
  the source model firms up:
  - `start`: include $C_p$ once $\beta$ reaches this value; default 0;
  - `initial_model`: a source model, among the model's input files, to estimate $C_p$ from early
    on ...
  - `initial_until`: ... while $\beta$ is at most this value; default 0.

  The model decides how $C_p$ follows from a mean model; the static model uses the sensitivity
  kernels above, from its settings
  - `cmu_file`: $C_\mu$, an $n \times n$ matrix for the $n$ uncertain inputs;
  - `kmu_file`: the kernels $\mathbf K_i$, an `.h5` file with one $N_{obs} \times N_{param}$
    dataset per uncertain input, in the order of $C_\mu$ (by name, with numbers in numerical
    order: `kernel2` before `kernel10`).

  Each update is logged, with its $\beta$ and the trace of $C_p$. Under MPI, the mean is taken
  over the samples of all processes.

The example `staticCp.pfg`, in `models/seismic/examples`, re-estimates $C_p$ from the uncertainty
of the elastic structure:

```{literalinclude} ../../models/seismic/examples/staticCp.pfg
:language: none
:caption: staticCp.pfg (the model)
:start-at: "    ; the static model"
:end-before: "        ; list of parametersets"
```

To compare with and without $C_p$, turn it off on the command line:

```bash
slipmodel.plexus sample --config=staticCp.pfg --slipmodel.model.cp=altar.models.cp.none
```

`staticCp_logit.pfg` samples the same problem with HMC, which also works with $C_p$: the gradient
of the data likelihood uses $C_\chi$.

```{note}
With a matrix $C_d$, or with $C_p$, the covariance is factored as a dense
$N_{obs} \times N_{obs}$ matrix, once, or at each $\beta$ step with an adaptive $C_p$. For many
observations, factor it in double precision (`dataobs.cd_dtype = float64`), and mind the memory.
```

## The check against the data

With a $C_p$, the {ref}`forward check <forward-check>` estimates $C_p$ at the posterior mean,
reports the uncertainty of the data both without and with it, and draws its band around the
predictions from $C_\chi$; the output file holds `data/sigma` ($\sqrt{\operatorname{diag} C_d}$)
and `data/sigma_chi` ($\sqrt{\operatorname{diag} C_\chi}$).

## References

1. Minson, S. E., Simons, M., and Beck, J. L., *Bayesian inversion for finite fault earthquake
   source models I — theory and algorithm*, Geophysical Journal International, 194, 1701 (2013).
2. Duputel, Z., Agram, P. S., Simons, M., Minson, S. E., and Beck, J. L., *Accounting for
   prediction uncertainty when inferring subsurface fault slip*, Geophysical Journal
   International, 197, 464 (2014).
3. Ragon, T., Sladen, A., and Simons, M., *Accounting for uncertain fault geometry in earthquake
   source inversions — I: theory and simplified application*, Geophysical Journal International,
   214, 1174 (2018).
4. Ragon, T., Sladen, A., and Simons, M., *Accounting for uncertain fault geometry in earthquake
   source inversions — II: application to the Mw 6.2 Amatrice earthquake, central Italy*,
   Geophysical Journal International, 218, 689 (2019).
