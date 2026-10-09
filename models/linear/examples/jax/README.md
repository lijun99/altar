# The linear model in jax

An example of a model whose forward model is written in [jax](https://jax.readthedocs.io), shown
here for illustration: it is not part of the AlTar package, which doesn't depend on jax.

- `JaxModel.py`: a mixin for `BayesianL2` models. A model defines `jax_forward(theta)`, the
  predicted data of one sample in `jax.numpy`; the mixin batches it over the chains, and the data
  gradient is `jax.grad` of the same L2 likelihood that `dataobs` computes. On the GPU, jax reads
  the samples from AlTar's managed memory in place, through `__cuda_array_interface__`.
- `LinearJax.py`: the linear model with `jax_forward = G @ theta`.

To run it, with jax installed (`pip install jax`, or `"jax[cuda13]"` for the GPU), from the
linear examples:

```bash
cd models/linear/examples
PYTHONPATH=jax altar-linear --config=linear.pfg --model=import:LinearJax.LinearJax
PYTHONPATH=jax altar-linear --config=linear_hmc.pfg --model=import:LinearJax.LinearJax --job.gpus=1
```
