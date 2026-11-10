#!/bin/bash
# the Illapel static runs with gaussian priors, one after another on one gpu
here=$HOME/scripts/altar/perturbation
gpu=${GPU:-1}
hmc="--controller.sampler.leapfrog_steps=10 --controller.sampler.step_size=0.01"
bash $here/run.sh $gpu gauss_perturbation
bash $here/run.sh $gpu gauss_cf_s100 --controller=altar.bayesian.cf_catmip --job.steps=100
bash $here/run.sh $gpu gauss_cf_s1000 --controller=altar.bayesian.cf_catmip --job.steps=1000
bash $here/run.sh $gpu gauss_hmc_r5x100_b3 --controller=altar.bayesian.hmc $hmc --job.steps=100 --controller.rounds=5 --controller.scheduler.burnin=3
bash $here/run.sh $gpu gauss_catmip_hmc_t20 --controller=altar.bayesian.catmip_hmc $hmc --job.steps=20
