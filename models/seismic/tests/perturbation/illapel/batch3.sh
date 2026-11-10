#!/bin/bash
# catmip_hmc with the dip slips started from their prior, for the evidence
here=$HOME/scripts/altar/perturbation
gpu=${GPU:-1}
hmc="--controller.sampler.leapfrog_steps=10 --controller.sampler.step_size=0.01"
prior="--model.psets.dipslip.prep=gaussian --model.psets.dipslip.prep.mean=0 --model.psets.dipslip.prep.sigma=5.0"
bash $here/run.sh $gpu gauss_catmip_hmc_t50_prior --controller=altar.bayesian.catmip_hmc $hmc --job.steps=50 $prior
bash $here/run.sh $gpu gauss_hmc_r20x100_b5_prior --controller=altar.bayesian.hmc $hmc --job.steps=100 --controller.rounds=20 --controller.scheduler.burnin=5 $prior
source $here/env.sh; cd $here && python compare.py gauss_catmip_hmc_t50_prior gauss_hmc_r20x100_b5_prior
