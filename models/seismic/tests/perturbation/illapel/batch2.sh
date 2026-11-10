#!/bin/bash
# longer hmc and catmip_hmc runs, to see what they need to reach the exact posterior
here=$HOME/scripts/altar/perturbation
gpu=${GPU:-1}
hmc="--controller.sampler.leapfrog_steps=10 --controller.sampler.step_size=0.01"
bash $here/run.sh $gpu gauss_hmc_r10x100_b5 --controller=altar.bayesian.hmc $hmc --job.steps=100 --controller.rounds=10 --controller.scheduler.burnin=5
bash $here/run.sh $gpu gauss_hmc_r20x100_b5 --controller=altar.bayesian.hmc $hmc --job.steps=100 --controller.rounds=20 --controller.scheduler.burnin=5
bash $here/run.sh $gpu gauss_hmc_r40x100_b5 --controller=altar.bayesian.hmc $hmc --job.steps=100 --controller.rounds=40 --controller.scheduler.burnin=5
bash $here/run.sh $gpu gauss_catmip_hmc_t50 --controller=altar.bayesian.catmip_hmc $hmc --job.steps=50
bash $here/run.sh $gpu gauss_catmip_hmc_t200 --controller=altar.bayesian.catmip_hmc $hmc --job.steps=200
source $here/env.sh; cd $here && python compare.py gauss_hmc_r5x100_b3 gauss_hmc_r10x100_b5 gauss_hmc_r20x100_b5 gauss_hmc_r40x100_b5 gauss_catmip_hmc_t20 gauss_catmip_hmc_t50 gauss_catmip_hmc_t200
