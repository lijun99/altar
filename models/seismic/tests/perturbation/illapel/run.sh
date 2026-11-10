#!/bin/bash
# usage: run.sh <gpu> <name> [settings]: one run of static_gauss.pfg, timed, in the foreground
gpu=$1; name=$2; shift 2
here=$HOME/scripts/altar/perturbation
source $here/env.sh
cd /baghdad-nobak/ljzhu/kinematic/Illapel.orig/altar2
mkdir -p $here/logs
uuid=$(nvidia-smi -i $gpu --query-gpu=uuid --format=csv,noheader)
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"; then echo "gpu $gpu busy"; exit 1; fi
start=$(date +%s.%N)
CUDA_VISIBLE_DEVICES=$gpu slipmodel.plexus sample --config=$here/static_gauss.pfg \
    --controller.archiver.output_dir=results/$name "$@" > $here/logs/$name.log 2>&1
status=$?
wall=$(echo "$(date +%s.%N) - $start" | bc)
echo "wall $wall s, status $status" >> $here/logs/$name.log
printf "%-24s %8.1f s  status %d  %s\n" $name $wall $status "$(grep -h 'log evidence:' $here/logs/$name.log | tail -1)"
