#!/bin/bash
# build the perturbation branch from its bundle into its own prefix, against altar2-numpy
set -e
here=$HOME/scripts/altar/perturbation
source ~/miniforge3/etc/profile.d/conda.sh; conda activate altar2-numpy
src=$HOME/tools/src2/altar-perturbation
prefix=$HOME/.claude/tmp/altar/perturbation-install
if [ ! -d $src ]; then git clone -q -b perturbation $here/perturbation.bundle $src
else cd $src && git fetch -q $here/perturbation.bundle perturbation && git checkout -q -B perturbation FETCH_HEAD; fi
cd $src && git log --oneline -1
cmake -S $src -B $src/build -DCMAKE_INSTALL_PREFIX=$prefix -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python3 -DCMAKE_CUDA_ARCHITECTURES=120 > $here/build.cfg.log 2>&1
cmake --build $src/build --target install -j 16 > $here/build.log 2>&1
echo "built into $prefix"; ls $prefix/bin | grep slipmodel
