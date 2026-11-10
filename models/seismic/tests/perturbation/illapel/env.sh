# source me: altar2-numpy with the perturbation build in front, its libraries preloaded
source ~/miniforge3/etc/profile.d/conda.sh; conda activate altar2-numpy
prefix=$HOME/.claude/tmp/altar/perturbation-install
export PYTHONPATH=$prefix/packages PATH=$prefix/bin:$PATH
export LD_PRELOAD=$(ls $prefix/lib/lib*.so | tr '\n' ' ')
export OPENBLAS_NUM_THREADS=${BLAS_THREADS:-4} OMP_NUM_THREADS=${BLAS_THREADS:-4} MKL_NUM_THREADS=${BLAS_THREADS:-4}
