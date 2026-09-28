# -*- Makefile -*-
#
# michael a.g. aïvázis
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the gpu layer of the framework, when mm finds cuda
altar-cuda.cuda.available := ${findstring cuda,$(extern.available)}

ifeq ($(altar-cuda.cuda.available), cuda)

altar-cuda.packages := altar-cuda.pkg
altar-cuda.libraries := altar-cuda.lib
altar-cuda.extensions := altar-cuda.ext
altar-cuda.tests :=

# the python package, installed as {altar.cuda}
altar-cuda.pkg.root := altar/cuda/
altar-cuda.pkg.stem := cuda
altar-cuda.pkg.pycdir := $(builder.dest.pyc)altar/cuda/

# the kernels, with their headers in {include/altar/cuda}; {cudaRandom.cu} is not built
altar-cuda.lib.root := altar/lib/libcudaaltar/
altar-cuda.lib.stem := cudaaltar
altar-cuda.lib.incdir := $(builder.dest.inc)altar/cuda/
altar-cuda.lib.languages := c++ cuda
altar-cuda.lib.extern := pyre cuda
altar-cuda.lib.sources.exclude = $(altar-cuda.lib.prefix)distributions/cudaRandom.cu
altar-cuda.lib.c++.flags += $($(compiler.c++).std.c++23)
altar-cuda.lib.cuda.flags += $(nvcc.std.c++20) --expt-relaxed-constexpr

# the bindings, the module {altar.cuda.ext.cudaaltar}
altar-cuda.ext.root := altar/ext/cuda/
altar-cuda.ext.stem := cudaaltar
altar-cuda.ext.pkg := altar-cuda.pkg
altar-cuda.ext.wraps := altar-cuda.lib
altar-cuda.ext.capsule :=
altar-cuda.ext.extern := altar-cuda.lib altar.lib pyre cuda pybind11 python
altar-cuda.ext.lib.c++.flags += $($(compiler.c++).std.c++23)
altar-cuda.ext.lib.prerequisites += altar.lib altar-cuda.lib

# the cuda libraries the kernels and the bindings use
cuda.libraries += cudart cublas cusolver curand

endif

# end of file
