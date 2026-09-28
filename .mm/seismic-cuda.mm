# -*- Makefile -*-
#
# michael a.g. aïvázis
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the gpu parts of the seismic models, when mm finds cuda
seismic-cuda.cuda.available := ${findstring cuda,$(extern.available)}

ifeq ($(seismic-cuda.cuda.available), cuda)

seismic-cuda.packages :=
seismic-cuda.libraries := seismic-cuda.lib
seismic-cuda.extensions := seismic-cuda.ext
seismic-cuda.tests :=

# the kernels, with their headers in {include/altar/models/seismic/cuda}
seismic-cuda.lib.root := models/seismic/lib/libcudaseismic/
seismic-cuda.lib.stem := seismic
seismic-cuda.lib.incdir := $(builder.dest.inc)altar/models/seismic/cuda/
seismic-cuda.lib.languages := c++ cuda
seismic-cuda.lib.extern := altar-cuda.lib pyre cuda
seismic-cuda.lib.prerequisites := altar-cuda.lib
seismic-cuda.lib.c++.flags += $($(compiler.c++).std.c++23)
seismic-cuda.lib.cuda.flags += $(nvcc.std.c++20) --expt-relaxed-constexpr

# the bindings, the module {altar.models.seismic.ext.cudaseismic}; they share the helpers of
# the framework's bindings, in {altar/ext/cuda}
seismic-cuda.ext.root := models/seismic/ext/cudaseismic/
seismic-cuda.ext.stem := cudaseismic
seismic-cuda.ext.pkg := seismic.pkg
seismic-cuda.ext.wraps := seismic-cuda.lib
seismic-cuda.ext.capsule :=
seismic-cuda.ext.extern := seismic-cuda.lib altar-cuda.lib pyre cuda pybind11 python
seismic-cuda.ext.lib.c++.flags += $($(compiler.c++).std.c++23) -I$(project.home)/altar/ext/cuda
seismic-cuda.ext.lib.prerequisites += altar-cuda.lib seismic-cuda.lib

endif

# end of file
