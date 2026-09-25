# -*- cmake -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2021 all rights reserved

# get support
include(altar_seismic)

# build the package
altar_seismic_buildPackage()
# build the library
# altar_seismic_buildLibrary()
# build the extension
# altar_seismic_buildModule()
# copy the driver
altar_seismic_buildDriver()

# if CUDA is supported
if(WITH_CUDA)
  altar_seismic_cuda_buildPackage()
  altar_seismic_cuda_buildLibrary()
  # altar_seismic_cuda_buildModule() is not built for now: its sources (kinematicg.cc,
  # moment.cc, static.cc) are old CPython/PyCapsule bindings against pyre's own old capsule
  # bridge (pyre/cuda/capsules.h), which no longer exists -- the same class of pybind11+
  # AnyGrid port still pending for altar's own distributions.cc/metropolis.cc/leapfrog.cc/
  # langevin.cc (see the note in altar/ext/cuda/cudaaltar.cc), just for seismic's model-
  # specific bindings instead of the core. {libcudaseismic} (the .cu kernel library, no
  # capsule dependency) still builds and stays ready for whenever that port happens.
  # altar_seismic_cuda_buildModule()
  altar_seismic_cuda_buildDriver()
endif()
# end of file
