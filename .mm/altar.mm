# -*- Makefile -*-
#
# michael a.g. aïvázis
# parasim
# (c) 1998-2020 all rights reserved
#

builder.dest = products

# project meta-data
altar.major := $(repo.major)
altar.minor := $(repo.minor)
altar.micro := $(repo.micro)
altar.revision := $(repo.revision)

# altar consists of a python package
altar.packages := altar.pkg ${if ${value cuda.dir}, altar.cudapkg}
# libraries
altar.libraries := altar.lib ${if ${value cuda.dir},altar.cudalib}
# python extensions
altar.extensions := altar.ext ${if ${value cuda.dir},altar.cudaext}
# and some tests
altar.tests := altar.pkg.tests

# the altar package meta-data
altar.pkg.stem := altar
altar.pkg.root := altar/
altar.pkg.drivers := altar

# libaltar meta-data
altar.lib.stem := altar
altar.lib.root := lib/libaltar/
altar.lib.extern := gsl pyre
altar.lib.c++.flags += $($(compiler.c++).std.c++17)

# the altar extension meta-data
altar.ext.stem := altar
altar.ext.root := ext/
altar.ext.pkg := altar.pkg
altar.ext.wraps := altar.lib
altar.ext.extern := altar.lib gsl pyre python
# compile options for the sources
altar.ext.lib.c++.flags += $($(compiler.c++).std.c++17)

# the altar cuda package metadata
altar.cudapkg.stem := altar/cuda
altar.cudapkg.root := cuda/
altar.cudapkg.drivers := altar



# the altar CUDA library metadata
altar.cudalib.stem := cudaaltar
altar.cudalib.root := lib/libcudaaltar/
altar.cudalib.incdir := $(builder.dest.inc)altar/cuda/
altar.cudalib.extern := gsl pyre cuda
# compile options for the sources
altar.cudalib.c++.flags += $($(compiler.c++).std.c++17)
altar.cudalib.cuda.flags += $(nvcc.std.c++14)

# the altar CUDA extension meta-data
altar.cudaext.stem := cudaaltar
altar.cudaext.root := cudaext/
altar.cudaext.pkg := altar.pkg
altar.cudaext.wraps := altar.cudalib
altar.cudaext.extern := altar.cudalib gsl pyre python cuda
# compile options for the sources
altar.cudaext.lib.c++.flags += $($(compiler.c++).std.c++17)
altar.cudaext.lib.cuda.flags += $(nvcc.std.c++14)

# the altar test suite
altar.pkg.tests.stem := altar
altar.pkg.tests.prerequisites := altar.pkg altar.ext
# individual test cases
tests.altar.application_run.clean = \
    ${addprefix $(altar.pkg.tests.prefix),llk.txt sigma.txt theta.txt}

# models
include emhp.def gaussian.def cdm.def linear.def mogi.def ${if ${value cuda.dir}, cudalinear.def seismic.def}
# end of file
