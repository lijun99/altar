# -*- Makefile -*-
#
# michael a.g. aïvázis
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the framework
altar.packages := altar.pkg
altar.libraries := altar.lib
altar.extensions := altar.ext
altar.tests :=

# the python package, and its driver
altar.pkg.root := altar/altar/
altar.pkg.stem := altar
altar.pkg.bin := altar/bin/
altar.pkg.drivers := altar

# the library, with its headers in {include/altar}
altar.lib.root := altar/lib/libaltar/
altar.lib.stem := altar
altar.lib.incdir := $(builder.dest.inc)altar/
altar.lib.extern := gsl pyre
altar.lib.c++.flags += $($(compiler.c++).std.c++23)

# the extension module; the bindings of the gpu layer, in {cuda}, belong to {altar-cuda}
altar.ext.root := altar/ext/
altar.ext.stem := altar
altar.ext.pkg := altar.pkg
altar.ext.wraps := altar.lib
altar.ext.capsule :=
altar.ext.extern := altar.lib gsl pyre pybind11 python
altar.ext.lib.directories.exclude := cuda
altar.ext.lib.c++.flags += $($(compiler.c++).std.c++23)

# end of file
