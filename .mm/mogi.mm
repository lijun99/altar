# -*- Makefile -*-
#
# michael a.g. aïvázis
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the mogi source model; its gpu forward model is a cuTile kernel, in the python package
mogi.packages := mogi.pkg
mogi.libraries := mogi.lib
mogi.extensions := mogi.ext
mogi.tests :=

# the python package, installed as {altar.models.mogi}, and its driver
mogi.pkg.root := models/mogi/mogi/
mogi.pkg.stem := mogi
mogi.pkg.pycdir := $(builder.dest.pyc)altar/models/mogi/
mogi.pkg.bin := models/mogi/bin/
mogi.pkg.drivers := altar-mogi

# the cpu forward model, with its headers in {include/altar/models/mogi}
mogi.lib.root := models/mogi/lib/libmogi/
mogi.lib.stem := mogi
mogi.lib.incdir := $(builder.dest.inc)altar/models/mogi/
mogi.lib.extern := gsl
mogi.lib.c++.flags += $($(compiler.c++).std.c++23)

# the extension module, {altar.models.mogi.ext.mogi}
mogi.ext.root := models/mogi/ext/mogi/
mogi.ext.stem := mogi
mogi.ext.pkg := mogi.pkg
mogi.ext.wraps := mogi.lib
mogi.ext.capsule :=
mogi.ext.extern := mogi.lib gsl pybind11 python
mogi.ext.lib.c++.flags += $($(compiler.c++).std.c++23)
mogi.ext.lib.prerequisites += mogi.lib

# end of file
