# -*- Makefile -*-
#
# michael a.g. aïvázis
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the reverso two chamber model; its gpu forward model is a cuTile kernel, in the python package
reverso.packages := reverso.pkg
reverso.libraries := reverso.lib
reverso.extensions := reverso.ext
reverso.tests :=

# the python package, installed as {altar.models.reverso}, and its driver
reverso.pkg.root := models/reverso/reverso/
reverso.pkg.stem := reverso
reverso.pkg.pycdir := $(builder.dest.pyc)altar/models/reverso/
reverso.pkg.bin := models/reverso/bin/
reverso.pkg.drivers := altar-reverso

# the cpu forward model, with its headers in {include/altar/models/reverso}
reverso.lib.root := models/reverso/lib/libreverso/
reverso.lib.stem := reverso
reverso.lib.incdir := $(builder.dest.inc)altar/models/reverso/
reverso.lib.extern :=
reverso.lib.c++.flags += $($(compiler.c++).std.c++23)

# the extension module, {altar.models.reverso.ext.reverso}
reverso.ext.root := models/reverso/ext/reverso/
reverso.ext.stem := reverso
reverso.ext.pkg := reverso.pkg
reverso.ext.wraps := reverso.lib
reverso.ext.capsule :=
reverso.ext.extern := reverso.lib pybind11 python
reverso.ext.lib.c++.flags += $($(compiler.c++).std.c++23)
reverso.ext.lib.prerequisites += reverso.lib

# end of file
