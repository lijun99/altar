# -*- Makefile -*-
#
# michael a.g. aïvázis
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the compound dislocation model; its gpu forward model is a cuTile kernel, in the python package
cdm.packages := cdm.pkg
cdm.libraries := cdm.lib
cdm.extensions := cdm.ext
cdm.tests :=

# the python package, installed as {altar.models.cdm}, and its driver
cdm.pkg.root := models/cdm/cdm/
cdm.pkg.stem := cdm
cdm.pkg.pycdir := $(builder.dest.pyc)altar/models/cdm/
cdm.pkg.bin := models/cdm/bin/
cdm.pkg.drivers := altar-cdm

# the cpu forward model, with its headers in {include/altar/models/cdm}
cdm.lib.root := models/cdm/lib/libcdm/
cdm.lib.stem := cdm
cdm.lib.incdir := $(builder.dest.inc)altar/models/cdm/
cdm.lib.extern :=
cdm.lib.c++.flags += $($(compiler.c++).std.c++23)

# the extension module, {altar.models.cdm.ext.cdm}
cdm.ext.root := models/cdm/ext/cdm/
cdm.ext.stem := cdm
cdm.ext.pkg := cdm.pkg
cdm.ext.wraps := cdm.lib
cdm.ext.capsule :=
cdm.ext.extern := cdm.lib pybind11 python
cdm.ext.lib.c++.flags += $($(compiler.c++).std.c++23)
cdm.ext.lib.prerequisites += cdm.lib

# end of file
