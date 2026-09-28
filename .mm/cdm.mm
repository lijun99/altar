# -*- Makefile -*-
#
# michael a.g. aïvázis
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the compound dislocation model, on the cpu
cdm.packages := cdm.pkg
cdm.libraries := cdm.lib
cdm.extensions := cdm.ext
cdm.tests :=

# the python package, installed as {altar.models.cdm}, and its driver
cdm.pkg.root := models/cdm/cdm/
cdm.pkg.stem := cdm
cdm.pkg.pycdir := $(builder.dest.pyc)altar/models/cdm/
cdm.pkg.bin := models/cdm/bin/
cdm.pkg.drivers := cdm

# the library, with its headers in {include/altar/models/cdm}
cdm.lib.root := models/cdm/lib/libcdm/
cdm.lib.stem := cdm
cdm.lib.incdir := $(builder.dest.inc)altar/models/cdm/
cdm.lib.extern := gsl pyre
cdm.lib.c++.flags += $($(compiler.c++).std.c++23)

# the extension module
cdm.ext.root := models/cdm/ext/cdm/
cdm.ext.stem := cdm
cdm.ext.pkg := cdm.pkg
cdm.ext.wraps := cdm.lib
cdm.ext.extern := cdm.lib altar.lib gsl pyre python
cdm.ext.lib.c++.flags += $($(compiler.c++).std.c++23)
cdm.ext.lib.prerequisites += altar.lib

# end of file
