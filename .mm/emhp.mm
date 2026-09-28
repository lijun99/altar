# -*- Makefile -*-
#
# michael a.g. aïvázis
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the emhp model
emhp.packages := emhp.pkg
emhp.libraries :=
emhp.extensions :=
emhp.tests :=

# the python package, installed as {altar.models.emhp}, and its driver
emhp.pkg.root := models/emhp/emhp/
emhp.pkg.stem := emhp
emhp.pkg.pycdir := $(builder.dest.pyc)altar/models/emhp/
emhp.pkg.bin := models/emhp/bin/
emhp.pkg.drivers := emhp

# end of file
