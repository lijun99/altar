# -*- Makefile -*-
#
# michael a.g. aïvázis
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the gaussian model
gaussian.packages := gaussian.pkg
gaussian.libraries :=
gaussian.extensions :=
gaussian.tests :=

# the python package, installed as {altar.models.gaussian}, and its driver
gaussian.pkg.root := models/gaussian/gaussian/
gaussian.pkg.stem := gaussian
gaussian.pkg.pycdir := $(builder.dest.pyc)altar/models/gaussian/
gaussian.pkg.bin := models/gaussian/bin/
gaussian.pkg.drivers := gaussian

# end of file
