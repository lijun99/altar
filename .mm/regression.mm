# -*- Makefile -*-
#
# michael a.g. aïvázis
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the linear regression model
regression.packages := regression.pkg
regression.libraries :=
regression.extensions :=
regression.tests :=

# the python package, installed as {altar.models.regression}, and its driver
regression.pkg.root := models/regression/regression/
regression.pkg.stem := regression
regression.pkg.pycdir := $(builder.dest.pyc)altar/models/regression/
regression.pkg.bin := models/regression/bin/
regression.pkg.drivers := altar-regression

# end of file
