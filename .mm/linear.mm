# -*- Makefile -*-
#
# michael a.g. aïvázis
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the linear model
linear.packages := linear.pkg
linear.libraries :=
linear.extensions :=
linear.tests :=

# the python package, installed as {altar.models.linear}, and its driver
linear.pkg.root := models/linear/linear/
linear.pkg.stem := linear
linear.pkg.pycdir := $(builder.dest.pyc)altar/models/linear/
linear.pkg.bin := models/linear/bin/
linear.pkg.drivers := altar-linear

# end of file
