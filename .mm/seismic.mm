# -*- Makefile -*-
#
# michael a.g. aïvázis
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the seismic slip models; their gpu parts are in {seismic-cuda}
seismic.packages := seismic.pkg
seismic.libraries :=
seismic.extensions :=
seismic.tests :=

# the python package, installed as {altar.models.seismic}, and its driver
seismic.pkg.root := models/seismic/seismic/
seismic.pkg.stem := seismic
seismic.pkg.pycdir := $(builder.dest.pyc)altar/models/seismic/
seismic.pkg.bin := models/seismic/bin/
seismic.pkg.drivers := seismic slipmodel slipmodel.plexus

# end of file
