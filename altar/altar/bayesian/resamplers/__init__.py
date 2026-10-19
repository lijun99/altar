# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

import altar
from .Resampler import Resampler as resampler

@altar.foundry(
    implements=resampler,
    tip="resample by the importance weights: multinomial, stratified, systematic or residual")
def importanceresampler():
    from .ImportanceResampler import ImportanceResampler
    __doc__ = ImportanceResampler.__doc__
    return ImportanceResampler

# end of file
