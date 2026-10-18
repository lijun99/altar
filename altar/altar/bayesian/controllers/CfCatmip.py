# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the package
import altar
# my superclass
from .Catmip import Catmip
# the sampler i work with
from ..samplers.Metropolis import Metropolis
# the model i sample with
from altar.models.CrossFade import CrossFade


# my declaration
class CfCatmip(Catmip, family="altar.controllers.cf_catmip"):
    """
    Cross-fade CATMIP (Minson, 2024): CATMIP from the conjugate posterior of the model to its
    posterior, fading its prior in and its conjugate prior out, instead of from its prior to its
    posterior; the data likelihood is never evaluated, and the evidence comes with the annealing.
    The model provides its conjugate posterior, e.g. the linear and the static slip models; i
    wrap it in a {altar.models.CrossFade} model, unless it is configured as one.
    """

    # protocol obligations
    @altar.export
    def posterior(self, model):
        """
        Wrap {model} in a {CrossFade} model, unless it is one already, then anneal
        """
        if not isinstance(self.sampler, Metropolis):
            self.error.log("cross-fade sampling supports the Metropolis sampler only, for now")
            raise SystemExit(1)
        from altar.models.cp.Adaptive import Adaptive
        if isinstance(getattr(model, "cp", None), Adaptive):
            self.error.log("cross-fade sampling needs a fixed C_p: the conjugate posterior is "
                           "computed once, before the annealing")
            raise SystemExit(1)
        # the model, cross-faded
        if not isinstance(model, CrossFade):
            model = CrossFade(name=f"{self.pyre_name}.crossfade").adopt(model=model)
        # the evidence starts from that of the conjugate model
        self.scheduler.log_evidence = model.log_evidence
        self.info.log(f"cross-fade: the conjugate model has log evidence {model.log_evidence} "
                      f"within the support of the prior")
        return super().posterior(model=model)


# end of file
