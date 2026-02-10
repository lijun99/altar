# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): michael a.g. aïvázis, Lijun Zhu

# support
import altar

# the simple application shell
from .Application import Application


class cudaApplication(Application, family="altar.shells.cudaapplication"):
    """
    Backward-compatible alias for the unified application shell.
    """


    # pyre framework hooks
    # interactive session management
    def pyre_interactiveSessionContext(self, context):
        """
        Go interactive
        """
        # protect against bad context
        if context is None:
            # by initializing an empty one
            context = {}
        # add some symbols
        context["altar"] = altar # my package
        # and chain up
        return super().pyre_interactiveSessionContext(context=context)


    # machine layout adjustments for MPI runs
    def pyre_mpi(self):
        """
        Transfer my {job} settings to the MPI shell
        """
        # get my shell
        shell = self.shell
        # if the programming model is not {MPI}
        if shell.model != "mpi":
            # something really bad has happened
            self.firewall.log(f"the {pyre_mpi} hook with model={shell.model}")

        # get my job parameters
        job = self.job
        # transfer the job settings
        shell.hosts = job.hosts
        shell.tasks = job.tasks

        # all done
        return self


# end of file
