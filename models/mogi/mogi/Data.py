# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# framework
import altar


# declaration
class Data(altar.tabular.sheet):
    """
    The layout of the observation geometry file; the observed LOS displacements themselves are
    read by the model's {dataobs}, in the same order
    """

    # the layout
    oid = altar.tabular.int()
    oid.doc = "an integer identifying the dataset of the observation, for its offset"

    x = altar.tabular.float()
    x.doc = "the EW coordinate of the observation"

    y = altar.tabular.float()
    y.doc = "the NS coordinate of the observation"

    theta = altar.tabular.float()
    theta.doc = "the incidence angle of the LOS vector, from the vertical, in radians"

    phi = altar.tabular.float()
    phi.doc = "the azimuth of the LOS vector, counterclockwise from east, in radians"


    # load data from a csv file
    def read(self, uri):
        """
        Load a data set from a CSV file
        """
        # make a CSV reader
        csv = altar.records.csv()
        # pull data from the file and populate me with immutable records
        self.pyre_immutable(data = csv.read(layout=self, uri=uri))
        # all done
        return self


    # dump my data into a CSV file
    def write(self, uri):
        """
        Save my data into a CSV file
        """
        # make a CSV writer
        csv = altar.records.csv()
        # ask it to save the data
        csv.write(sheet=self, uri=uri)
        # all done
        return self


# end of file
