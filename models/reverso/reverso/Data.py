# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis (michael.aivazis@para-sim.com)
# grace bato           (mary.grace.p.bato@jpl.nasa.gov)
# eric m. gurrola      (eric.m.gurrola@jpl.nasa.gov)
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved


# framework
import altar


# the dataset
class Data(altar.tabular.sheet):
    """
    The layout of the observation geometry file; the observed (east, north, up) displacements
    of each observation are read by the model's {dataobs}, in the same order
    """

    # the layout
    oid = altar.tabular.int()
    oid.doc = "an integer identifying the data source"

    t = altar.tabular.float()
    t.doc = "the time of the observation, in seconds since the start of the inflow"

    x = altar.tabular.float()
    x.doc = "the EW coordinate of the location of the observation, from the chambers"

    y = altar.tabular.float()
    y.doc = "the NS coordinate of the location of the observation, from the chambers"


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
