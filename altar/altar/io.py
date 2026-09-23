# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2026 parasim inc
# (c) 2010-2026 california institute of technology
# all rights reserved
#


# externals
import os
import numpy
# the package
import altar


class FileIO:
    """
    Reads and writes model input/output files, dispatching on the file suffix:
    '.txt' (text), '.bin'/'.dat' (raw binary), '.h5' (HDF5)
    """

    # meta-methods
    def __init__(self, ifs=None, error=None, precision=None, **kwds):
        super().__init__(**kwds)
        # the mounted input filesystem, for {load}; not needed for {save}
        self.ifs = ifs
        # a journal channel to report missing/unsupported files to; optional
        self.error = error
        # the default numpy dtype to load/save as, when the caller doesn't specify one
        self.precision = precision
        # all done
        return


    def load(self, filename, shape=None, dataset=None, dtype=None):
        """
        Load {filename}, found through {self.ifs}, into a gsl vector or matrix
        """
        # the desired precision
        dtype = dtype or self.precision

        ifs = self.ifs
        try:
            # get the path to the file
            file = ifs[filename]
        # if the file doesn't exist
        except ifs.NotFoundError:
            # complain, if i have a channel to complain to
            if self.error is not None:
                self.error.log(f"missing input: no '{filename}' in '{ifs.path()}'")
            # and raise the exception again
            raise

        # get the suffix to determine the format
        suffix = file.uri.suffix
        if suffix == '.txt':
            # text file
            cpuData = numpy.loadtxt(file.uri.path, dtype=dtype)
        elif suffix in ('.bin', '.dat'):
            # raw binary; the caller must know the shape, since nothing else does
            if shape is None:
                raise ValueError(f"must specify shape for binary input '{filename}'")
            cpuData = numpy.fromfile(file.uri.path, dtype=dtype)
        elif suffix == '.h5':
            # hdf5
            import h5py
            h5file = h5py.File(file.uri.path, 'r')
            # if the caller didn't say which dataset, assume the only or first one
            if dataset is None:
                dataset = list(h5file.keys())[0]
            cpuData = numpy.asarray(h5file.get(dataset), dtype=dtype)
            h5file.close()
        else:
            raise ValueError(f"unsupported input suffix '{suffix}' for '{filename}'")

        # reshape, if requested
        if shape is not None:
            cpuData = cpuData.reshape(shape)

        # convert to gsl and return
        return self.toGsl(cpuData)


    def save(self, filename, data, dataset=None, mode='a'):
        """
        Save {data} (a gsl vector/matrix, or anything numpy can coerce) to {filename},
        dispatching on its suffix
        """
        # get a plain numpy array, regardless of what {data} actually is
        cpuData = data.ndarray() if hasattr(data, 'ndarray') else numpy.asarray(data)

        # {filename} may be a plain str or a pyre path-like object
        path = str(filename)
        suffix = os.path.splitext(path)[1]

        if suffix == '.txt':
            numpy.savetxt(path, cpuData)
        elif suffix in ('.bin', '.dat'):
            cpuData.tofile(path)
        elif suffix == '.h5':
            import h5py
            # the dataset name to use inside the file
            dataset = dataset or 'data'
            h5file = h5py.File(path, mode=mode)
            # replace an existing dataset by the same name, rather than erroring out
            if dataset in h5file.keys():
                del h5file[dataset]
            h5file.create_dataset(name=dataset, data=cpuData)
            h5file.close()
        else:
            raise ValueError(f"unsupported output suffix '{suffix}' for '{filename}'")

        # all done
        return


    # implementation details
    def toGsl(self, cpuData):
        """
        Convert a numpy array into a gsl vector or matrix
        """
        if cpuData.ndim == 1:
            vec = altar.vector(shape=cpuData.shape[0])
            vec.ndarray()[:] = cpuData
            return vec
        if cpuData.ndim == 2:
            mat = altar.matrix(shape=cpuData.shape)
            mat.ndarray()[:] = cpuData
            return mat
        raise ValueError(f"unsupported data dimensions {cpuData.shape}")


# end of file
