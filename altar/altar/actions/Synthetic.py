# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import numpy
import pathlib
import shutil
# get the package
import altar
# the samples, as the forward check reads them
from .Forward import load_samples


# declaration
class Synthetic(altar.panel(), family='altar.actions.synthetic'):
    """
    Make synthetic data for a recovery test: run the forward model on a true model, add noise
    drawn from the data covariance, and write the data into a copy of the input directory, which
    a run can then sample; the true model may be a checkerboard over some parameter sets
    """


    # user configurable state
    theta = altar.properties.path(default="results/step_final.h5")
    theta.doc = "the true model, the mean of these samples: an archived step, or a .txt/.h5 " \
                "file with one sample per row"

    dataset = altar.properties.str(default=None)
    dataset.doc = "the dataset holding {theta} in a plain .h5 file; default the first one"

    checkerboard = altar.properties.list(schema=altar.properties.str())
    checkerboard.doc = "the parameter sets to replace by a checkerboard"

    grid = altar.properties.list(schema=altar.properties.int())
    grid.doc = "the (rows, columns) of the checkerboard parameter sets, the columns running " \
               "fastest; default a single column"

    block = altar.properties.int(default=1)
    block.doc = "the size of the squares of the checkerboard, in parameters"

    values = altar.properties.list(schema=altar.properties.float(), default=[0.0, 1.0])
    values.doc = "the values of the two colors of the checkerboard"

    noise = altar.properties.bool(default=True)
    noise.doc = "add noise drawn from the data covariance"

    cp = altar.properties.bool(default=False)
    cp.doc = "draw the noise from C_chi = C_d + C_p at the true model, for a model with a C_p"

    seed = altar.properties.int(default=0)
    seed.doc = "the seed of the noise"

    output = altar.properties.path(default="synthetic")
    output.doc = "the directory for the copy of the inputs with the synthetic data, and the " \
                 "true model, truth.txt"


    # commands
    @altar.export(tip="make synthetic data from a true model")
    def default(self, plexus, **kwds):
        """
        Build the true model, predict its data for each model of an ensemble, add the noise and
        write them out with the rest of the inputs
        """
        model = plexus.model
        θ = load_samples(model=model, path=self.theta, dataset=self.dataset)
        truth = θ.mean(axis=0)
        for name in self.checkerboard:
            pset = model.psets[name]
            truth[pset.offset:pset.offset + pset.count] = self.pattern(count=pset.count)

        channel = plexus.info
        output = pathlib.Path(str(self.output))
        output.mkdir(parents=True, exist_ok=True)
        channel.line(f"synthetic: the mean of '{self.theta}' -> '{output}'")
        if self.checkerboard:
            channel.line(f"  a checkerboard of {self.values[0]:g} and {self.values[1]:g},"
                         f" squares of {self.block}, over {', '.join(self.checkerboard)}")
        rng = numpy.random.default_rng(self.seed)
        members = getattr(model, "models", None)
        cases = set()
        for name, member, columns in (
                [(name, member, model.columns(name)) for name, member in members.items()]
                if members else [(None, model, numpy.arange(model.parameters))]):
            # the inputs, once per case
            case = pathlib.Path(str(member.case))
            if case not in cases:
                shutil.copytree(case, output, dirs_exist_ok=True)
                cases.add(case)
            data = self.predict(plexus=plexus, model=member, truth=truth[columns], rng=rng)
            self.write(dataobs=member.dataobs, data=data, path=output)
            label = f"{name}: " if name is not None else ""
            channel.line(f"  {label}{data.size} observations -> '{output / str(member.dataobs.data_file)}'")
        numpy.savetxt(output / "truth.txt", truth[None, :])
        channel.line(f"  the true model -> '{output / 'truth.txt'}'")
        channel.log()
        return 0


    def pattern(self, count):
        """
        A checkerboard of {count} parameters on my grid, row by row
        """
        rows, columns = self.grid if self.grid else (count, 1)
        if rows * columns != count:
            raise ValueError(f"a {rows} x {columns} checkerboard doesn't fit {count} parameters")
        i, j = numpy.divmod(numpy.arange(count), columns)
        color = (i // self.block + j // self.block) % 2
        return numpy.where(color == 0, self.values[0], self.values[1])


    def predict(self, plexus, model, truth, rng):
        """
        The data {model} predicts for {truth}, with noise from its covariance
        """
        data = model.forward_problem(application=plexus, theta=truth[None, :])["data"][0]
        if not self.noise:
            return data
        dataobs = model.dataobs
        cp = getattr(model, "cp", None)
        if self.cp and cp is not None:
            cp.apply(model=model, theta=truth)
        elif cp is not None:
            model.update_covariance(cp=None)
        covariance = dataobs.covariance()
        z = rng.standard_normal(data.size)
        if numpy.ndim(covariance) == 0:
            data = data + numpy.sqrt(float(covariance)) * z
        else:
            data = data + numpy.linalg.cholesky(numpy.asarray(covariance, dtype=float)) @ z
        # masked observations stay out of the likelihood; keep them at zero
        if dataobs.mask is not None:
            data = data * dataobs.mask
        return data


    def write(self, dataobs, data, path):
        """
        Replace the observations in the copy of the data file of {dataobs} under {path}
        """
        filename = path / str(dataobs.data_file)
        dataset = dataobs.datafile_dataset
        if filename.suffix == ".h5" and dataset is None:
            import h5py
            with h5py.File(filename, "r") as h5:
                dataset = list(h5.keys())[0]
        altar.io.FileIO().save(filename=filename, data=data, dataset=dataset)
        return


# end of file
