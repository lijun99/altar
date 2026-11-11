# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the design of a trajectory model: the time series of many stations and components, the
# "blocks", on one common time axis, and the groups of basis functions fitted to every block
#
# the case file, hdf5, format "altar.gnss.trajectory/1":
#   attrs        format, reference (the epoch of time zero, ISO date, for the record), groups
#                (the group names, in parameter order)
#   time         (n,) days since the reference, common to all the blocks
#   blocks/      station (B,) and component (B,) names, lon and lat (B,) in degrees
#   raw          (B, n) observed positions, NaN where missing
#   sigma        (B, n) their standard deviations, NaN where missing
#   data         (B n,) raw / sigma, the whitened observations, 0 where missing
#   mask         (B n,) 1 where observed, 0 where missing
#   groups/NAME  one per group of basis functions, kind and settings in its attrs:
#                  polynomial   order, t0, unit, first (the lowest power, default 0)
#                  sinusoid     periods, t0
#                  isplines     t_start, t_end, scales, spacing, degree, internal_scaling
#                  steps        dataset "times", (B, k) days, NaN padded
#                  logarithmic  dataset "times", (B, k) days, NaN padded; tau
# the first three are shared, the same columns for every block; the last two are per block
#
# the parameters: for each group, in the order of the "groups" attribute, the coefficients of
# block 0, then block 1, ..., so group g holds B x p_g parameters, a parameter set of AlTar
# whose name is the group's; observations: block 0's n days, then block 1's, ...; the model
# predicts the whitened data, w * (sum over groups of G_g theta_g), with w = 1 / sigma, 0
# where missing


# externals
import numpy
from . import basis

FORMAT = "altar.gnss.trajectory/1"
SHARED = ("polynomial", "sinusoid", "isplines")
PER_BLOCK = ("steps", "logarithmic")


class Group:
    """
    A group of basis functions: its design matrix, (n x p) if shared by all blocks or (B x n x p)
    if each block has its own, and a table of its columns
    """

    def __init__(self, name, kind, G, columns):
        self.name, self.kind, self.G, self.columns = name, kind, G, columns
        self.shared = G.ndim == 2
        self.width = G.shape[-1]
        return

    def __repr__(self):
        where = "shared" if self.shared else "per block"
        return f"{self.name} ({self.kind}, {where}, {self.width} columns)"


class Design:
    """
    The blocks, their whitened observations and weights, and the groups of basis functions
    """

    def __init__(self, time, station, component, weight, groups, data=None, mask=None,
                 lon=None, lat=None, reference=None):
        self.time = numpy.asarray(time, dtype=float)
        self.station, self.component = list(station), list(component)
        self.weight = numpy.asarray(weight, dtype=float)       # (B, n), 1/sigma, 0 if missing
        self.groups = groups
        self.data, self.mask = data, mask
        self.lon, self.lat, self.reference = lon, lat, reference
        self.blocks, self.days = self.weight.shape
        return

    @property
    def observations(self):
        return self.blocks * self.days

    def parameters(self, name=None):
        """
        The number of parameters of the group {name}, or of all of them
        """
        groups = self.groups if name is None else [g for g in self.groups if g.name == name]
        return sum(self.blocks * g.width for g in groups)

    def predict(self, theta, offsets):
        """
        The whitened predictions, (samples x observations), of the rows of {theta}, whose group
        g starts at {offsets[g.name]}
        """
        θ = numpy.atleast_2d(numpy.asarray(theta, dtype=float))
        u = numpy.zeros((θ.shape[0], self.blocks, self.days))
        for g in self.groups:
            c = θ[:, offsets[g.name]:offsets[g.name] + self.blocks * g.width]
            c = c.reshape(θ.shape[0], self.blocks, g.width)
            u += c @ g.G.T if g.shared else numpy.einsum("sbk,bnk->sbn", c, g.G)
        return (u * self.weight).reshape(θ.shape[0], -1)

    def adjoint(self, v, offsets, out):
        """
        Fill the groups' columns of {out}, (samples x parameters), with J^T v for the whitened
        (samples x observations) {v}, J the Jacobian of {predict}
        """
        v = numpy.asarray(v).reshape(-1, self.blocks, self.days) * self.weight
        for g in self.groups:
            a = v @ g.G if g.shared else numpy.einsum("sbn,bnk->sbk", v, g.G)
            out[:, offsets[g.name]:offsets[g.name] + self.blocks * g.width] = \
                a.reshape(v.shape[0], -1)
        return out

    def posterior(self, offsets, sigma, mean=None):
        """
        The exact posterior means and standard deviations of the parameters under independent
        gaussian priors, {sigma} (and {mean}, default 0) by group name, a solve per block,
        since the likelihood is block diagonal; also returns the posterior covariance of each
        block, by block, over its columns in group order
        """
        P = max(offsets[g.name] + self.blocks * g.width for g in self.groups)
        mu, sd = numpy.zeros(P), numpy.zeros(P)
        mean = {} if mean is None else mean
        data = self.data.reshape(self.blocks, self.days)
        covariances = []
        for b in range(self.blocks):
            X = numpy.hstack([g.G if g.shared else g.G[b] for g in self.groups]) \
                * self.weight[b][:, None]
            idx = numpy.concatenate([offsets[g.name] + b * g.width + numpy.arange(g.width)
                                     for g in self.groups])
            s = numpy.concatenate([numpy.full(g.width, float(sigma[g.name])) for g in self.groups])
            m = numpy.concatenate([numpy.full(g.width, float(mean.get(g.name, 0.0)))
                                   for g in self.groups])
            A = X.T @ X + numpy.diag(1 / s ** 2)
            L = numpy.linalg.cholesky(A)
            C = numpy.linalg.inv(L).T @ numpy.linalg.inv(L)
            mu[idx] = C @ (X.T @ data[b] + m / s ** 2)
            sd[idx] = numpy.sqrt(numpy.diag(C))
            covariances.append(C)
        return mu, sd, covariances

    def describe(self):
        lines = [f"{self.blocks} blocks x {self.days} days = {self.observations} observations, "
                 f"{int(self.weight.astype(bool).sum())} valid"]
        lines += [f"  {g!r}: {self.blocks * g.width} parameters" for g in self.groups]
        return lines


def build_group(name, kind, attrs, time, blocks, times=None):
    """
    The design matrix of a group from its {kind} and settings
    """
    if kind == "polynomial":
        first = int(attrs.get("first", 0))
        G = basis.polynomial(time, int(attrs["order"]), float(attrs.get("t0", 0.0)),
                             float(attrs.get("unit", 365.25)), first)
        columns = {"power": numpy.arange(first, first + G.shape[1])}
    elif kind == "sinusoid":
        periods = numpy.atleast_1d(attrs["periods"]).astype(float)
        G = numpy.hstack([basis.sinusoid(time, p, float(attrs.get("t0", 0.0))) for p in periods])
        columns = {"period": numpy.repeat(periods, 2), "part": numpy.tile(["cos", "sin"], periods.size)}
    elif kind == "isplines":
        G, columns = basis.isplines(time, float(attrs["t_start"]), float(attrs["t_end"]),
                                    numpy.atleast_1d(attrs["scales"]).astype(float),
                                    float(attrs.get("spacing", 1.0)), int(attrs.get("degree", 2)),
                                    bool(attrs.get("internal_scaling", False)))
    elif kind in PER_BLOCK:
        times = numpy.asarray(times, dtype=float).reshape(blocks, -1)
        if kind == "steps":
            G = numpy.stack([basis.step(time, ts) for ts in times])
        else:
            G = numpy.stack([basis.logarithmic(time, ts, float(attrs["tau"])) for ts in times])
        columns = {"times": times}
    else:
        raise ValueError(f"unknown group kind '{kind}' of group '{name}'")
    return Group(name, kind, G, columns)


def _text(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def load(path):
    """
    The design of the case file {path}
    """
    import h5py
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("format") != FORMAT:
            raise ValueError(f"'{path}': not a {FORMAT} case")
        time = h5["time"][:]
        station = _text(h5["blocks/station"][:])
        component = _text(h5["blocks/component"][:])
        lon = h5["blocks/lon"][:] if "blocks/lon" in h5 else None
        lat = h5["blocks/lat"][:] if "blocks/lat" in h5 else None
        sigma = h5["sigma"][:]
        mask = h5["mask"][:].astype(bool).reshape(sigma.shape)
        data = h5["data"][:]
        groups = []
        for name in _text(h5.attrs["groups"]):
            g = h5["groups"][name]
            attrs = dict(g.attrs)
            kind = attrs.pop("kind")
            kind = kind.decode() if isinstance(kind, bytes) else kind
            times = g["times"][:] if "times" in g else None
            groups.append(build_group(name, kind, attrs, time, len(station), times))
        reference = h5.attrs.get("reference")
    with numpy.errstate(divide="ignore", invalid="ignore"):
        weight = numpy.where(mask & (sigma > 0), 1 / sigma, 0.0)
    return Design(time, station, component, weight, groups, data=data, mask=mask.ravel(),
                  lon=lon, lat=lat, reference=reference)


def write(path, time, station, component, raw, sigma, groups, lon=None, lat=None,
          reference=""):
    """
    Write a case file; {groups} is a list of (name, kind, settings) with the per-block
    "times" among the settings of steps and logarithmic groups
    """
    import h5py
    raw, sigma = numpy.asarray(raw, dtype=float), numpy.asarray(sigma, dtype=float)
    mask = numpy.isfinite(raw) & numpy.isfinite(sigma) & (sigma > 0)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        data = numpy.where(mask, raw / sigma, 0.0)
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = FORMAT
        h5.attrs["reference"] = str(reference)
        h5.attrs["groups"] = [name for name, _, _ in groups]
        h5["time"] = numpy.asarray(time, dtype=float)
        h5["blocks/station"] = numpy.array(station, dtype="S")
        h5["blocks/component"] = numpy.array(component, dtype="S")
        if lon is not None:
            h5["blocks/lon"], h5["blocks/lat"] = numpy.asarray(lon), numpy.asarray(lat)
        h5["raw"], h5["sigma"] = raw, sigma
        h5["data"], h5["mask"] = data.ravel(), mask.ravel().astype("i1")
        for name, kind, settings in groups:
            g = h5.create_group(f"groups/{name}")
            g.attrs["kind"] = kind
            for key, value in settings.items():
                if key == "times":
                    g["times"] = numpy.asarray(value, dtype=float)
                else:
                    g.attrs[key] = value
    return path


# end of file
