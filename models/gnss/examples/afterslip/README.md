# Afterslip on a strike-slip fault

A synthetic case for the gnss slip model, `altar.models.gnss.slip`: a vertical strike-slip
fault of 150 triangles (60 km long, 20 km deep), 25 stations, two years of daily positions, a
1 m coseismic rupture on day 365 and a year of logarithmic afterslip below it. The model
recovers the slip history from the time series, with the offsets, trends, seasonal terms and
the coseismic steps integrated out, and samples its posterior exactly.

```bash
python make_case.py                 # trajectory.h5, greens.h5, noise.json, truth.npz
altar-gnss --config=afterslip.pfg   # results/step_final.h5, 256 samples
python compare.py                   # the samples against the truth
```

`make_case.py` needs `cutde` for the Green's functions of the triangles, and `h5py`. Set
`job.gpus = 1` in `afterslip.pfg` to solve on the GPU, though a case this small runs as fast on
the CPU (about 6 s).

`compare.py` prints the final slip of the patches against the truth (correlation, coverage of
the 90% intervals) and the moment of the afterslip at a few days. From about two weeks after the
event, the moment is within 25% of the truth and inside the 90% intervals.

The slip's temporal basis is the case's I-spline groups, which start at the earthquake and are
short early on (4 and 8 days). With a basis that spans the whole window, before the event
too, the slip trades off against the trends and the coseismic steps. See the guide,
`docs/guide/GnssSlip.md`.
