# Exact integration of the source time function (model.integration)

Part A of `~/gps/sse_detections/docs/altar_kinematic_plan.md` (2026-11-17), on branch `kinematic`.

## What changed

- `stf.h`: `stf_cumulative` (V, the cumulative slip fraction, and W, its integral, with their
  derivatives in Tr) and `stf_interval`, the fraction of a source point's slip in an interval:
  - `sample` (0, the default): v(s) dt, as before;
  - `step` (1): V(s + dt) - V(s);
  - `bin` (2): [W(s + dt) - 2 W(s) + W(s - dt)] / dt, the change of the bin average of V.
- All four casts (cpu/cuda x rectangles/triangles) and their adjoints call `stf_interval`;
  `setSTF(kind, peak, mode)`, `set_stf(..., mode=0)`, `model.integration`,
  `SlipHistory(..., integration=)`.

## Checks (2026-11-17, RTX 4060 laptop, private build ~/.claude/tmp/altar/kinstf-install)

- `regression.py`: sample mode bit-identical to 8e5b872, Mb and gradient, triangles and
  rectangles, float64 and float32 (12 arrays, all equal).
- `../integration.py`: rise times 0.1-5 dt; step against quadrature 5e-16, bin 3e-14; slips in
  all 1e-16 to 5e-16 on triangles and rectangles. Sample, same inputs: off by 0.83 (triangles),
  0.17-0.40 (rectangles).
- `../kinematic_cpu.py --gpu --integration={sample,step,bin}`: gradients against central
  differences 1e-9 to 3e-7, float32 against float64 below 4e-5, cpu against cuda below 7e-15.
- `../stf.py --gpu`: step and bin at dt 8 s with rise times 4-6 s; gradients 2-3e-9 on both
  backends, cuda against cpu below 4e-16.
- history, kinematic_mesh, timeshift, polar, arrivals, noise: unchanged, passed.

## Findings

- `precision.py`, float32 against float64 by Tr/dt (0.5, 2, 10, 30, 100): sample and step stay
  at 1e-7 to 1e-5; bin, a second difference of W, is 1.4e-6 at 10, 2e-5 at 30, 1.7e-4 at 100.
  Fine for its use, Tr near dt; noted in the guide.
- `sample_totals.py`: the plan's table says sample is exact for dt <= Tr/2. Only at some ratios:
  the symmetric triangle is exact at Tr = 2 dt, but 0.89-1.11 at 3 dt and 0.96-1.04 at 5 dt;
  the 0.25 triangle 0.67-1.33 at 2 dt. One point, all onset phases.
- bin keeps the plan's uniform formula, Cbar_k - Cbar_(k-1), so the predictions are relative to
  the bin before the first; with onsets after t0 (t0s = 0, arrivals >= 0) that is the same as
  sse_toolkit's `bin_average`, whose cumulative starts at -inf.
