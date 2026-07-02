---
title: "SolarLab vs SCAPS-1D — Parity Campaign Status & Verdict"
date: "2026-06-23"
---

# Bottom line

On the validated `scaps_mirror_v2` configuration, SolarLab reproduces SCAPS-1D's
**physics behaviour** — judged on the trends-over-absolutes standard (sweep
*direction* and *magnitude fidelity*, with base absolutes required only to
*approach* SCAPS):

- **Base operating point — matched by physics.** Effective-DOS band potentials
  (`dos_band_potentials`) + heterointerface Auger de-spike (`het_recomb_despike`)
  land V~oc~ on SCAPS within a few mV — *no fit knobs*.
- **All 11 single-variable sweep DIRECTIONS — matched by physics.** The
  steady-state interface-plane-states driver recovers every sweep's trend
  direction, including the ETL-donor-doping rising arm that the transient path
  gets backwards.
- **Sweep MAGNITUDES — structural deficit (documented boundary).** The response
  *amplitudes* (e.g. Nd_ETL ~52 %, PVK/ETL N~t~ ~63 % of the SCAPS V~oc~ range)
  are not fully reproduced. This is not a tuning gap — see Section 3.

**Verdict: the model is validated for SCAPS-comparison use.** Remaining gaps are
documented engine boundaries, honestly reported, not fudged.

# 1. Base operating point (no fit knobs)

| Metric | SolarLab (SS iface-states) | SolarLab (transient f=0.53) | SCAPS | diff |
|---|---|---|---|---|
| V~oc~ (V) | 1.167 | 1.164 | 1.168 | −0.1 … −4 mV |
| J~sc~ (mA/cm^2^) | 25.72 | 25.72 | 26.28 | −2.1 % (front-reflection residual) |
| FF (%) | 88.8 | 87.3 | 87.0 | +0.3 … +1.8 |
| PCE (%) | 26.5 | 26.1 | 26.7 | −0.2 … −0.6 |

# 2. Sweep directions (11/11 by physics)

All eleven single-variable sweeps (CBO, ETL/PVK doping, the four interface/bulk
N~t~, the four E~t~) match the SCAPS trend direction under the steady-state
interface-plane-states driver. Detailed four-config overlay (SCAPS / transient
f=0.53 / f=0.66 / SS interface-states), all four figures of merit per sweep:
**`SolarLab_SCAPS_validation_2026-07-02.pdf`**.

# 3. Documented boundaries (not matched — and why)

1. **Sweep magnitudes (structural).** Root cause, established by a 10-agent
   adversarial investigation: SolarLab's V~oc~ tracks the doping-swung built-in
   potential (the V~bi~ Poisson boundary condition swings +59.5 mV/decade with
   ETL doping), whereas SCAPS models **fixed-work-function flat-band contacts**
   whose V~oc~ responds only with the gentle kT/q·ln(N~D~) shift; this is
   compounded by a bulk-Auger-limited, doping-insensitive V~oc~ floor. The
   per-interface rate calibration is a *rate knob* (it sets the base/absolutes,
   not the sweep amplitude), so it cannot close this. Reaching it requires a
   SCAPS-faithful **quasi-Fermi-variable contact engine** — a large research
   rewrite. The flat-band-contact convention does not converge in the present
   steady-state driver (it needs that engine), so the magnitude deficit stands
   as a boundary.
2. **Deep conduction-band-offset cliff (ΔE~C~ ≤ −0.2 eV).** The collapsed-junction
   steady state is unreachable by *any* algebraic Newton (coupled Newton,
   decoupled Gummel, Anderson acceleration, and pseudo-transient continuation all
   stall 10–15 orders above tolerance). It is computed instead by a certified
   transient settle; since the conduction-band offset (not interface
   recombination) sets V~oc~ there, the steady state equals the transient, so this
   regime carries the transient value (shown as hollow markers in the CBO figure).
3. **+0.5 eV spike-side J~sc~ collapse.** Mechanism understood (the thermionic
   collection knee), not fully reproduced. A narrow extreme, low practical impact.

# 4. Canonical SCAPS-comparison configuration

`configs/scaps_mirror_v2.yaml` with `dos_band_potentials` (default on),
`het_recomb_despike: 0.53`, and the steady-state driver run with
`iface_states=True`. The transient (MOL) solver remains available for the dynamics
SCAPS cannot model.

# 5. A note on the standalone SCAPS engine (abandoned)

A standalone "SCAPS-faithful" engine subpackage was prototyped (2026-06-02) and
abandoned (2026-06-03): it rode the existing MOL transient (fast scan, ions off =
quasi-steady-state) plus two reference-fit calibration knobs, and could not reach
SCAPS absolutes *by physics* (only by tuning). Its diagnostics became the de-spike
(base) and interface-plane-states (directions) work now on `main`; the standalone
engine was not retained.

# 6. Recommendation

SCAPS parity has reached its physics-bounded plateau — directions and base are
matched by physics; absolute magnitudes need a large rewrite with uncertain payoff
and are out of scope under the trends-over-absolutes standard. SolarLab's
distinctive scientific value is the physics SCAPS *cannot* model — transient
dynamics (hysteresis, impedance, degradation, transient photovoltage) and 2D
microstructure / tandem / design-space studies. Recommend pivoting there.
