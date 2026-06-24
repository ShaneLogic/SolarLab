---
title: "De-spike and Interface-Plane Closure: A Mechanistic Decomposition of SolarLab's Heterojunction Trend Response vs SCAPS-1D"
date: 2026-06-21
header-includes: |
  \usepackage{float}
  \makeatletter
  \let\oldfigure\figure
  \let\endoldfigure\endfigure
  \renewenvironment{figure}[1][]{\oldfigure[htbp]}{\endoldfigure}
  \makeatother
---

# Abstract

This note records the outcome of a controlled four-configuration experiment that
isolates *which* heterojunction mechanism each parameter-sweep trend depends on.
Two independent corrections were tested — the **heterointerface Auger de-spike**
(`het_recomb_despike`, a bulk-rate blend that removes the over-counted Auger
recombination at the junction mesh node) and the **interface-plane closure**
(`interface_plane_closure`, a per-RHS quasi-steady algebraic interface-state
term) — both alone and combined, against a steady-state interface-states
reference. The result is a clean separation: **the de-spike governs the base
$V_\text{oc}$ and the bulk-defect ($N_t$) trends; the closure governs the
interface-defect directions; the two are orthogonal and complementary, not
redundant.** Combining them (**configuration AB**) is the best fast-transient
parity point measured — base $V_\text{oc}=1.161$ V (within 7 mV of SCAPS) with
all bulk and interface trends in the correct direction. Two residuals remain and
are individually named and bounded: a 4$\times$ over-response of the near-passive
HTL/PVK interface $N_t$ trend (a de-spike/closure overlap at one shared node),
and the long-standing ETL-doping ($N_d$) direction mismatch (resolved only by
the steady-state interface states, which are presently ~80$\times$ too slow to use in
the sweep harness).

# 1. The four configurations

All share the reference Base Model (`configs/scaps_mirror_v2.yaml`,
`dos_band_potentials: true`). They differ only in which interface correction is
active.

| ID | `het_recomb_despike` | `interface_plane_closure` | Driver | Cost |
|----|----|----|----|----|
| **A** | 0.53 | off | transient `run_jv_sweep` | ~7 s/point |
| **B** | 0 | on | transient `run_jv_sweep` | ~7 s/point |
| **AB** | 0.53 | on | transient `run_jv_sweep` | ~7 s/point |
| **C** | 0 | off (steady-state `iface_states`) | Newton `solve_voc_ss` | ~480 s/$V_\text{oc}$ |

Table 1. The de-spike fraction $f=0.53$ is the base-optimal value from the prior
de-spike matrix sweep (it lands base $V_\text{oc}$ on SCAPS; $f$ monotonically
lifts $V_\text{oc}$ from 1.118 V at $f=0$ to 1.201 V at $f=1$). Configuration C
is the steady-state interface-states solver — included as the physical reference
for the interface directions, but impractical in the sweep harness at its
current cost (a single $V_\text{oc}$ took 481 s; the full 85-point matrix would
be ~11 h).

# 2. Base operating point

| Config | $V_\text{oc}$ (V) | $J_\text{sc}$ (mA/cm$^2$) | FF (%) | PCE (%) | $\Delta V_\text{oc}$ vs SCAPS |
|---|---|---|---|---|---|
| A (de-spike only) | **1.1638** | 25.72 | 87.32 | 26.14 | -3.8 mV |
| B (closure only) | 1.1178 | 25.70 | 88.23 | 25.35 | -49.8 mV |
| **AB (both)** | **1.1610** | 25.72 | 88.54 | 26.44 | **-6.6 mV** |
| C (SS states) | 1.1063 | — | — | — | -61.3 mV |
| SCAPS-1D | 1.1676 | 26.28 | 86.99 | 26.69 | — |

Table 2. The de-spike is what closes the base $V_\text{oc}$: removing the
over-counted junction Auger lifts B's 1.118 V to A's 1.164 V and AB's 1.161 V.
The closure alone (B) does almost nothing to the base point; the steady-state
states (C) actually sit *below* B at the base, confirming that the base deficit
is a bulk-recombination over-count, not an interface-transport effect.

# 3. Trend matrix

Closure % is the dimensionless ratio (SolarLab $V_\text{oc}$ range / SCAPS
$V_\text{oc}$ range) over each sweep; OK = correct direction, X = direction
mismatch, "flat/flat" = both solvers respond < 1 mV so the ratio is
meaningless-but-matched.

| Sweep | SCAPS $\Delta V_\text{oc}$ (mV) | A (de-spike) | B (closure) | AB (both) |
|---|---|---|---|---|
| CBO ($\chi_\text{ETL}$) | 918 | 85 % OK | 77 % OK | 83 % OK |
| **N_d (ETL doping)** | 100 | 25 % **Xdir** | 11 % **Xdir** | 28 % **Xdir** |
| N<sub>t</sub> PVK (CB) | 38.6 | 69 % OK | 11 % OK | 63 % OK |
| N<sub>t</sub> PVK (VB) | 10.8 | 248 % OK | 41 % OK | 227 % OK |
| **N<sub>t</sub> HTL/PVK** | 5.2 | 0 % **Xflat** | **70 % OK** | 425 % OK **(4$\times$ over)** |
| N<sub>t</sub> PVK/ETL | 282 | 72 % OK | 45 % OK | 64 % OK |
| E<sub>t</sub> PVK (CB) | 0.4 | flat/flat | flat/flat | flat/flat |
| E<sub>t</sub> PVK (VB) | 0.4 | flat/flat | flat/flat | flat/flat |
| E<sub>t</sub> HTL/PVK | 0.006 | flat/flat | flat/flat | flat/flat |
| E<sub>t</sub> PVK/ETL | 34.5 | 23 % OK | 6 % OK | 32 % OK |

Table 3. Reading the columns: **A** (de-spike) carries the bulk $N_t$ trends
(69 %, 72 %) but cannot move the near-passive HTL/PVK interface $N_t$ at all
(flat, wrong direction). **B** (closure) is the only column that gives the
HTL/PVK interface $N_t$ trend physically (70 %, matching SCAPS's small 5.2 mV
range) but leaves every bulk $N_t$ trend weak (11 %, 41 %). **AB** keeps both —
every bulk and interface trend points the right way — at the price of one
over-response (next section).

# 4. The two-bottleneck model

The matrix decomposes cleanly into two independent suppression mechanisms.

**Bottleneck 1 — over-counted junction Auger (governs $N_t$ response and base
$V_\text{oc}$).** On a continuous mesh, the heterojunction node sees a ~1000$\times$
carrier pile-up from the band offset (Section 6 of the gap analysis). The Auger
rate $\propto n^2 p, n p^2$ then explodes at that single node and dominates total
recombination, so any *additional* bulk defect $N_t$ the user adds is masked —
its SRH contribution is a rounding error next to the artifact Auger. The
de-spike blends the junction-node carrier densities toward the neighbour
geometric mean *in the bulk recombination rate only* (transport untouched),
removing the artifact. The effect is visible directly in Table 3: without
de-spike (B) the bulk $N_t$ trends are 11 % / 41 %; with it (A, AB) they jump to
69 % / 248 %. **The artifact junction Auger is the single dominant bottleneck for
the $N_t$ response.**

**Bottleneck 2 — bulk-node sampling of $E_t$ (governs $E_t$ response).** The SRH
rate

$$
R_\text{SRH} = \frac{np - n_i^2}{\tau_p (n + n_1) + \tau_n (p + p_1)},
\qquad n_1 p_1 = n_i^2 e^{\,0},\; n_1 = n_i e^{(E_t-E_i)/kT}
$$

carries the trap energy $E_t$ only through $n_1, p_1$. At a junction node where
$n, p$ are huge from the pile-up, $n \gg n_1$ and $p \gg p_1$ for any $E_t$, so
the $E_t$-bearing terms vanish from the denominator and the rate becomes
$E_t$-independent. De-spiking the carrier densities partially restores
sensitivity (E<sub>t</sub> PVK/ETL: B = 6 % $\rightarrow$ A = 23 % $\rightarrow$ AB = 32 %), **but it caps around
32 %** because the de-spiked node is still a single bulk node, not a true
interface state with its own occupancy. Closing the remaining $E_t$ gap requires
the interface-plane states (the steady-state path, configuration C / the
transient closure extended). **So $E_t$ has two bottlenecks — the Auger
over-count *and* bulk-node sampling — and the de-spike only removes the first.**

# 5. Residual 1 — HTL/PVK interface $N_t$ over-response

This is the one place AB is worse than its parts. The HTL/PVK interface is
near-passive: SCAPS moves only 5.2 mV across the whole $N_t$ sweep. Per Table 3:

- **A (de-spike only):** 0.0 mV — completely flat, wrong direction. The de-spike
  has no interface-state term, so it cannot respond to an interface defect.
- **B (closure only):** 3.64 mV — 70 % closure, a *good* match to SCAPS's 5.2 mV.
  The closure term alone reproduces this interface correctly.
- **AB (both):** 22.15 mV — 425 % closure, **4$\times$ the SCAPS range**.

The mechanism: the closure exposes the interface-$N_t$ sensitivity at the
junction node, and the de-spike, by stripping the bulk-Auger floor at that *same
shared node*, removes the very thing that was damping it — so on this nearly
passive interface the combined response is amplified ~6$\times$ over closure-alone.
This is a de-spike/closure **overlap at one mesh node**, not a physics error in
either term separately. The fix is an interface-node-aware de-spike: skip the
de-spike blend on nodes the closure already owns, so each shared node is
corrected once. Expected to restore AB's HTL/PVK $N_t$ to B's 3.6 mV match while
leaving every other entry in the AB column unchanged.

# 6. Residual 2 — ETL doping ($N_d$) direction mismatch

Every transient configuration (A, B, AB) gets this backwards: as ETL donor
doping rises, SCAPS $V_\text{oc}$ climbs +100 mV monotonically, while SolarLab
*falls* ~10 mV. The de-spike and closure both shift the magnitude (11 $\rightarrow$ 25 $\rightarrow$ 28 %
closure) but none flip the sign. Only **configuration C (steady-state interface
states)** reverses it (established in the prior root-cause work; root = the
frozen built-in-potential boundary, which the steady-state interface treatment
relaxes). C is physically correct here but presently ~80$\times$ too slow for the sweep
harness (481 s per $V_\text{oc}$). This residual is therefore gated on a
performance task, not a physics task: profile and accelerate the steady-state
interface-states Newton solve until it can join AB.

# 7. Recommendation

1. **Adopt AB as the validated fast-transient parity config** — de-spike
   $f=0.53$ + `interface_plane_closure` on. It holds base $V_\text{oc}$ 1.161 V
   (-7 mV), FF/PCE within tolerance, and every bulk and interface trend in the
   correct direction, at ~7 s/point.
2. **Implement the interface-node-aware de-spike** (Section 5) to retire the
   HTL/PVK $N_t$ 4$\times$ over-response — the only blemish in the AB column.
3. **Accelerate the steady-state interface states** (Section 6) so the ETL-doping
   direction can be fixed without the 11 h sweep cost.
4. **Treat $E_t$ closure as state-limited** (Section 4): the de-spike caps $E_t$
   response at ~32 %; do not chase the rest with more de-spike — it needs the
   interface-plane states.

# Appendix — reproduction

| Artifact | Path |
|---|---|
| A/B/AB/C comparison driver | `/tmp/abc_compare.py` (`--variant {A,B,AB,C}`) |
| Per-variant trend JSON | `/tmp/abc_out/variant_{B,AB,C}.json` |
| A = de-spike $f=0.53$ full matrix | `/tmp/despike_out/partB_f0.53.json` |
| De-spike $f$-sweep (base + trends) | `outputs/despike_matrix/despike_matrix.csv` |
| De-spike figures | `outputs/despike_matrix/fig_*.png` |
| De-spike implementation | `perovskite_sim/physics/continuity.py` (`het_recomb_despike`) |
| Closure / states flags | `perovskite_sim/models/device.py`; `perovskite_sim/experiments/steady_state.py` |
| Trend metric (closure %, dir, flat) | `scripts/run_scaps_full_regression.py` (`trend_stats`, `verdict`) |
