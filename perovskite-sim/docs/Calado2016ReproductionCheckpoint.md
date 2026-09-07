# Calado 2016 Reproduction Checkpoint

Status: Parameter-flow and optical checks passed; the single-ion density matrix
completed at 61/121/241 nodes with an analytic density Jacobian on the supported
research path. Both solvers remain quantitatively different from the paper, 2026-09-06.
The objective is not achieved. No remote push is authorized.
Goal status: blocked pending original Fig. 1 input/protocol/state evidence.
The latest shared-input turnaround audit is at the end of this file. Further
paper alignment needs the original Fig. 1 input/protocol/state, which remains
unconfirmed; do not substitute a fitted physical parameter for that evidence.

## Architecture Constraint

The user explicitly requires preserving SolarLab's physical model and core
numerical architecture. Keep the existing constitutive laws, coupled
Poisson/carrier/ion model, Scharfetter-Gummel discretization, Method of Lines
and Radau architecture. MATLAB/Driftfusion is an independent reference only.
Do not replace the production engine with pdepe or a different DAE solver.
Parameter-flow repairs, explicit measurement protocols, numerical
implementation diagnostics and verification remain in scope.

## Objective And Acceptance

Reproduce both the numerical J-V curves and the hysteresis trends in
Calado et al., *Evidence for ion migration in hybrid perovskite solar cells
with minimal hysteresis*, DOI: 10.1038/ncomms13831. Also demonstrate that
changing physical parameters in the real frontend changes the corresponding
backend state, fluxes, and computed hysteresis through a controlled matrix.

Completion requires all of the following, not only a passing loader test:

- Fig. 1e and Fig. 1f curve comparison with the paper's parameters and an
  explicit preparation, scan, illumination, and sampling protocol.
- The reported Fig. 1e HI of 0.00 and Fig. 1f HI of 1.84, using the paper's
  definition, with tolerances set from curve extraction and numerical
  refinement before accepting a result.
- Controls with no mobile-ion inventory and frozen ions, and the contact-SRH
  on/off comparison; no numerical artifact may be called ionic hysteresis.
- A density/diffusivity/scan-rate/contact-SRH matrix whose interpretation is
  supported by ion redistribution, current components and numerical checks.
- Actual frontend edit -> job request -> backend material arrays -> solver
  -> displayed result evidence, including explicit zero and empty inputs.
- Grid, voltage/time sampling and solver-tolerance checks. Failed or
  unconfirmed states and discarded points must remain visible in evidence.

## Sources Read

- Main paper: Zotero attachment `3Q93V5FL`, 10 pages. SHA-256:
  `e9a3ca554f96cf2a8d0b5b1118957bb3252d0a727c553329633158acc800218e`.
- Supplementary Information: 18-page PDF from the local paper-source cache,
  copied to `outputs/calado-reproduction/sources/calado2016-supplement.pdf`.
  SHA-256: `7c4e8cddba46815347ec2aab3b149dd282b23885d3f383e11ab56a4571083f3b`.
- Publisher: <https://www.nature.com/articles/ncomms13831>.
- SI tables 1-3, notes 1-3, and figure captions 1-12 were read. Main-paper
  Fig. 1 was rendered and inspected. Its filled vector-curve outlines have
  now been extracted using `scripts/extract_calado_fig1_reference.py`, with
  separate axis calibration for each panel and a stroke-width envelope.

The live Zotero SQLite file was locked. The actual PDF was located directly
in Zotero storage. The publisher SI download was unavailable through the
terminal DNS; the cached PDF was checked for valid PDF structure, page
count, content, and hash. A similarly named HTML error page was rejected.

## Paper Contract

| Quantity | Paper | SolarLab SI mapping |
| --- | --- | --- |
| Architecture | p / i / n, 200 / 400 / 200 nm | HTL / absorber / ETL |
| Mobile species | One positive species, called `a` in the paper | `P` / frontend `c`; not the frontend negative species `a` |
| Ionic background | Uniform negative static density `Nion` | `P0` / `c0`, enters `q(P-P0)` |
| Ionic density | 1e19 cm^-3 | `P0 = 1e25 m^-3` |
| Ionic mobility | 1e-12 cm^2/(V s) | `D_ion = mu * kT/q = 2.585e-18 m^2/s` at 300 K |
| Electronic mobilities | 20 cm^2/(V s), all layers | `mu_n = mu_p = 2e-3 m^2/(V s)` |
| DOS / gap | 1e20 cm^-3 / 1.6 eV | `ni = N0 exp(-Eg/(2kT))` |
| Doping | 3e17 cm^-3, p/n contact layers | `N_A` / `N_D = 3e23 m^-3` |
| Permittivity / built-in voltage | 20 / 1.3 V | `eps_r = 20`, `V_bi = 1.3` |
| Bimolecular coefficient | 1e-10 cm^3/s | `B_rad = 1e-16 m^3/s` |
| Bottom-cathode contact SRH | tau_n = tau_p = 2e-15 s | Contact-volume SRH in both 200 nm contacts |
| Top-cathode control | Contact SRH absent | Must verify negligible/exactly disabled SRH |
| Trap levels below CB | p-contact: 1.4 eV; n-contact: 0.2 eV | `n1`, `p1` from Boltzmann DOS |
| Generation | Uniform 2.5e21 cm^-3/s in i-layer | `G = 2.5e27 m^-3/s`, absorbed current about 160 A/m^2 |
| External carrier BCs | J_n(0)=0, n(L)=n0; p(0)=p0, J_p(L)=0 | Mixed blocking-minority / fixed-majority conditions |
| Ionic BCs | Confined to intrinsic layer, no sources/reactions | Blocking ionic flux at both absorber boundaries |
| Spatial method | MATLAB pdepe, uniform 0.67 nm; 1200 points | External reference only; SolarLab retains SG/MOL/Radau |
| HI | Pmax,reverse / Pmax,forward - 1 | Different from current SolarLab normalized HI |

The paper's `a` is positive. Adding frontend `a0` and `D_a` would introduce a
second mobile species absent from the paper. Table 3 prints an electron
charge differing from the physical constant; it notes higher-precision
constants were used. Do not tune physical constants to compensate for a fit.

## Confirmed Differences To Investigate

1. Main Methods specify experimental dark preparation at -1 V for about
   30 s, then illumination and a -1 -> +1.2 V scan at 0.04 V/s, shutter
   closed during the high-bias dwell, then illumination and reverse scan.
   The simulation protocol is called similar, not specified step-for-step.
   Existing `plot_calado_fig1f.py` instead prepares at dark 0 V for 120 s,
   ramps illumination at 0 V, walks to -1 V under illumination, and keeps
   illumination on during the 3 s high-bias dwell. These are physical
   history changes, not harmless numerical preparation.
2. The frontend transient driver uses 0 -> Vmax -> 0 and a short illuminated
   carrier preconditioner. It does not currently execute the paper protocol.
3. The preset uses all-carrier Dirichlet contacts, while SI equations 29-32
   specify zero minority-carrier flux at the outer contacts.
4. The preset approximates uniform generation with weak Beer-Lambert
   absorption. The external script overrides it; the ordinary UI does not.
5. Existing paper plots discard isolated solver outliers in postprocessing.
   This does not certify the state history or pointwise numerical solution.
6. Existing direct source inspection shows `P0` and `D_ion` reaching node
   arrays. `Number('')` makes an empty positive numeric field zero; an empty
   negative optional field is omitted and defaults to zero. Full UI request
   and solver evidence is still required before accepting the linkage.
7. Full-mode selected-layer reads currently reconstruct only `device` and
   `layers`, dropping top-level grid/hint metadata. Assess before using a
   paper-specific grid contract through the frontend.

## Verified Progress

### Paper Reference

`outputs/calado-reproduction/reference/reference.json` and four CSVs now
contain independently extracted vector-curve references, not a solver fit:

| Panel | Forward Pmax (W/m2) | Reverse Pmax (W/m2) | Extracted HI |
| --- | --- | --- | --- |
| Fig. 1e | 153.5721 | 153.5592 | -0.0000843 |
| Fig. 1f | 28.8509 | 81.9116 | 1.83914 |

The Fig. 1f graphical-line-width HI interval is [1.73816, 1.94413]. This is
graphical uncertainty only. The extracted curves were plotted and visually
checked against the original figure. Forward and reverse Voc in Fig. 1f are
0.49911 V and 0.74547 V; forward Jsc is 134.955 A/m2, reverse 161.803 A/m2.

### Frontend Fixes And Trace

- Fixed the stale mount-time physics tier in `device-panel.ts`. The active
  config now determines field gates and the Full-mode layer builder.
- Fast/Legacy edits now notify the workspace; its stored tier and badges
  follow the config. The previously stale mode could leave negative-ion
  controls editable while the backend correctly ran Legacy and disabled them.
- Full-mode selected-layer reads now retain top-level grid and hint metadata.
- 21 focused frontend tests passed: `device-panel-ion-mode.test.ts`,
  `device-panel-workspace-config.test.ts`, `config-editor-dual-ions.test.ts`.
  The runner took 404.49 s, mostly environment setup; test execution was 330 ms.
- Real browser -> real `/api/jobs` -> result checks ran using the actual
  DevicePanel and JVPane in `frontend/tests/ion-flow.html`. This isolated
  fixture does not read or overwrite the user's saved workspace.
- Explicit zeros and keyboard-cleared positive-ion fields both submitted
  zero `D_ion/P0` in all layers and returned HI about 6.14e-12.
- Restoring absorber `D_ion=2.585e-18`, `P0=1e25` submitted those exact values
  and returned SolarLab HI -0.0199825 under the ordinary UI protocol.
- Raw requests and returned arrays are saved as `no-ions-frontend.json`,
  `empty-ions-frontend.json`, and `nominal-frontend.json` in
  `outputs/calado-reproduction/ui-protocol/`.
- Browser automation's empty-string fill was a no-op; actual keyboard
  selection/deletion was used and the empty DOM value was checked explicitly.

### Direct Backend Controls

`scripts/audit_ion_parameter_flow.py` records configured values, compiled
arrays, inventories, per-point statuses, current components and raw profiles.
At requested N=60, 31 points, 1 V/s, Vmax=1.2 V, ordinary UI protocol:

| Case | Paper-definition HI | Status |
| --- | --- | --- |
| No ions, all D and initial populations zero | 6.14064e-12 | Completed |
| Frozen ions, original uniform population, D=0 | 4.97747e-9 | Completed |
| Original Calado mobile-ion parameters | -0.0195911 | Completed |

The two HI definitions explain the nominal frontend/backend scalar
difference. These runs do not constitute paper reproduction or a completed
physical parameter matrix.

### Paper-Alignment Probes

`scripts/probe_calado_paper_alignment.py` changes only diagnostic input
parameters and orchestration; production equations and numerical kernels are
unchanged. The `paper` contact option uses existing Full-mode mixed-contact
support: Sn_left=0, Sp_right=0, majority contacts Dirichlet.

- Dark -1 V for 30 s, uniform generation, 0.04 V/s, 3 s dark high-bias hold,
  mixed contacts, tanh N=60: completed, HI=0.434233, elapsed 158.36 s.
- Same conditions, uniform 121 nodes: completed, HI=0.380531, elapsed
  300.34 s. This 6.67 nm mesh is coarse and not paper-resolution evidence.
- Same tanh N=60 case without contact SRH: failed at dark -1 V startup after
  recursive subdivision, with overflow/non-convergence; elapsed 476.89 s.
  This is a numerical startup defect to investigate, not a physical curve.
- Protocol/contact corrections alone have therefore not closed the main
  numerical discrepancy. No outlier deletion was used in these new probes.

### Driftfusion Reference Supplied By User

Repository: <https://github.com/barnesgroupICL/Driftfusion>.
The README identifies the main release as the 2019-2022 code and associates
the 2022 paper with v1.1.1. An exact 2016 source commit has not yet been
identified. Terminal Git access fails DNS, but public source files were read
through the web tool:

- `Core/df.m`: coupled pdepe state (V,n,p,c,a), single positive species when
  N_ionic_species=1; finite-site correction in diffusion; zero ion sources.
- `Core/pc.m`: q=1 in electron-charge units and e=1.60217662e-19 C, confirming
  that the SI table's printed charge should not become a fitting parameter.
- `Protocols/equilibrate.m`: separate electronic and ionic stabilization,
  accelerated ions during equilibrium preparation, then K_a=K_c=1 restored.
- `Protocols/doJV.m`: freezes ionic mobility during a 1 ms light ramp,
  and uses a continuous time-dependent voltage sweep, not a staircase of
  separately restarted fixed-voltage solves.

Installed MATLAB R2026a was found. Normal and non-JVM terminal batch
startups both failed with a Qt NEON compatibility error and restricted
preference/crash paths. After the user manually started MATLAB, Computer Use
listed `com.mathworks.matlab` as running, but refused application control
(`Computer Use was not approved to use MATLAB`). The bundled Python Engine
supports Python 3.13; importing/discovering sessions with its native modules
failed at `SharedMemory:find:shm_open`, permission error 1. Manual startup
does not resolve these session permission restrictions. No independent
MATLAB numerical result has yet been obtained.

A continuous-waveform probe was added using the existing `run_transient`
callable-bias interface, with an optional explicit dark-0 V initial history.
Its first run completed preparation but then failed due to an incorrect
positional call (`t_eval` supplied twice). The corrected keyword call completed:
`alignment/paper-contacts-ramp-dark0-seed-n60-v2`, HI=0.47864475, 88.02 s.
The original failed artifact is not a numerical convergence result.

### Potential Profiles And Follow-up Probes

Fig. 1h's eight electrostatic-potential profiles have now been extracted from
the original PDF vectors. The plotted electron energy is sign-reversed to
compare to SolarLab phi. Axis ticks set the calibration, and overlapping
stroke outlines define the graphical envelope. Source PDF and compared NPZ
hashes are recorded in `reference/potential-reference.json`; individual CSVs
and `reference/potential-comparison.png` preserve the complete profiles.
Four geometry tests pass, including rejection of disconnected paths and
explicit opt-in to the envelope of multiply intersecting strokes.

At forward 0 V, the paper's central-absorber field (250-550 nm) is about
4.31e5 V/m, versus about 2.90e5 V/m at N=60. The midpoint potential agrees
closely, but the slope does not. This is a direct field discrepancy, not
merely an HI-definition mismatch. The nominal 1.1 V source profiles have
right-hand plateaus corresponding to roughly 1.08 V; their legend rounding
must be respected before setting high-bias pointwise acceptance limits.

Additional completed probes, all retaining production physical equations:

| Output under `alignment/` | Paper-definition HI | Elapsed (s) |
| --- | --- | --- |
| `paper-contacts-ramp-dark0-seed-n100` | 0.48190310 | 253.32 |
| `sensitivity-half-D-ramp-n60` | 3.35681421 | 72.38 |
| `paper-contacts-ramp-no-hold-n60` | 0.46692818 | 87.66 |

The half-D case deliberately changes the input diffusivity and is sensitivity
evidence only, not paper reproduction or a proposed parameter correction.
The N=60/100 pair does not replace a full convergence ladder. Mobile-ion
inventory drift across the nominal N=60/100 and half-D scans is at most
1.4e-15 relative; no ion loss is evident. The N=60 nominal run has a tiny
negative P minimum (-1.99e-7 m^-3), below its scalar absolute tolerance;
this is recorded rather than hidden as exact positivity.

The paper's simulation Methods say that reverse inherits forward's final
state, whereas the experimental protocol describes a dark high-bias hold.
Both orchestration choices were tested explicitly; neither closes the gap.

Correction to earlier failure localization: the no-SRH continuous-ramp
artifacts (with and without a 1 ms dark voltage ramp) contain `P_prepared`.
Dark preparation therefore succeeded; the first illuminated sample failed.
The added dark voltage ramp did not resolve it (302.05 s, nonfinite Jacobian).
The initial constant-bias sample now uses the existing fixed-bias recovery
ladder instead of bypassing it through the callable-bias branch. That control
still fails after bisection at the first light-on point (413.96 s), so this
orchestration repair is not claimed as a solver fix. New runs record the exact
stage/attempt and preserve partial branch arrays and the prepared state.

The light-on failure was subsequently overcome without changing the physical
waveform or core algorithm, and independently checked:

| No-contact-SRH control at N=60 | HI | Elapsed (s) |
| --- | --- | --- |
| Scalar atol=1 m^-3, unchanged dwell | 0.00094163747 | 113.05 |
| Scalar atol=0.1 m^-3, unchanged dwell | 0.00094172110 | 143.89 |
| Original atol=1e-6 m^-3; first 1e-8 s integrated separately | 0.00094165069 | 364.13 |

The last option partitions the same instantaneous light-on dwell into
1e-8 s plus the remaining duration; it does not ramp illumination, freeze
ions, change mobilities, or alter equations. The strict-partitioned and
atol=1 paths agree over 0-1.1 V to 1.27e-4 A/m2 in current, 1.30e-10 V in
potential, and 1.49e-11 in ion density normalized to P0. For the hysteretic
case, atol=1 gives HI=0.47864472460 (61.36 s), versus 0.47864474775 at
atol=1e-6: the tolerance setting does not close the paper discrepancy.
No production tolerance default or integration kernel was changed.

The control gives forward/reverse Voc=1.068524/1.068534 V and
Jsc=160.3799/160.3806 A/m2. The extracted paper has Voc about 1.0689 V and
Jsc about 161.79 A/m2. The small current-scale offset and coarse 20 mV
voltage sampling remain unresolved, so near-zero HI is not a full Fig. 1e
pointwise certificate. `control_stack` also still uses tau=1 s as a negligible
SRH approximation, not an exact zero-SRH switch. Its mobile population changes
substantially during the scan (max node change / P0 about 1.41), while total
inventory is conserved to 3.6e-15 relative. Thus negligible hysteresis does
not mean that ions are frozen.

`comparison/jv-comparison.png` overlays the two simulated cases on the paper
vectors, using matching N=60 and atol=1 settings. The corresponding JSON
records hashes and pointwise errors without current rescaling or removing
outliers. Fig. 1f's forward-branch RMSE is 51.28 A/m2 over the plotted
reference interval, making the remaining discrepancy explicit. The final
extractor/helper checks passed 12 focused tests; Calado YAML parsing was
compared against HEAD and confirms that only comments changed, not values.

Upstream version tracing additionally checked the official 2018 wiki:
<https://github.com/barnesgroupICL/Driftfusion/wiki/Procedures> documents the
older `pinParams` / `pindrift` API, distinct from current `pc` / `df`.
<https://zenodo.org/records/6045592> explicitly archives v1.1.1 for the 2022
paper, not a confirmed 2016 reproduction package. The modern example
`3_layer_test.csv` does not contain the 2016 SI parameter set. No exact
2016 source/deck has been certified.

Another workflow advanced HEAD to `69a2cc4` and is removing the autoloop
subsystem. Those changes are not this audit's work. Inspection of its current
`solver/mol.py` diff showed removal of the default-off generated-lever hook;
this audit has not edited physical equations or core numerical modules.

## Outstanding Work

- Validate the successful light-on startup procedures over finer spatial,
  voltage/time and tolerance ladders before integrating a production protocol.
- Align explicit preparation and voltage/illumination waveforms against the
  reference code, keeping physical time and ion mobility changes visible.
- Freeze the reference code version and obtain an independent reference run.
- Close both Fig. 1e and Fig. 1f numerical curves, not only the HI scalar.
- Perform the full physical parameter matrix through the frontend/backend
  route, including valid negative-ion-mode cases as extensions of the
  single-positive-ion paper model.
- Establish spatial/time/tolerance convergence and complete the objective
  audit. The goal is active and is not achieved.

## Latest Refinement And MATLAB Handoff

The follow-up grid probe `paper-contacts-atol1-n200` completed at 199 actual
nodes (715.68 s). HI=0.48377888, forward Pmax=55.14449 W/m2 and
reverse Pmax=81.82223 W/m2. N=60/100/200 HI is 0.47864/0.48190/0.48378;
the forward power remains near 55 W/m2, versus paper 28.85087 W/m2.
This makes the unresolved forward-branch discrepancy much larger than the
observed refinement changes. It is still not a complete convergence certificate.

The voltage sampling probe `paper-contacts-atol1-dv5mV-n60` completed with
441 points per branch (106.72 s), HI=0.47918982. Both branches retained a
fixed 0.5 s initial illuminated dwell, using the new explicit
`--light-on-dwell-seconds` option so that refinement does not change history.
The historical 20 mV result is HI=0.47864472; voltage sampling is not the
main cause of the paper discrepancy.
At common voltage samples, the maximum potential difference between 20 and
5 mV sampling is 5.52e-13 V, and the maximum current difference is
1.44e-5 A/m2 over both complete scans. The small HI change is therefore
mostly maximum-power sampling, not a changed ionic trajectory.

`scripts/run_calado_driftfusion_reference.m` is a new independent MATLAB
adapter, not a replacement production engine. It:

- Reads the frozen `paper-contacts-atol1-n60/alignment.json` by default.
- Downloads the official v1.1.1 archive, verifies Zenodo's published MD5
  `b8d2fd78d9467967af94c4f617ffea92`, and extracts a fresh tree per run.
- Calls upstream `df` and `dfana` unchanged, with explicit SI-to-cm unit
  conversions and source-matched ni, n1/p1, doping, mobilities and V_bi.
- Uses one positive species, flat transport bands, uniform generation and
  the recorded preparation/scan/illumination sequence. An explicit frozen-ion
  electronic initialization precedes the 120 s initial dark history.
- Blocks minority contacts. Finite majority S=1e8 cm/s approximates the
  SolarLab Dirichlet limit and must be independently refined before accepting
  solver agreement. No 2016-source identity is claimed for this 2022 release.
- Saves prepared and forward/reverse raw `.mat` states, CSV current curves,
  separate upstream and left-contact current readouts, ion inventories,
  density minima, source/adapter hashes, stage status and failure reports.
- Keeps its files under ignored `outputs/calado-reproduction/driftfusion/`
  and restores MATLAB's prior path and working directory on exit.

The standalone MATLAB R2026a `mlint` executable runs in this session; the new
adapter passes it with no messages. This is syntax/static checking only:
the adapter has NOT yet been executed by MATLAB, and no independent numerical
result exists. Terminal network access still cannot resolve Zenodo. Browser
inspection of the archive works, but its download is blocked by the browser.
The adapter's explicit `websave` step is for the user's MATLAB session.

In the open MATLAB Command Window, first run the coarse API/connectivity check:

```matlab
addpath('/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/perovskite-sim/scripts')
run_dir = run_calado_driftfusion_reference("", LayerPoints=[30 60 30]);
```

After the coarse run succeeds, calling the function without `LayerPoints`
uses 1201 nodes at approximately 0.667 nm. The coarse run is not paper-grid
validation. The folder returned in `run_dir` is the evidence handoff; the
agent can read it without requiring desktop control.

The inspected v1.1.1 `pc.m` sets AbsTol=1e-6 in native cm^-3 for the carrier
states, corresponding to 1 m^-3. Its electrostatic state also receives this
scalar tolerance in volts, unlike SolarLab's eliminated Poisson state. This
unit conversion supports the density-tolerance diagnosis but does not make
the two solvers' full error-control policies identical.

## Frontend Waveform Integration

The actual J-V operation now offers an explicit research-history template.
The ordinary staircase remains the default. `experiments/waveform_jv.py`
uses the existing material builder, fixed-bias recovery ladder, callable-bias
Radau integration and current decomposition. It does not overwrite device
mobilities, populations, lifetimes, temperature or contact parameters.

The versioned waveform document declares the start voltage, dark-0 seed,
dark prebias, branch-start dwell, high-bias hold and its illumination. The
source selector distinguishes device optics from an explicit uniform absorber
generation rate; null and zero have different meanings. Uniform generation
is integrated over the existing dual cells so its total is exactly q*G*L,
unlike the earlier diagnostic script's node-sampled boundary approximation.
This source quadrature difference is explicit, not a parameter fit.

Both `/api/jv` and `/api/jobs` accept the waveform. J-V, current decomposition
and spatial views share the same execution path and retain its history,
protocol hash, both HI definitions, source budget and inventory checks.
The result is labelled `finite_time_diagnostic`: it does NOT claim the
standard driver's local branch certificate or a full grid certificate.
Incompatible algebraic/charged-interface combinations fail before a worker
is submitted. The DC external-circuit endpoint refuses this new option.

Spatial export now carries optional `P_neg`. The plot supports forward/reverse
selection and one voltage sample at a time, with n, p, c and a curves. The
previous plot displayed forward n/p only despite saving ion arrays. Full-mode
layer editing now also has an expandable Device settings section for Mode,
T, Phi, built-in potential and Robin contacts, with edits read back correctly.

Real-browser checks through the actual DevicePanel, JVPane and backend:

| Saved evidence in `ui-protocol/` | Result |
| --- | --- |
| `waveform-nominal-frontend.json` | HI_P=0.47949251; SolarLab HI=0.32409256; exact generation budget 160.2176634 A/m2 |
| `waveform-cleared-frontend.json` | Keyboard-cleared D_c/c0 submit as zero in all layers; HI_P=2.97e-7; zero ion-inventory drift |
| `waveform-negative-spatial-frontend.json` | Positive population zero; a0=2e24 and D_a=1e-18 reach P_neg profiles; five samples in each branch; negative inventory drift 4.81e-16 |

The nominal browser case used Full mode with the preset's default all-carrier
Dirichlet contacts, not silently substituted mixed contacts. The negative-ion
case is a short 0-0.2 V linkage test, not a paper J-V curve or a valid Pmax/HI
measurement; HI_P is correctly reported as undefined. Its actual reverse
profile and negative-ion trace were inspected in the browser.

Desktop (1920 px) and an actual 375 px iframe viewport were visually checked.
The frontend-focused suite passed 75 tests, TypeScript checking passed, and
34 focused Python waveform/API/spatial tests passed. Spatial-export tests
that referenced a removed preset now use the retained Calado anchor; no
deleted preset was restored. This is targeted coverage, not a full-suite pass.

One earlier zero-ion browser attempt lost its client trace when editing the
test HTML triggered Vite reload. It is not accepted as evidence. The later
keyboard-cleared control has a captured job ID and a completed response;
the fixture now exposes submitted job IDs to make observation recoverable.

The paper's quantitative hysteretic discrepancy and the complete physical
parameter matrix remain open. No independent MATLAB output directory exists
yet. No physical constitutive law, SG flux, Poisson kernel or core integrator
was modified by this work. New work remains uncommitted and unpushed.

## Legacy Source And Additional Exclusions

The `2018-EIS` branch is accessible in the browser. Earlier raw-URL cache
misses did not establish that the branch was absent. Its history was imported
with October 2018 committer dates, so date filtering did not locate the older
author lineage. File-specific history led to the initial commit labelled
"Code received from Phil Calado":

- <https://github.com/barnesgroupICL/Driftfusion/tree/c46f54b88a9ca4d64b762b229e6c6aa243ac771d>
- `pinParams.m` has the 200/400/200 nm homojunction and mobility 20 cm2/Vs
  for both electronic species in all layers. Its demo defaults are NOT the
  paper's full deck: ionic mobility is 1e-8, SRH is disabled and scan endpoints
  differ. No original 2016 workspace/measurement script has been certified.
- `pindrift.m` identifies author edits through July 2017. Its inspected
  equations use state (n,p,a,V), constant unit time prefactors for densities,
  pure drift-diffusion ions in the absorber, zero ionic flux outside, bulk
  contact SRH and BC=1's blocked minority/fixed majority contacts. The voltage
  ramp is `Vstart + (Vend-Vstart)*t/tmax`; no rate denominator defect was found.
- Crucially, `pinParams.m` really sets current-conversion `e=1.61917e-19 C`;
  this is not merely a typo confined to the SI table. The separate PDE charge
  variable is q=1 in electron units. Relative to SolarLab's physical Q, that
  output convention scales all currents by 1.0106064248 but cancels from HI.
  It plausibly explains the approximately 1% plateau offset, not the HI gap.
  Under the stated 1200-node uniform-grid/midpoint-generation interpretation,
  the estimated legacy plateau is 161.78196 A/m2, consistent with the extracted
  161.79 A/m2. This grid interpretation is an inference, not an executed result.
  SolarLab's physical Q has NOT been changed.
- The old core's `calcJ=2` evaluates drift/diffusion at the middle mesh point,
  despite comments calling it a boundary current. Its repeated `odeset` calls
  replace prior options. These observations require a real independent run;
  they do not establish the cause of the published hysteresis discrepancy.

Two additional runs completed without any production-kernel changes:

| Diagnostic output under `alignment/` | HI | Elapsed (s) |
| --- | --- | --- |
| `author-precision-ramp-n60` | 0.4802197563 | 52.38 |
| `neutral-contact-background-ramp-n60` | 0.4786447323 | 18.58 |

The first computes unrounded ni, doping and n1/p1 from the legacy source's
N0/workfunction/trap formulas, adopts its absolute permittivity via an explicit
diagnostic eps_r conversion, uses the paper's ionic mobility and removes the
negligible absorber SRH approximation. The second adds immobile positive ions
and exactly compensating static backgrounds to the contacts. Neither explains
the gap, and neither changes the retained production preset's parameter values.

Supplementary S5b's six positive-ion curves have now been extracted with
source hash and graphical stroke envelopes into `reference/figS5b_*.csv` and
`reference/ion-profile-reference.json`. `reference/ion-profile-comparison.png`
was visually checked. Against the N=200 run, forward 0 and 0.4 V density RMSE
is 0.00874 and 0.00710 times 1e25 m^-3 over the compared intrinsic-region
window. Reverse profiles differ more. The ion legend's 1.0 V versus caption's
1.1 V is explicitly retained as a source-label caveat. Agreement in density
alone does not certify the residual electric field or J-V response.

The extractor/helper suite again passed 12 focused tests. All local probe
processes completed. Independent MATLAB output is still absent, and a fresh
Computer Use check again returned `Computer Use was not approved to use MATLAB`.
This same external execution dependency has persisted through at least three
consecutive goal turns since the MATLAB adapter handoff. The remaining
quantitative discrepancy cannot currently be assigned to source history,
reference discretization or SolarLab implementation with defensible evidence.
Further model changes or a fitted diffusivity would not establish the requested
same-model reproduction. The next required evidence is the user-run independent
comparison (command above), including a runtime error if the adapter needs repair.

New runs and source files go under `outputs/calado-reproduction/`. Preserve
the existing staged preset cleanup and do not recreate the deleted presets.

## MATLAB Startup Compatibility Repair

The user's first manual run with `LayerPoints=[30 60 30]` stopped at line 19:
MATLAB could not resolve `java.io.File`. This occurred while resolving the
source JSON path, before downloading Driftfusion or starting any simulation.
It is an adapter startup failure, not evidence of a numerical-solver failure.

The adapter now obtains the absolute input path using MATLAB `fileattrib`.
The second explicit Java dependency, `java.security.MessageDigest`, was also
replaced by streaming MD5/SHA-256 checks through `/usr/bin/openssl`, available
on this Mac. Paths are POSIX-quoted and passed through input redirection;
nonzero command status or malformed hashes fail closed. The pinned release
checksum and source/code provenance checks remain mandatory. This Java-free
checksum route requires macOS/Linux with `/usr/bin/openssl`.

Verification: R2026a `mlint` found no syntax errors; its only message advises
the newer `filePermissions` API instead of `fileattrib`. Ten OS-level digest
comparisons passed against independent Node crypto hashes (empty, text,
binary and the real source JSON through relative/absolute paths). Nine POSIX
path-quoting cases and missing-input failure handling also passed. These
checks do not constitute execution of the wrapper in MATLAB.

No physical parameters, equations, mesh settings or numerical integration
methods were changed in this repair. A successful independent MATLAB solver
run and the quantitative paper-curve comparison remain outstanding.

## Verified MATLAB MCP Connection

The resumed goal audit obtained new live evidence rather than assuming the
previous installation failure persisted:

- `codex mcp get matlab --json` now reports an enabled STDIO server at
  `~/.matlab/agentic-toolkits/bin/matlab-mcp-server`, configured for
  `--matlab-session-mode=existing`, the Simulink tool extension, disabled
  telemetry, and a 600 s tool timeout. The binary and extension file exist.
- The real `detect_matlab_toolboxes` MCP call succeeded: MATLAB R2026a
  Update 3 (26.1.0.3276743), Simulink R2026a, MATLAB MCP Server Toolbox 0.4.0,
  and Java 11.0.32.1+10-LTS from Amazon Corretto. The read-only connection to
  MATLAB is verified, not merely the presence of a configuration file.
- The real `check_matlab_code` MCP call on the reference adapter succeeded.
  Its only diagnostic is the informational `fileattrib` replacement advice
  at line 21. This is not a syntax error or evidence of solver execution.
- `evaluate_matlab_code` was rejected before execution with
  `MCP tool call requires approval, but approval policy is never`.
  The requested version/arithmetic probe therefore did not run. No alternate
  execution route or permission change was used to evade that decision.
- `outputs/calado-reproduction/driftfusion` is still absent. There is no
  independent numerical result and no live reference job to wait on.

Next action: continue in a client session that permits approval of MATLAB
code execution, first verify a small calculation, then run the coarse
independent adapter. Do not reinstall the working Java/toolkit setup or
replace the physics/core integrator to work around this client restriction.
The full paper-curve reproduction and physical parameter matrix remain open.
This audit changed only this checkpoint; no remote push was performed.

## Frontend Neutral-Ion Controls Before Client Restart

Four new jobs were submitted through the actual DevicePanel and JVPane browser
fixture, not by changing a Python-side device after submission. All completed
with the same Full-mode all-carrier Dirichlet contacts, 61 actual nodes, 111
samples per branch, 40 mV/s ramps, dark-0 seed 120 s, dark -1 V preparation
30 s, illuminated branch dwell 0.5 s and dark high-bias hold 3 s. This is the
frontend investigation protocol, not a newly certified paper reproduction.

| Case | Absorber D_c (m2/s) | c0 (m^-3) | HI_paper |
| --- | --- | --- | --- |
| empty | 0 | 0 | 2.9709869476e-7 |
| frozen | 0 | 1e25 | 2.9709869476e-7 |
| nominal | 2.585e-18 | 1e25 | 0.4794925076 |
| half_D | 1.2925e-18 | 1e25 | 3.3588431426 |

The complete request, job ID, device snapshot and 222 spatial profiles for
each case are saved in `outputs/calado-reproduction/ui-protocol/` as
`waveform-{no-ions,frozen,nominal,half-D}-spatial-frontend.json`.
`scripts/summarize_ion_frontend_matrix.py` checks request/result consistency,
rebuilds the declared grid/material arrays without integrating, verifies the
protocol hash and ionic charge closure, and checks frozen/empty populations.
Its report and inspected plot are in `ui-protocol/control-matrix-20260906/`.

Empty populations remain exactly zero; frozen populations remain exactly at
their reference density. Frozen versus empty HI is identical; their maximum
potential difference is 8.97e-15 V. Spatial states are not bitwise identical:
the initial exact-array comparison rejected this roundoff. The report retains
actual differences and explicitly declared diagnostic tolerances, not a claim
of bitwise equivalence or a convergence certificate. All six diagnostic
state comparisons pass; the maximum recomputed charge-closure discrepancy
over the four cases is 9.32e-10 C/m3. Moving-ion inventory drift is below
1.25e-13. The six focused summary-script tests pass.

These results show that edited c0 and D_c reach the actual backend and alter
ion redistribution/HI. The half-D result is sensitivity evidence, NOT a fitted
replacement for the paper's diffusion coefficient. Density, scan-rate,
negative-ion and optics matrices still need their full scoped coverage.

The user requested a client restart during this work. No further MATLAB call
or new simulation was started after that request. All four browser jobs and
the summary/tests finished, their browser tab was closed, and no agent-started
simulation remains live. Resume only after the user returns: verify MATLAB
code execution first, then run the independent reference adapter. Do not
change permission settings, core physics, or the SG/MOL/Radau architecture.

## First Executed Independent Reference Attempt

After the goal was resumed, the minimal MATLAB MCP code execution returned
`MATLAB_EXECUTION_CHECK=4`. The earlier execution-approval blocker is no longer
the current failure. No client permission setting was changed by the agent.

The adapter was then called with `LayerPoints=[30 60 30]` and the default
completed `paper-contacts-atol1-n60/alignment.json` input. MATLAB downloaded
the official Driftfusion v1.1.1 archive, passed its mandatory pinned MD5 check,
extracted a fresh upstream tree and entered the control-case initial
electrostatic stage. The upstream `pdepe`/`ode15s` integration failed at t=0
because it could not meet tolerances above the minimum step size. The adapter
rejected the truncated/nonfinite result and recorded a failed status.

Evidence: `outputs/calado-reproduction/driftfusion/run-20260906-054956-630/status.json`.
The verified archive is now cached in the parent directory as
`Driftfusion-v1.1.1.zip`. No forward/reverse curve or prepared state was produced;
this attempt is not evidence of a quantitative disagreement between solvers.
The MCP call is terminal, not an ongoing simulation to poll or restart blindly.

Per the newly installed `matlab-debug-code` skill, the agent requested user
confirmation before investigating this MATLAB error's variables/call stack.
Pending confirmation, no debug experiment, parameter change or solver edit
was performed. On approval, diagnose the reference initialization and adapter
first, preserving the user's SolarLab model and core numerical architecture.

## Positive-Ion Density And Scan-Rate Diagnostics

While MATLAB-error debugging confirmation remained pending, six further jobs
were submitted through the real browser controls and completed. No MATLAB
debug inspection or reference-parameter edit was performed in this interval.
The baseline is still the frontend Full-mode all-carrier Dirichlet case,
not the paper's mixed-contact reference. Preparation, illumination, holds,
111 voltage samples, 61 actual spatial nodes, rtol=1e-4 and atol=1 m^-3 remain
fixed. A scan-rate change alters only the ramp durations, not the 0.5 s
branch-start dwell or the 3 s dark high-bias hold.

| New record suffix in `ui-protocol/waveform-*-spatial-frontend.json` | Changed input | HI_paper |
| --- | --- | --- |
| zero-c | c0=0, retaining D_c=2.585e-18 | 2.9709869476e-7 |
| half-c | c0=5e24 | 2.3635309470 |
| double-c | c0=2e25 | 0.2042875429 |
| rate-0p02 | scan rate=0.02 V/s | 0.1983174651 |
| rate-0p08 | scan rate=0.08 V/s | 3.4445829294 |
| double-D | D_c=5.17e-18 m2/s | 0.1986760337 |

The nominal c0=1e25, D_c=2.585e-18 m2/s, 0.04 V/s case remains
HI_paper=0.4794925076. The zero-population result demonstrates that a nonzero
diffusivity cannot generate an ionic population: all saved positive-ion
arrays are exactly zero and the HI matches the earlier empty/frozen controls.

The audit now declares the varied axis. For `--vary ions`, nonionic inputs
and the full scan history must match. For `--vary scan_rate`, only the request's
v_rate may differ. Every result is checked against its own expected protocol,
numerical controls, rebuilt material arrays, and exact uniform-source budget.
The tool also reports the absorber mean field and ionic centroid at the shared
0.4 V sample. This avoids interpreting a common preconditioning peak density
as evidence that changing scan rate had no effect.

At c0=0 the 0.4 V forward/reverse mean-field difference is only 3.62e-7 V/m.
At nominal c0, the mean field is +0.193879 MV/m forward and -0.603381 MV/m
reverse. At 20/40/80 mV/s, the absolute branch field differences are
0.395114/0.797260/1.294205 MV/m. Those state-history changes accompany the HI
changes; they are model-response evidence, not independent validation of the
published curves or a monotonic law for every density/rate regime.

Reports and visually inspected PNGs are saved in `ui-protocol/` under
`density-matrix-20260906`, `rate-matrix-20260906`, and
`diffusivity-matrix-20260906`. The actual frontend reverse-branch selector
and sample slider were exercised at 0.4 V; its inspected screenshot is
`ui-protocol/double-D-reverse-0p4V.png`. All raw requests, job IDs and spatial
arrays are retained. The maximum moving-ion inventory drift in these new
cases is below 9.22e-13; the maximum recomputed charge-closure discrepancy is
1.40e-9 C/m3. Ten focused audit tests pass; `git diff --check` passes.

Only the diagnostic summary script, its tests and this checkpoint were
edited. No production physics, numerical kernel, preset value, or frontend
implementation was changed. All new browser jobs are terminal. The remaining
work includes independent MATLAB initialization diagnosis, quantitative paper
comparison, matrix refinement, negative-ion and optics coverage. Do not fit
D_c or c0 to call the paper reproduced.

## Resumed MATLAB And Optical Diagnostics, 2026-09-06

The user explicitly resumed after restarting the session. Live MCP execution
returned MATLAB R2026a Update 3 and `2+2=4`; execution approval is no longer
the current blocker. Earlier requests for extra debugging confirmation are
historical and do not represent the current authorization state.

### Reference Initialization

The existing three-variable electronic initialization change was tested in
`driftfusion/run-20260906-125850-587`. It still failed at t=0 in the first
electrostatic stage. It was not a verified repair by itself.

The adapter now saves native `parameters.mat` and `initial-input.mat` files,
preserving shapes and precision for isolated runtime diagnostics. Six
single-factor tests used unchanged upstream `df` and `pc` from the verified
v1.1.1 archive. Final diagnostic evidence is in
`driftfusion/initialization-probe-20260906-130526-678/`:

| Initial electronic seed variation | Result |
| --- | --- |
| Original mapped input, no ionic state | Failed at t=0 |
| Disable radiation only in the frozen seed | Completed |
| Extend seed from 1e-12 to 1e-9 s | Failed |
| Increase scalar AbsTol from 1e-6 to 1e-3 cm^-3 | Failed |
| Change Poisson normalization from 1e6 to 20 | Completed |
| Upstream default device, frozen electronic seed | Completed |

The two successful mapped-input variants then completed the same 1 ms
electronic relaxation with transport, radiation and SRH restored. Their
endpoint potential difference is at most 1.1102e-16 V; maximum n/p differences
are 64 cm^-3 against the 3e17 cm^-3 majority scale. This is an initialization
comparison, not a full J-V or discretization certificate.

The adapter now disables radiation and SRH only during the initial 1e-12 s
zero-transport seed, restoring both before electronic relaxation and all
physical preparation/scans. The upstream equilibration source explicitly
allows initially disabling radiative recombination. No upstream solver or
SolarLab production equation was changed. More precise stage labels were
added, and the original Poisson normalization remains unchanged in the adapter.

`driftfusion/run-20260906-130609-882/control/electronic.mat` confirms that this
control-case electronic stage now completes. Adding the ionic state for the
120 s dark-zero history fails at t=0; neither a prepared ionic state nor a
J-V curve was obtained. The hysteretic contact-SRH case has not yet reached
this stage in the adapter.

The separate `probe_ionic_start.m` test started at 13:07:55 local time in
`driftfusion/ionic-start-probe-20260906-130755-477/`. The original normalization
returned a truncated result. The normalization-20 attempt did not return
before the MCP call's 600 s communication timeout. A subsequent minimal
`MATLAB_RESPONSIVE` call was still pending at this checkpoint. A timeout or
the saved `running` JSON is NOT proof that the MATLAB computation stopped.
Do not resubmit this diagnostic until live MATLAB responsiveness/termination
is verified. The user was asked to interrupt `probe_ionic_start` with Ctrl+C,
keeping MATLAB open. Desktop access was denied; no alternate GUI/OS control
was used after that denial. Remaining ionic-start cases have not been run.

### Optical Backend Matrix

`scripts/probe_calado_optical_inputs.py` now attempts all seven cases even
when a worker fails. It preserves each raw error and exits unsuccessfully
with `checks.json.status=failed` if any required case is missing. It does not
emit a passing matrix or plot from incomplete paired results. Two focused
orchestration tests pass, including failures in the second and fourth rows.

All seven real backend jobs were attempted and reached terminal status in
`backend-optics-20260906-complete-attempt/`. These are backend replays of a
captured real frontend request, NOT seven new browser interactions. The
ion-free device, full-mode contacts and scan history remain fixed.

| Case | Forward Jsc (A/m2) | Generation budget (A/m2) | Status |
| --- | --- | --- | --- |
| Baseline Beer-Lambert | 150.52699 | 156.97735 | Completed |
| Double absorber alpha | 295.14833 | 307.80259 | Completed |
| Half photon flux | 75.26348 | 78.48867 | Completed |
| Uniform G=2.5e27 m^-3 s^-1 | 153.56526 | 160.21766 | Completed |
| Same uniform G, alpha=0 | 153.56526 | 160.21766 | Completed |
| alpha=0, device optics | Not available | Zero source requested | Failed |
| Phi=0, device optics | Not available | Zero source requested | Failed |

The uniform-source pair has exactly equal saved forward and reverse current
arrays: alpha is deliberately bypassed by that explicit source selection.
Successful cases have paper-definition HI between 1.56e-7 and 6.03e-7.
These results support optical parameter wiring, not complete optical or
paper reproduction acceptance.

Both zero-source runs exhaust the continuous-ramp RHS budget. An isolated
capture located the alpha-zero failure at reverse -0.70 -> -0.72 V over
0.5 s. Entry state and input request are preserved in
`dark-ramp-probe-1788672079979287000/`. With the same physical ramp and a
20000-evaluation per-subinterval diagnostic cap, two legs failed, four and
eight completed. A follow-up from that saved entry in
`dark-ramp-probe-1788672209184890000/` found sixteen legs completed and
thirty-two failed. Returned currents for 4/8/16 legs were
1.6401e-5 / 1.7085e-5 / 1.5718e-5 A/m2. This is NOT monotonic refinement
or a certified recovery. No subdivision fallback was added to production.

Standalone R2026a mlint found no syntax error in the reference adapter; its
only diagnostic is the existing `fileattrib` modernization advice. The combined
optical-probe, matrix-audit, waveform-driver and waveform-API suite passed
40 tests; `git diff --check` passed. These are focused contracts, not a complete
numerical certificate. No new commit or remote push was made. The paper's quantitative HI/curve discrepancy,
zero-source ramp robustness, frontend optical interactions and full numerical
refinement remain open.

## Completed Independent J-V Reference, 2026-09-06

The user interrupted the unresponsive diagnostic. The live MATLAB MCP check
then returned `MATLAB_AFTER_INTERRUPT=4`. Both earlier 600 s calls are terminal;
their timeouts did not themselves establish termination of the internal solve.
No additional unbounded MATLAB run was submitted.

### Bounded And Observable Reference Execution

New helpers under `scripts/` provide an observation-only copied driver,
cooperative wall-time limits, a per-run `STOP` marker, JSON/text heartbeat
logs, and a native MATLAB waitbar with Cancel support. `latest_trial_time_s`
is explicitly not accepted integration progress. The original `df.m` remains
untouched; the derived copy is mechanically reversible to the original apart
from its function name and one observer call per PDE mesh evaluation. Both
source and derived-driver hashes are saved. This does not change PDE equations,
`pdepe`/`ode15s` options, or any SolarLab production kernel.

Defaults are 30 wall seconds per stage and 300 wall seconds per run; the
limits are checked cooperatively at PDE callbacks, not OS-enforced deadlines.
Standalone ionic diagnostics used 10 s per case. All those calls terminated.
The normalization-20 case repeatedly evaluated near physical time 0.0357009 s
and hit its explicit timeout, replacing the earlier opaque long wait.

`scripts/tests/CaladoReferenceMonitorTest.m` passed 13 MATLAB tests covering
logs, timeout, stop markers, GUI cancellation, source-copy behavior, rejection
of truncated/nonfinite/negative solutions, and electrostatic bias-lift invariants.
Standalone R2026a mlint found no syntax issues; only the previous `fileattrib`
modernization advice remains. An initial GUI-test attempt saw a stale function
signature; the currently loaded signature was checked and the repeated suite
passed. No default-path reset or MATLAB-wide variable clearing was used.

### Source-Aligned Ionic Initialization

The earliest referenced author-code commit was downloaded from GitHub as
`driftfusion/Driftfusion-c46f54b.zip`, commit
`c46f54b88a9ca4d64b762b229e6c6aa243ac771d`, SHA-256
`586066783803b8d13c2fb817a632265823b216502067377c1a6c401626229c96`.
It is a 2018 import with author edits through 2017, not a certified original
2016 reproduction deck. It was read, not executed: its unmodified function
changes root graphics defaults and assigns `sol` into the base workspace.

The legacy implementation initializes the positive ionic state to `NI`
throughout the device and includes an equal static background. In the contacts,
ionic flux is zero and the initial pair cancels. Its plotting code subtracts
the static contact population. This clarifies how its full-domain state
represents the paper's confined mobile species; the SI's piecewise Poisson
equations were also visually checked on supplementary PDF page 15.

The old modern-Driftfusion adapter instead used `Ncat=[0 NI 0]`. The nodal
ionic state and half-grid static background then disagree by up to
5e18 cm^-3 at the interfaces on the coarse grid. Tests under
`driftfusion/ionic-start-probe-20260906-141134-427/` separated this effect:
zero ionic population and uniform frozen population both complete. Matching
the half-grid background to the nodal interpolation returns a finite moving-ion
solution, but with n/p minima near -2.68e13 cm^-3 and c near -8.94e17 cm^-3.
That result is rejected as physically invalid and that background modification
was NOT adopted by the reference adapter. A separate frozen-electrostatic
initialization alone also failed to complete the subsequent moving-ion hold.

The explicit `NeutralContactIonPadding=true` reference option implements the
legacy compensated contact population while leaving contact ionic mobility
zero. It is recorded in metadata and defaults to false; it does not edit the
SolarLab input configuration. At 121 nodes it still failed after about 3.48 s
of physical time (`ionic-start-probe-20260906-142259-817`). At 1201 nodes it
completed the 120 s dark hold with nonnegative densities and relative global
ionic-inventory drift 1.875e-15 (`ionic-start-probe-20260906-142304-345`).
Fine resolution and the declared reference-state layout were both necessary
in these tests. This is not evidence that extra mobile ions belong in the
SolarLab contact layers.

### Bias-Step Consistency

The next failure was the instantaneous 0 -> -1 V boundary step. The prior
potential did not satisfy the new boundary condition. The adapter now adds
the homogeneous Poisson solution to the last potential frame, with voltage
drop weighted by cumulative `dx/epsilon`. Electron, hole and ionic density
arrays remain exactly unchanged and no physical time is added. Tests include
nonuniform epsilon, zero flux divergence of the lift, unchanged earlier frames,
and a no-op when the boundary already matches. The original equations and
core time integration are unchanged. Invalid reference solutions are saved
as `rejected-stage.mat`; density values below -10*AbsTol are not accepted.

### Completed Runs And Remaining Discrepancy

Both control and hysteretic cases completed with all required stage outputs,
positive-density checks and raw MAT/CSV files:

| Run under `driftfusion/` | Spatial layout | Control HI | Hysteretic HI |
| --- | --- | --- | --- |
| `run-20260906-143437-305` | 1201 interface-aligned nodes | 0.00026995 | 0.49499289 |
| `run-20260906-144543-050` | Exactly 1200 global uniform nodes | 0.00026711 | 0.49385747 |

The latter matches the paper's stated node count more literally. Both keep
the frozen source's physical parameters and declared 120 s dark-zero seed,
30 s dark -1 V preparation, 0.5 s light-on dwells, 0.04 V/s continuous scans
and 3 s dark high-bias hold. Neutral contact padding is explicit. The control
still uses tau=1 s as negligible SRH, not an exact zero-SRH switch. Finite
majority-contact S=1e8 cm/s is still an approximation to Dirichlet.

At 1200 nodes the hysteretic forward/reverse Pmax is
54.66779/81.66589 W/m2, Jsc is 154.15610/160.06692 A/m2, and Voc is
0.585470/0.779897 V. The corresponding SolarLab N=200 probe has HI=0.48377888;
the paper extraction gives HI=1.83913698 and forward/reverse Pmax
28.85087/81.91156 W/m2. The main forward-branch discrepancy remains large.

`independent-comparison-20260906/` contains visually checked control and
hysteretic overlays plus `comparison.json` with input hashes and errors.
No current rescaling, fitting, or outlier deletion was applied. Over the paper's
extracted forward-curve interval, MATLAB/SolarLab RMS disagreement is
1.46055 A/m2, while their respective errors against the paper are
50.94193 and 51.54681 A/m2. Reverse MATLAB/SolarLab RMS disagreement is
0.27675 A/m2. These are cross-solver diagnostics, not a full convergence or
physics certificate: common parameter/protocol assumptions can still be wrong.
The paper says the simulated scan was similar to experiment, but does not
fully specify its initial state and exact prehistory. That uncertainty is
not permission to fit a preparation time or diffusivity to the target HI.

The auxiliary left/right current readout discrepancy is retained rather than
ignored. For the 1200-node control its full-scan maximum is about 19.49 A/m2
in the high-injection range, versus about 0.411 A/m2 over 0-1.1 V. For the
hysteretic case the maxima are 1.47e-4 A/m2 forward and 0.07699 A/m2 reverse.
This readout and the spatial/time/tolerance ladders still require validation.

### User-Requested Matrix Figures

`ui-protocol/jv-matrix-display-20260906/` now contains diffusivity, density and
scan-rate loop plots plus full-range companions and `curves-and-audit.json`.
The original spatial endpoints exported x in nm and densities in m^-3; the
plotting helper uses the rebuilt SI grid and checks that mapping. Since those
spatial responses did not export J arrays, currents were reconstructed from
stored n/p/P/phi using the existing current formula, including displacement.
No transient solver was run. The branch-start samples lack prior states and
are omitted explicitly. Recomputed HI agrees with the saved HI to at most
2.53415e-8. Source hashes and single-variable input checks are retained.

All MATLAB simulations and tests started in this update have returned; the
progress windows are closed. No commit or remote push was made. Full paper
reproduction, source/history alignment, numerical refinement, and the outstanding
zero-source optical ramp failures remain open. The goal is not achieved.

## Source History And Zero-Source Progress, 2026-09-06

This section supersedes the earlier zero-source failure status for the
explicitly tested research tolerance of 100 m^-3. It does not close paper
reproduction or establish a general spatial/time convergence certificate.

### Early Author Reference Executed And Audited

The pinned early author source described above was subsequently executed
through `outputs/calado-reproduction/driftfusion/run_legacy_reference.m`.
The observation-only copied driver disables root graphics defaults and base
workspace exports, retains original PDEs and solver options, and uses bounded
stages. The reviewed `v2struct.m` dependency came from a declared public mirror;
its SHA-256 is
`9e38deaec4ce1e2e31c1cea7ea7c6960ac3e09ad92357c24766e56b3f76ba9d4`.
This is still a common-input comparison, not a certified 2016 input deck.

Both cases completed in `driftfusion/legacy-run-20260906-154235-215/`.
`audit_legacy_currents.py` independently compares the native midpoint readout,
direct face fluxes, and carrier-continuity balances at both contacts. The raw
control current has large spikes and gives an invalid apparent HI of 1.75364;
the left/right continuity readouts instead give HI=0.00027434325 and agree to
within 8.31e-8 A/m2. Raw values are retained, not silently replaced. Global
continuity agreement does not establish pointwise PDE convergence.

For the hysteretic case, continuity readout gives HI=0.49385768544, with
forward/reverse Pmax=54.667754/81.665845 W/m2. This closely matches the modern
1200-node result HI=0.49385746844. The paper's HI=1.84 discrepancy therefore
also appears in the early author implementation under these shared inputs.

### Preparation And Scan-Start Hypotheses

`alignment/preparation-convergence-300s-n60/` completed with dark -1 V
preparation extended from 30 to 300 s, leaving the other inputs unchanged.
HI changed from 0.47864472460 to 0.47864484794, only 1.23e-7. In this probe,
insufficient dark preparation does not explain the paper discrepancy.

The peer-review reply explicitly says the authors reran their simulations
at 40 mV/s after a reviewer questioned the earlier 70 mV/s value. Retaining
40 mV/s is therefore required; selecting 70 mV/s to improve a fit is not
justified. The 2018-EIS branch was inspected through immutable GitHub blobs
cached as `driftfusion/2018-*.blob.json`; its README and examples concern the
later impedance paper, not the requested 2016 J-V reproduction.

The diagnostic adapters now allow scan start to differ from dark prebias.
Changing only scan start from -1 to -0.2 V, while retaining dark -1 V/30 s
preparation, yielded:

| Diagnostic | Forward Jsc (A/m2) | HI |
| --- | ---: | ---: |
| SolarLab `alignment/scan-start-minus02-n60/` | 54.81685 | 0.58105287 |
| Early reference `legacy-run-20260906-164107-329/`, continuity readout | 50.24768 | 0.60315396 |
| Paper extraction | 134.95465 | 1.83913698 |

The large low-voltage suppression does not match the paper. This is a
rejected explanatory hypothesis, not a new reproduction preset. The plot's
display interval alone is not evidence for the author's actual scan start.

### Density-Matrix Mechanism Check

The frontend labels now distinguish fixed countercharge c0/a0 from the
separate neutral initialization c_init=c0 and a_init=a0. The density plot
was relabeled accordingly; payload names P0/P0_neg and equations did not
change. At 0.4 V, the saved frontend snapshots give the following forward
values using the existing recombination functions and current readout:

| Population/background | Central mean E (MV/m) | Integrated R equivalent (mA/cm2) | J (mA/cm2) |
| --- | ---: | ---: | ---: |
| 0 | -2.03707 | 0.94043 | 15.08133 |
| 0.5*c_ref | 0.59882 | 11.05660 | 4.96516 |
| c_ref | 0.34091 | 2.42236 | 13.59940 |
| 2*c_ref | 0.18325 | 0.81064 | 15.21112 |

Central E is averaged over x=300-500 nm; x increases from the p contact to
the n contact. Positive E opposes the no-ion collection field here. Absorber
recombination contributes less than 0.001 mA/cm2 in these samples, so the
listed recombination loss is dominated by the contact regions. These are
same-voltage observations, not a measured ionic time constant or a proof of
a universal monotone concentration-HI relation. The four source hashes in
`ui-protocol/jv-matrix-display-20260906/curves-and-audit.json` were checked.

### Zero-Source Arithmetic And Tolerance Evidence

`probe_dark_tolerances.py` restarts from the captured -0.70 -> -0.72 V,
0.5 s entry, not a regenerated history. Its initial guard incorrectly treated
G_optical=None as an invalid zero-source case; the guard was corrected to
evaluate the existing Beer-Lambert fallback. No production generation change
was made. Results are in `dark-tolerances-20260906-v2/`:

- Scalar atol=1 exhausted 20,000 RHS calls; maximum trial time was 0.03485 s.
- Scalar atol=100 and 10000 completed, as did three componentwise policies.
- Terminal carrier densities remained positive in completed probes, but
  microamp-scale face-current variation remained; success alone is not an
  accuracy certificate for these tiny dark currents.

At the saved atol=100 terminal state, the left hole-current calculation
subtracts two terms near 3e23 m^-3. Its binary64 flux quantum is about
6.83e-7 A/m2. Decimal-80 evaluation of the same input state gives
1.688455e-5 A/m2 versus the production 1.640126e-5 A/m2, demonstrating
subtractive precision loss without changing the state or equations.

An isolated algebraic SG rewrite using B(-z)=B(z)+z made the scalar-atol=1
saved segment complete, but the tighter componentwise-0.01 case still failed.
Those diagnostics are in `dark-tolerances-sg-arithmetic-20260906/`.
The SG rewrite was NOT adopted in production, and no log-state transform,
density clipping, physical-model change or solver replacement was introduced.

### Complete Optical Matrices And Research Default

`scripts/probe_calado_optical_inputs.py --atol-m3 ...` now records an explicit,
common numerical override without changing the captured physical inputs.
All seven real backend-worker cases completed and passed paired checks at
atol=100 and 10000 m^-3, in `backend-optics-atol100-20260906/` and
`backend-optics-atol10000-20260906/`, respectively:

- alpha=0 and Phi=0 produce exactly identical complete J arrays and G=0.
- Uniform G and uniform G with alpha=0 produce exactly identical J arrays.
- Doubling alpha increases the absorbed-photon budget and Jsc; halving Phi
  halves the photon budget and approximately halves Jsc.
- Zero-source Jsc is about +/-1.70846e-5 A/m2 for the two directions;
  its power-based HI is correctly undefined. It is not forced to zero.

Comparing the five illuminated cases that completed at atol=1 against 100,
the maximum full-range pointwise J difference is 4.44201e-5 A/m2. Across all
seven cases, the corresponding 100-vs-10000 maximum is 3.00690e-5 A/m2.
Physical requests and voltage arrays were checked equal after removing only
the declared tolerance field. This is scalar-atol stability evidence at one
grid and rtol, not a complete refinement certificate.

The mobile-ion paper-contact N=60 probe also completed at atol=100 under the
original -1 -> 1.2 V history (`alignment/paper-contacts-atol100-n60/`). HI is
0.47864475547 versus 0.47864472460 at atol=1, a change of 3.09e-8. Thus this
numerical tolerance change does not explain or fit the paper's HI=1.84.

The continuous research driver's default density atol is now 100 m^-3,
consistently in the frontend, API fallback and waveform function. Explicit
user tolerances remain unchanged in dispatch and result metadata. The standard
staircase, SCAPS paths and core run_transient default are unchanged. Existing
browser panels that already hold an explicit value of 1 must be edited or
remounted to adopt the new default. Backend replay is not a new browser test.

Final focused checks passed: 46 Python tests covering optical orchestration,
matrix audits, waveform history and API dispatch; 26 frontend ion/waveform
tests; TypeScript no-emit checking; and `git diff --check`. All newly started
processes returned. Port 5173 is still held by Docker, but the terminal HTTP
probe could not connect; the live browser page was not reverified in this turn.

Paper reproduction, ionic matrix refinement and the precision of extremely
small dark currents remain open. No commit or remote push was made.

## Refinement And Fine-Grid Prebias Checkpoint, 2026-09-06 17:38

The source-history check now includes immutable author-source blob
`c18da3f316fc15c02a24155e12634492906af83e`, cached as
`driftfusion/2018-equilibrate_minimal.m.blob.json`. It explicitly constructs
dark short-circuit states at Vapp=0 as initial conditions for subsequent
experiments. It is later code, not proof of the original 2016 J-V history.
The corresponding alternative SolarLab diagnostic, dark 0 V preparation and
scan start -0.2 V, completed in `alignment/dark-zero-scan-minus02-n60/` with
HI=0.44567680181. It does not explain the paper result and was not adopted.

A read-only snapshot of Zotero SQLite plus its WAL/SHM passed quick_check.
The two matching DOI/title records have only the main PDF as attachments;
both PDFs have the previously recorded identical SHA-256. No original
simulation input deck or raw data attachment was found in those records.

### Independent Reference Grid Ladder

The existing bounded adapter completed both control and hysteretic cases at
2400 and 4800 global uniform nodes, without material or history adjustments:

| Nodes | Run under driftfusion | Hysteretic HI | Control HI |
| ---: | --- | ---: | ---: |
| 1200 | run-20260906-144543-050 | 0.49385746844 | 0.00026711165 |
| 2400 | run-20260906-171556-843 | 0.49258358561 | 0.00075947858 |
| 4800 | run-20260906-171909-893 | 0.48977938778 | 0.00088547475 |

All stages returned and passed the adapter's density/finite-output checks.
Over the extracted paper forward interval [-0.19, 0.566] V, successive-grid
maximum J changes were 2.28662 and 0.488448 A/m2, while RMS errors against
the paper were 50.94193, 51.02261 and 51.24817 A/m2. Reverse RMS errors
against the paper remained about 16.2 A/m2. Thus the large paper discrepancy
persists on this ladder. HI changes do not establish an asymptotic order or
a joint space/time/tolerance certificate. In particular, the small control HI
is not monotonically approaching zero and remains visible in the evidence.

### Captured Frontend Density Matrix On Finer Grids

`outputs/calado-reproduction/probe_density_refinement.py` replays the same
four spatial-view requests, preserving all physical inputs and scan history.
Only requested grid size and a common explicit atol=100 m^-3 differ from the
captured requests. Each case runs in its own owned subprocess with a 240 s
limit. SHA-256, returned protocol/controls, charge closure and ionic inventory
are checked by the existing matrix auditor. Raw records and aggregate status
are in `density-refinement-20260906/`.

| Requested grid / actual nodes | No ions | Half c_ref | c_ref | Double c_ref |
| --- | ---: | ---: | ---: | ---: |
| 60 / 61 | 3.11710e-7 | 2.36353090 | 0.47949246 | 0.20428755 |
| 120 / 121 | 5.69703e-8 | 2.41691871 | 0.48294760 | 0.20235037 |
| 240 / 241 | timeout | timeout | timeout | timeout |

All eight completed cases passed their audits. The highest observed ionic
inventory drift was 1.46e-12 relative. The concentration-HI ordering survives
the first refinement, but all four finest-grid cases timed out. The aggregate
status is deliberately `failed`, not a passing three-grid matrix. Timeouts
are not proof that a mathematical solution does not exist.

### Fine-Grid Failure Localized

The no-ion 241-node dark 0 V initialization was profiled both directly and
inside the actual backend worker. Both completed in about 5.6 s with the same
6270 RHS evaluations and identical input DeviceStack. BLAS reported one
thread. Dense LU factorization took about 3.8 s; the observation does not
support blaming frontend parameter loss or worker-thread configuration.

The next hold was isolated through `capture_fine_grid_prebias.py`. The actual
accepted dark-zero state is saved in
`fine-grid-prebias-entry-20260906/entry.npz`, along with its request and stage
metadata in `report.json`. At the subsequent -1 V, 30 s hold, the first Radau
attempt used 99,698 RHS evaluations within the diagnostic's 25 s limit, while
maximum trial time reached only 0.000366938 s. Trial time is not accepted
progress. The diagnostic intentionally terminated there, without claiming a
completed scan or replacing the failed finest-grid matrix.

The next numerical investigation can restart from this saved entry. No new
production solver or physical-model edit was made in this update. All MATLAB,
matrix-worker and diagnostic processes started here have returned. The prior
46 Python / 26 frontend tests were not rerun because this update adds only
diagnostic artifacts and checkpoint text; `git diff --check` was rechecked.
The paper reproduction goal and full matrix refinement remain incomplete.

## Analytic Jacobian And Completed Density Ladder, 2026-09-06

This section supersedes the fine-grid prebias timeout for the supported
single-ion continuous-waveform path. Historical failed runs remain intact.

### Saved-Entry Diagnosis

The saved zero-inventory entry still has D_c=2.585e-18 m2/s in the absorber:
only c0 was set to zero in that single-variable control. The diagnostic
therefore retains the zero-inventory ion transport Jacobian block rather
than setting D_c to zero. Finite-difference checks of the n, p and ionic
blocks were performed with perturbations large enough to resolve the
corresponding potential response in binary64.

`fine-prebias-options-v2-20260906/` compares the same accepted entry, physical
RHS, -1 V/30 s hold, density coordinates and Radau solver:

| Diagnostic | Outcome | RHS evaluations |
| --- | --- | ---: |
| Default finite-difference Jacobian, atol=100 | Budget exhausted | 30001 |
| Finite differences, atol=10000 | Budget exhausted | 30001 |
| Existing componentwise tolerance policy | Budget exhausted | 30001 |
| Algebraic SG arithmetic rewrite alone | Budget exhausted | 30001 |
| Analytic Jacobian, original SG arithmetic, atol=100 | Completed in 1.94 s | 576 |
| Analytic Jacobian plus arithmetic rewrite | Completed in 1.73 s | 549 |

The three initial directional checks had relative L2 errors from 1.6e-10
to 3.3e-8. Separate species-block checks cover all four density settings.
This supports a Jacobian approximation/conditioning issue at the failed
entry; it is not evidence that every possible solver failure has one cause.
The algebraic SG arithmetic rewrite remains diagnostic only.

### Production Integration Boundary

`experiments/waveform_jacobian.py` differentiates the existing charge closure,
discrete Poisson solve, SG carrier/ion fluxes and bulk recombination. Poisson
remains eliminated inside MOL, and the state remains physical densities.
No DAE integrator, quasi-steady carrier replacement, density clipping or new
constitutive law was introduced.

`solver/mol.py::run_transient` accepts an optional density-RHS Jacobian;
omitting it preserves the historical finite-difference default. The fixed
hold helper forwards a supplied Jacobian through its existing subdivision,
time-origin reset and fallback ladder. The continuous waveform driver chooses
the analytic tangent for its verified single-ion Boltzmann slice and records
`jacobian_evaluator` plus `jacobian_fallback_reason` in API results.

Unsupported combinations, including dual ions, selective outer contacts,
interface currents/states, field-dependent mobility, radiative reabsorption,
explicit bulk defects, and configurations that may reach steric clipping,
retain the existing finite-difference path without removing physical terms.
This means the mixed-contact paper-alignment diagnostic is not silently
migrated to the restricted tangent. Standard staircase and SCAPS drivers do
not select the new analytic builder, and their default integration remains
unchanged.

### Completed Production Matrix

Before integration, diagnostic injection reproduced all four 61-node curves
and completed the 241-node nominal case (HI=0.48421136175, 225 Radau
segments). The subsequent production replay uses no diagnostic injection:
four density cases, three grids, the same physical inputs and history,
rtol=1e-4 and atol=100 m^-3. Each owned subprocess had a 600 s limit.

All twelve jobs completed and were re-audited in
`density-production-refinement-20260906/audit.json`:

| Actual nodes | c0=0 | c0/2 | c0 | 2*c0 |
| ---: | ---: | ---: | ---: | ---: |
| 61 | 2.97099e-7 | 2.36353099 | 0.47949249 | 0.20428756 |
| 121 | 2.96245e-7 | 2.41691852 | 0.48294775 | 0.20235032 |
| 241 | 2.70586e-7 | 2.40392305 | 0.48421136 | 0.20318214 |

The last-refinement relative HI changes are 0.541%, 0.261%, and 0.409% for
the three nonzero populations. The ordering survives all three grids.
This is not a Richardson-order estimate or a joint grid/time/tolerance
certificate. The maximum ionic inventory drift is 4.8651e-15 relative.

All stored electron/hole densities and transport-active ionic densities are
nonnegative. Some frozen ionic nodes have small signed numerical deviations;
the largest magnitude is 0.511321 m^-3, below the declared 100 m^-3 atol.
An initial strict raw-positivity audit rejected a -0.027718 m^-3 frozen-node
value; the final audit explicitly distinguishes transport-active positivity
from the declared absolute tolerance for frozen-node invariance. Raw values
are retained and not clipped. These checks must not be described as strict
positivity of every raw floating-point density.

### Optical And Regression Checks

All seven optical backend-worker cases also passed on the production analytic
path using the original captured atol=1 requests, in
`backend-optics-production-jacobian-atol1-20260906/`. The alpha=0/Phi=0 and
uniform-G/uniform-G-plus-alpha=0 complete-array equality checks passed.
The research UI default remains the previously validated atol=100.

The expanded unit/regression selection initially reported missing YAML inputs
from presets deliberately removed earlier. For compatibility checking only,
the two required historical YAML files (nip_MAPbI3 and ionmonger_benchmark)
were extracted from HEAD's parent into a temporary directory; the current
Calado YAML was copied there. No removed preset was restored in the project.
With those temporary inputs, 181 tests passed and 2 default-marked tests were
deselected. Results are recorded in `waveform-jacobian-regression-20260906.xml`.
Frontend checks passed 27 tests, TypeScript no-emit checking passed, and
`git diff --check` passed. This is not a claim that the whole repository's
test suite passes directly with its currently missing historical fixtures.

All jobs and checks in this update returned. No commit or remote push was
made, and unrelated worktree changes were preserved. Full paper reproduction
is still open: the verified density-matrix baseline is about 0.484, the
independent MATLAB result remains about 0.49, and the paper reports 1.84.

## Mixed Contacts, Sampling And Source Audit, 2026-09-06

This section supersedes the earlier exclusion of selective outer contacts
from the analytic waveform Jacobian. It does not close paper reproduction.

### Completed Numerical Checks

The analytic tangent now includes existing mixed Dirichlet/Robin carrier
contacts, including minority-blocking contacts. It uses the production outer
control-volume widths and contact signs; pinned rows/columns apply only to
Dirichlet variables. No new boundary law, density floor, state coordinate or
time integrator was introduced. Dual-ion and other unsupported physical
closures still use the existing finite-difference path.

The paper-contact continuous scans completed at 61 and 241 actual nodes:
`alignment/paper-contacts-analytic-n60/` gives HI=0.47864474005 in 12.54 s;
`alignment/paper-contacts-analytic-n240/` gives HI=0.48402712635 in 256.76 s.
The 61-node analytic/finite-difference HI difference is 1.54e-8.

`density-sampling-refinement-20260906/audit.json` compares 111 and 441 samples
per branch at 121 spatial nodes, with the same physical history and fixed
0.5 s light-on dwells. All four cases completed:

| Density | HI, 111 samples | HI, 441 samples |
| --- | ---: | ---: |
| zero | 2.96244695e-7 | 2.96244679e-7 |
| half reference | 2.41691852 | 2.41916192 |
| reference | 0.48294775 | 0.48365241 |
| twice reference | 0.20235032 | 0.20060633 |

For the nonzero cases the relative changes are below 0.9%. Restricting the
fine-run reconstructed currents to the coarse voltage samples recovers the
coarse HI within 7.23e-8; common-voltage potential differences are below
3.21e-12 V. Thus the observed change is primarily sampled maximum-power
resolution, not changed ionic trajectories. These checks do not establish a
joint asymptotic space/time convergence certificate.

The prior focused regression result is 191 passed, 2 deselected, recorded in
`waveform-selective-reference-regression-20260906.xml`. Historical preset
fixtures were supplied from a temporary directory, not restored to the repo.
This is not a direct whole-repository test-suite result.

### Curve Geometry And Current Convention

`reference-endcap-audit-20260906/` retains the original J-V CSV slices but
distinguishes true centerline voltage domains from graphical stroke endcaps.
Envelope checks include every centerline-domain sample; midpoint errors
exclude cap-intersecting slices. This does not change either paper HI.

`author-current-convention.json` checks the source-supported reporting factor
1.61917e-19 / 1.602176634e-19 = 1.01060642481. The former is used for current
readout in early author code; SI Table 3 lists 1.619e-19 C. Under this
conditional reporting convention, all reference control samples fall inside
the published graphical envelope across the 1200/2400/4800 reference runs.
High-hysteresis forward curves still fail, and HI is unchanged. The original
2016 runtime is not confirmed, so this is not proof of its precise convention.
SolarLab's physical Q and all raw solver arrays remain unchanged.

### Versioned Primary Sources

The two arXiv PDFs were downloaded through MATLAB with bounded web requests,
validated as PDFs, and rendered for visual inspection. Sources and SHA-256:

- <https://arxiv.org/pdf/1606.00818v1>, 40 pages:
  `724e410ecf8bc4eecaba5caded4c2b4903f96e7b05b34266a65c37b746c76fe3`.
- <https://arxiv.org/pdf/1606.00818v2>, 45 pages:
  `8da70e9e1f767d3e99076e8e14459a0530bfc9ccbabba926f952198f151a7a65`.

The inspected Fig. 1 caption specifies 70 mV/s in v1 and 40 mV/s in v2.
`sources/audit_arxiv_versions.py` extracts each version using its own axis
ticks and vector geometry; `arxiv-version-audit-20260906/` contains the CSVs,
overlaid figure and audit. Approximate vector-derived values are:

| Source | HI | Forward Jsc (A/m2) | Reverse Voc (V) |
| --- | ---: | ---: | ---: |
| arXiv v1 | 1.973 | 139.83 | 0.8050 |
| arXiv v2 | 1.847 | 134.79 | 0.7446 |
| Published reference | 1.839 | 134.95 | 0.7455 |

These are graph-derived values, not author raw data. V2 and the published
curves closely agree; v1 differs, most clearly in reverse Voc. The evidence
does not support simply treating the published graph as an unchanged v1 plot.
V1's carrier-mobility table also contains dimensionally inconsistent entries;
v2 corrects both to 20 cm2/V/s, consistent with the published table and inputs.

A declared historical-rate-only probe, `alignment/arxiv-v1-rate-only-n60/`,
changes 40 to 70 mV/s while preserving the 0.5 s light-on dwells and all other
inputs. It completed in 12.48 s with HI=2.03639711, forward Jsc=123.19999 A/m2
and reverse Voc=0.774019 V. Neither the older nor the final graph is reproduced
pointwise. This is a rejected replacement for the required 40 mV/s condition,
not a tuned preset or evidence that the published scan rate is wrong.

### Source-Declared Potential Bias

V2 Fig. 1h explicitly labels the high-bias profiles as 1.08 V, whereas the
published legend rounds them to 1.1 V. Their right-hand plateaus independently
give 1.07998 and 1.07997 V. The extractor now retains the published label in
`voltage_V` and separately records/uses `comparison_voltage_V=1.08`.
`reference-bias-aligned-20260906/` compares those source-declared biases to
existing 241-node states; no trajectory or potential offset was fitted.

This removes the approximately 20 mV right-plateau mismatch (residuals below
25 microvolts), but absorber RMSE remains 21.1 mV forward and 30.2 mV reverse.
The reverse absorber error actually increases relative to comparison at the
rounded bias. Interior-field disagreement remains a physical diagnostic,
not a plotting-label explanation for the high-hysteresis current gap.

V2 also exposed zero-length PDF path segments. `stroke_endcaps` now ignores
exactly repeated vertices before computing tangents; no geometry or tolerance
was changed. Nine focused geometry tests pass, including duplicate vertices
and rejection of degenerate outlines. Regenerating the published reference
produces a byte-identical J-V `reference.json`; inspected updated figures have
no overlap. The J-V reference HI remains 1.83913697991.

All processes and MATLAB calls in this source audit returned. No commit or
push was made. Live Docker deployment remains unverified after the prior
permission denial; no alternative access to that denied resource was used.
The original Fig. 1 simulation script/state is still unconfirmed. Matching
the declared common-input solvers and stable parameter trends does not prove
the requested quantitative paper reproduction; the full goal remains open.

## Turnaround Reference And Remaining Input Blocker, 2026-09-06

### Bounded Turnaround Hypotheses

The preprint's 1.08 V profile label does not by itself prove a 1.08 V scan
endpoint. The early author source's `pinParams.m` sets Vstart=1.1 for its
default reverse scan, but that code is not a confirmed 2016 input deck.
These are explicitly declared diagnostic hypotheses, not preset changes.

`probe_calado_paper_alignment.py` now accepts `--scan-end-voltage`, defaults
to the unchanged 1.2 V endpoint, and rejects invalid/unresolved scan intervals
before loading a model or writing results. At 61 nodes, 40 mV/s, unchanged
material parameters and 0.5 s branch-start dwells:

| Endpoint | Dark turnaround hold | HI | Reverse Voc (V) |
| --- | ---: | ---: | ---: |
| 1.20 V | 3 s | 0.47864474 | 0.779606 |
| 1.08 V | 3 s | 0.45762217 | 0.750630 |
| 1.08 V | none | 0.45293855 | 0.740877 |
| 1.10 V | none | 0.45402966 | 0.744430 |

The 241-node 1.10 V/no-hold case completed in 247.76 s with HI=0.45681022
and reverse Voc=0.744783 V. `turnaround-audit-20260906/audit.json` records
current and potential errors, hypotheses, input hashes and forward-prefix
checks. The 61-node variants change the forward potential by at most
1.58e-13 V and the full forward current by 6.84e-6 A/m2 before turnaround.
Changing a later endpoint does not resolve the earlier forward discrepancy.

### Actual MATLAB Reference And Adapter Fidelity

The common-input MATLAB adapter previously hardcoded both scan endpoints.
It now reads declared start/end values from the source record, checks them
before reference setup, derives each branch duration from range/scan rate,
and records `scan_protocol`. Older records without these fields retain the
historical -1 to +1.2 V defaults. No upstream PDE or solver option changed.

`driftfusion/run-20260906-204915-959/` completed both cases at 1200 uniform
nodes and 421 samples per branch, using the 241-node SolarLab source record.
Metadata and CSV/MAT endpoints confirm -1 to +1.1 V and 52.5 s per branch.
Driftfusion gives HI=0.00027391 for the no-contact-SRH control and 0.46750474
with contact SRH, with reverse Voc=0.745844 V for the hysteretic case.
The existing 13 MATLAB monitor/solution/bias-lift tests also passed. Code
Analyzer reported only the pre-existing `fileattrib` modernization advisory.

`audit_turnaround_reference.py` independently reads the saved CSV/MAT arrays,
checks their shape, finiteness, bias/time relation and potential boundaries,
and produces `turnaround-audit-20260906/reference-comparison.json`.
No new integration or charge rescaling is involved. The native reference's
forward-prefix potential differs from its 1.2 V baseline by only 3.66e-9 V;
normalized density differences remain below 4.98e-9. The complete current
prefix differs by at most 8.52e-5 A/m2, including the changed final derivative
stencil used for displacement-current postprocessing.

The interior potential errors to the paper agree between independent codes:

| Branch / bias | SolarLab absorber RMSE (mV) | Driftfusion absorber RMSE (mV) |
| --- | ---: | ---: |
| forward, 0 V | 11.819 | 11.816 |
| forward, 0.4 V | 16.412 | 16.411 |
| forward, 0.8 V | 19.994 | 19.996 |
| forward, 1.08 V | 21.106 | 21.115 |
| reverse, 1.08 V | 0.554 | 0.556 |
| reverse, 0.8 V | 1.329 | 1.298 |
| reverse, 0.4 V | 15.646 | 15.655 |
| reverse, 0 V | 25.079 | 25.071 |

The turnaround hypothesis improves early reverse profiles but leaves later
reverse evolution and all forward profiles mismatched. Forward J-V RMSE to
the paper is 50.827 A/m2 (SolarLab) and 50.141 A/m2 (Driftfusion), with no
forward samples inside the published stroke envelope. Reverse RMSE is about
2.26 A/m2 for both. A nearby Voc does not establish pointwise reproduction.

### Current Checks And Completion Boundary

The current checkout passed 63 focused Python tests without historical preset
fixtures (`turnaround-contract-regression-20260906.xml`), 38 frontend tests
covering ion inputs/mode activation and waveform controls, and TypeScript
no-emit checking. These are scoped tests, not a whole-repository certificate.

- Physical parameter mapping and current input/dispatch tests are verified.
- Saved real-frontend matrices demonstrate density, diffusivity, scan-rate,
  negative-ion and optical responses with the recorded numerical scope.
- Empty/frozen controls and positive-ion spatial/sampling ladders support
  the reported trends; they do not certify every parameter combination.
- The live Docker-served deployment remains unverified after the recorded
  access denial; no alternative route to that resource was used.
- The required Fig. 1f quantitative result still fails (1.839 reference HI
  versus approximately 0.46-0.49 in the examined shared-input histories).
- The original 2016 Fig. 1 script, runtime parameter record and initial state
  remain unconfirmed despite the publisher/SI, preprints, Zotero attachments
  and early/modern upstream source checks recorded above.

The publisher's Data Availability statement directs data requests to
`dataexss@imperial.ac.uk`. The next needed evidence is the Fig. 1e/f voltage
and time arrays, illumination/prebias sequence, parameter structure and
starting `.mat` state (or the script that constructs it). No external request
has been sent. Further arbitrary physical-parameter fitting would not prove
the user's requested reproduction and is not authorized by this checkpoint.

Input-blocker audit: third consecutive impasse check. The task output tree
still has no newer files, and every source hash in the shared-input reference
comparison verifies unchanged. The previous recheck made no scientific
progress and was not a verified wait; all submitted simulation/tool handles
had already returned. The same original Fig. 1 input/protocol/state dependency
remains unresolved. The goal is now marked blocked, not complete.
Resume with the original author script/state or an authenticated runtime
parameter and timeline record. Existing passing tests, figures and the
presentation do not satisfy the missing quantitative paper-reproduction gate.

### Presentation Handoff

The user-requested 45-slide update is complete in the repository workspace:
`../outputs/presentations/260907-ion-results/output/260907-SolarLab-ion-results-v2.pptx`,
SHA-256 `8b51c49f31222271e0a2434773f4a4c912104db3390c146aa1d52afaccc21dcd`.
New slides 37-41 replace the obsolete original slide 37. They use the verified
frontend matrices and the 1.2 V/3 s shared-input reference, not the exploratory
turnaround hypotheses. The source iCloud deck was left unchanged because its
directory was outside the writable workspace. No commit or push was made.

## User-Requested Dense Ion Sweeps, 2026-09-07

This separate figure-update request does not close the original-paper input
blocker. The two normalized parameters D/Dref and c0/cref were each sampled
at 0, 0.1, ..., 2.0. All 42 real backend jobs completed, starting from the
captured nominal frontend request and changing only the selected absorber
field. The original 61-node, 111-sample, 40 mV/s waveform and rtol=1e-4,
atol=1 m^-3 settings were retained. No production equations or solver code
were changed, and no new browser interaction is claimed.

`outputs/calado-reproduction/dense-ion-matrix-20260907/` contains each full
request, result, log, state audit and reconstructed J-V curve. `matrix.json`
checks all 42 points, the repeated nominal point, and the four original
anchors. The maximum ionic inventory drift is 5.13e-15 relative.

The denser parameter grid exposes maxima missed by the four-point figures:

| Parameter ratio | HI | Forward Pmax (W/m2) | Reverse Pmax (W/m2) |
| --- | ---: | ---: | ---: |
| D/Dref = 0.2 | 96.203833 | 0.805157 | 78.264393 |
| c0/cref = 0.2 | 9.995071 | 7.217705 | 79.359174 |

These are sampled maxima, not fitted continuous peak locations. Four extra
peak checks retained the same physical history: 441 samples at 61 and 121
nodes gave diffusivity-peak HI=95.825114 and 99.903531, and density-peak
HI=9.999269 and 9.974491. The large diffusivity HI reflects weak but positive
forward maximum power. These checks are not a full convergence certificate.

The updated PPT changes only slides 39/40 figures and their directly affected
scientific copy. Each contains 21 loops (42 branches) and 21 HI bars with
Arial text and editable data. Other 43 slide XML parts and the shared design
remain unchanged. Repository-root output:
`results/DenseIonResults/Output/SolarLabDenseIonResultsV4.pptx` (archive folder, see `CLAUDE.md` → Archived documents).
SHA-256: `bb9a11ba4cbcb690c189059f5a5d4c0b04e1baa280425c70884addb96d441c9c`.
The previous PPT is preserved. No commit or remote push was made.
