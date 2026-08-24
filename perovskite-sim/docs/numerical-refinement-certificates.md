# Phase 1 numerical refinement certificates

本文档定义 `reproducibility/numerical_refinement_registry.yaml` 的执行和证据契约。当前 registry 有 29 条 `grid x tolerance` lane，覆盖 Phase 1 minimum/resolved、ion-aware DC/impedance、2D uniform/combined limits、Phase 3 interface charge-off/charged research closure、Phase 4.1 DAE slices、c-Si statistics/ionization/BGN/bulk traps、P4.4 CIGS graded optics、外部 Rs/Rsh DC mapping 与 lumped electrothermal terminal-MPP coupling。阈值、配置内容哈希、adapter 和矩阵都在求解前固定；修改阈值必须使用新的 lane ID，不能根据已有结果原地放宽。

## 状态契约

| status | 含义 |
|---|---|
| `certified` | 矩阵完整；所有 cell 执行成功；最后两级 grid、最后两级 tolerance 和每个 cell 的 quality gate 全部通过 |
| `partial` | 矩阵完整且执行成功，但至少一个预注册收敛门槛未通过 |
| `failed` | cell 缺失/执行失败，observable 缺失或 shape/units 不一致，protocol 缺失/被篡改/跨矩阵不一致，或 manifest 不完整 |

证书的 `unconverged_dimensions` 保存具体失败维度，例如 `grid:voc_V`、`tolerance:jv_normalized`、`quality:max_backward_error` 或 `protocol:inconsistent_across_matrix`。缺失矩阵不会降级成 `partial`，而是 fail-closed 为 `failed`。

收敛比较只使用两个预注册 terminal pair：

- grid：最小 tolerance factor 下的最后两级 grid；
- tolerance：最大 grid 下的最后两级 tolerance factor。

每个 completed cell 都必须精确提供 registry 声明的 observable 和
quality 名称及单位；observable shape 必须在全矩阵一致。所有 quality
gate 则在全部九个 cell 上检查。不能只挑选一个已收敛标量或一个方便的网格报告。

## 预注册矩阵

| lane | grid | tolerance | 核心 observable |
|---|---|---|---|
| `scaps-mirror-frozen-ion-ss` | N30/60/90 | residual factor 1/0.1/0.01 | Voc、Jsc、FF、PCE、归一化 J-V |
| `ionmonger-mobile-ion-transient` | N30/60/90 | componentwise atol factor 1/0.1/0.01 | reverse Voc、hysteresis、terminal-current trace、正离子库存 |
| `ionmonger-ion-aware-dc-v1` | N30/60/90 | componentwise atol factor 1/0.1/0.01 | DC 电流、最大 site occupancy、正离子 centroid |
| `ionmonger-ion-aware-dc-resolved-v2` | N60/90/120 | componentwise atol factor 1/0.1/0.01 | resolved DC 电流、最大 site occupancy、正离子 centroid |
| `ionmonger-ion-aware-impedance-resolved-v1` | N60/90/120 | FD step factor 1/0.5/0.25 | impedance magnitude/phase |
| `csi-qf-frequency-domain` | N100/200/300 | FD step 1/0.5/0.25 | C-V、Mott-Schottky intercept/effective doping |
| `csi-qf-frequency-domain-resolved-v2` | N200/300/400 | FD step 1/0.5/0.25 | resolved C-V、Mott-Schottky intercept/effective doping |
| `twod-uniform-limit` | x/y multiplier 1/2/4 | componentwise atol factor 1/0.1/0.01 | 2D-to-1D J-V envelope、Voc、Jsc |
| `twod-mobile-ion-interface-srh-v1` | matched x/y intervals 4/6/8 | componentwise atol factor 1/0.1/0.01 | complete current、interface current、lateral response、site fraction、ion redistribution |
| `external-series-shunt-dc-v1` | N20/30/40 | componentwise atol factor 1/0.1/0.01 | full terminal trace、Voc/Jsc/FF/PCE、Kirchhoff balance |
| `external-series-shunt-dc-operating-quadrant-v2` | N20/30/40 | componentwise atol factor 1/0.1/0.01 | terminal operating quadrant、Voc/Jsc/FF/PCE、zero coupling |
| `electrothermal-terminal-mpp-v1` | N10/15/20 | componentwise atol factor 1/0.1/0.01 | 300--380 K MPP response、operating T、terminal MPP |
| `electrothermal-terminal-mpp-resolved-v2` | N20/30/40 | componentwise atol factor 1/0.1/0.01 | v1 unchanged contract on a higher grid ladder |
| `electrothermal-terminal-mpp-grid-resolved-v3` | N40/60/80 | componentwise atol factor 1/0.1/0.01 | grid-resolved MPP response、operating T、first-law closure |
| `interface-recombination-charge-off` | N30/60/90 | QF residual factor 1/0.5/0.25 | two-sided interface flux、归一化 J-V、Voc |
| `interface-charge-equilibrium-referenced-v1` | N30/60/120 | QF residual factor 1/0.5/0.25 | charged current、occupancy、sheet charge、trace shift |
| `interface-charge-device-stress-v1` | N30/60 | QF residual factor 1/0.5 | 9-point Et/CBO/Nd/Nt current、dark occupancy、sheet charge、trace shift |
| `interface-charge-device-stress-resolved-v2` | N30/60/90 | QF residual factor 1/0.5 | v1 同一 9-point observable 与门限，N90 为 terminal grid |
| `no-ion-dae-transient-v1` | 单层区间 8/16/32 | BE time-step factor 1/0.5/0.25 | MoL 终态误差、carrier inventory response、dense/structured 等价性 |
| `single-positive-ion-dae-transient-v1` | 单层区间 8/16/32 | BE time-step factor 1/0.5/0.25 | MoL 终态 n/p/P/phi 误差、正离子守恒、dense/structured 等价性 |
| `dual-mobile-ion-dae-transient-v1` | 单层区间 8/16/32 | BE time-step factor 1/0.5/0.25 | MoL 终态 n/p/P+/P-/phi 误差、逐离子守恒、shared-site bounds、dense/structured 等价性 |
| `algebraic-interface-state-dae-transient-v1` | 每层区间 4/8/16 | BE time-step factor 1/0.5/0.25 | MoL 终态 n/p/interface/phi 误差、interface occupation、clamp-inactive、dense/structured 等价性 |
| `single-ion-algebraic-interface-dae-transient-v1` | 每层区间 4/8/16 | BE time-step factor 1/0.5/0.25 | combined MoL n/p/P/interface/phi 误差、守恒、clamp-inactive、dense/structured 等价性 |
| `single-ion-algebraic-interface-dae-transient-resolved-v2` | 每层区间 8/12/16 | BE time-step factor 1/0.5/0.25 | resolved combined DAE 与同一 topology/quality contract |
| `degenerate-pn-equilibrium-v1` | 每层区间 40/80/160 | Poisson residual factor 1/0.1/0.01 | FD generalized-SG equilibrium、耗尽宽度、峰值场、空间电荷平衡 |
| `incomplete-ionization-temperature-equilibrium-v1` | 每层区间 40/80/160 | Poisson residual factor 1/0.1/0.01 | donor/acceptor freeze-out、温度曲线、耗尽宽度、峰值场 |
| `incomplete-ionization-bgn-temperature-equilibrium-v1` | 每层区间 40/80/160 | Poisson residual factor 1/0.1/0.01 | freeze-out + Slotboom BGN、effective gap、空间电荷平衡 |
| `bulk-energy-distributed-trap-equilibrium-v1` | 每层区间 40/80/160 | Poisson residual factor 1/0.1/0.01；每格 energy order 16/32/64 | trap occupancy/recombination/absolute charge、analytic charge tangent、Gauss balance |
| `cigs-graded-optics-v1` | optical slices 8/16/32 | inverse KK factor 1/0.5/0.25 -> order 96/192/384 | absorbed flux、generation profile/centroid、reflectance、Carron/uniform/photon-budget gates |

归一化 J-V/trace 使用 absolute `L_inf <= 0.5%`，Voc 使用 absolute `<= 1 mV`，Jsc 使用 relative `<= 0.2%`。离子库存漂移 gate 为 `<= 1e-10`。c-Si 的 all-face spread 和 backward error gate 分别为 `<= 5e-4` 和 `<= 1e-10`。其余 lane-specific quality gate 见 registry；它们仍是 internal candidate gate，不能解释成外部物理误差条带。

2D 的 `grid` 是同步乘数：每层竖直区间数为 `5 x multiplier`，横向区间数为 `4 x multiplier`，匹配的 1D 区间总数为三层竖直区间总数。它不是只细化 y 而固定 x。1D 和 2D 两侧都比较电子加空穴的传导电流，不混入仅 1D 路径具有的离子/位移电流；除 refinement gate 外，每个 cell 还要求绝对 2D-to-1D 归一化差不超过 0.5%，横向均匀性同时检查 `n` 和 `p`。

## Protocol provenance

每个 production adapter 在 cell metadata 中写入：

- `protocol`：canonical protocol document；
- `protocol_schema`：experiment、numerical execution 或 bundle schema；
- `protocol_hash`：canonical document 的 SHA-256。

mobile-ion J-V 使用真实 `ExperimentProtocol`；c-Si 将每个 DC bias 的 QF impedance protocol 明确组成 bundle；frozen steady ladder 使用不虚构 scan rate 的 numerical protocol；charge-off interface lane 使用专用 two-sided QF protocol，绑定 contact certificate、暗态 occupancy reference 与 illumination ladder；charged interface lane 进一步绑定 `-q*N_t*(f-f_eq)`、暗态 bit identity、dark-bias/light targets 与 local IFT gates；device-stress companion 将 baseline 加 `E_t/CBO/N_D/N_t` 各两个 one-factor 端点、逐设备暗态参照和 charge/barrier 符号门限写入同一 canonical protocol；2D bundle 分列实际执行的 1D forward/reverse protocol、被比较的 forward branch 和 2D ascending finite-time protocol。2D 的 1D `v_rate` 显式设为 `V_step / settle_time`，使两条路径的每点 dwell 相同。

external-circuit lane 将 intrinsic J-V protocol、area-normalized Rs/Rsh
topology、source result 与 terminal mapping hash 分开绑定；electrothermal lane
进一步冻结 fresh-state temperature trials、sampled terminal-MPP rule、thermal
control volume、root envelope 与完整 first-law inputs。低网格 v1/v2 的 partial
证据保留，不能由 v3 certified artifact 覆盖。

ion-aware DC 使用专用 frozen physical protocol，记录固定偏压、有效温度、
明暗历史、初态来源/可选初态 SHA-256、blocking ion 边界、ordered endpoint
ladder、连续通过次数和全部物理门槛。其外层 numerical execution protocol
另行记录 grid/tolerance 来源、componentwise `atol`、`rtol`、`max_nfev`
及 `Radau -> BDF` 方法阶梯。

no-ion DAE protocol 固定 c-Si 配置中的单层切片、ohmic/no-ion/no-interface
拓扑、`(log n, log p, phi)` 坐标、dense/structured backward-Euler 控制量和
高精度 Radau 参考。为保持扩散问题在空间加密后的时间误差可比，步数按
`base_steps * (N/N_ref)^2 / tolerance_factor` 预注册；实际 cell 的 N 和步数
写在 metadata，不进入跨矩阵必须一致的 protocol document。

single-positive-ion DAE protocol 独立固定 IonMonger MAPbI3 absorber 切片、
ohmic/blocking/no-interface 拓扑、`(log n, log p, logit(P/P_lim), phi)` 坐标、
finite-site steric law、10 mV/10 ms 暗态历史以及同样的 N 平方时间步阶梯。
每个 cell 同时执行 strict Radau/MoL、dense-central BE 和
structured-analytic BE，并分列 carrier/ion/algebraic residual、离子库存与
site-occupancy 证据；它不继承 no-ion DAE 的 protocol 或证书。

dual-mobile-ion DAE protocol 在独立 capability contract 中加入一正一负两个
blocking unit-charge species、positive/negative/vacancy 三态 shared-site softmax、
逐 species inventory 与 physical-density storage。负离子的 diffusion、density 和
site-limit 是 protocol 明确冻结的合成输入，不解释为 source IonMonger publication
参数。每个 cell 同样运行 strict Radau/MoL、dense-central BE 和
structured-analytic BE；该证书不继承 single-ion 结果。

algebraic-interface-state DAE protocol 独立固定两层、单界面、ohmic、无离子、
charge-off topology，四个 DOS-bounded Fermi-Richardson algebraic states，以及
bulk projection、cross-plane exchange 和 shared-occupancy interface-SRH 的
clamp-inactive analytic tangent。这里的相邻 bulk/interface 交叉导数不表示启用
可配置 cross-node carrier sampling；后者与 `InterfaceDefect`、dynamic states、
interface charge、selective contacts 等仍显式排除。每格同时执行 locally
eliminated-QSS Radau/MoL、dense-central BE 和 structured-analytic BE。

single-ion plus algebraic-interface DAE protocol 将 synthetic blocking positive
ion 与上述四个 algebraic trace states 组合，并逐 cell 同时检查 ion inventory、
site bounds、interface clamp、状态/Poisson residual 和 dense/structured tangent
等价性。v1 与 resolved-v2 使用不同 grid ladder，证书不能互相替代。

combined 2D protocol 固定 Neumann-x、ohmic、single-positive blocking ion、一个
finite-width grain boundary 与一个 clamp-inactive cross-node `InterfaceDefect`
sheet，并为每个 cell 构造完全匹配的 explicit `jv-2d-execution-protocol-v1`。
外层 numerical protocol 固定 4/6/8 matched x/y intervals、0.0/0.05/0.10 V、
每点 10 ns dwell 及 tolerance ladder；完整边界见
`docs/twod-combined-numerical-certificate.md`。

degenerate-PN protocol 固定 symmetric high-doping c-Si p+/n+、dark equilibrium、
fully-ionized FD charge、semiconductor-work-function ohmic contacts 和
diffusion-enhanced generalized-SG flux。bulk recombination 明确关闭；解析 abrupt
depletion width/peak field 只是内部 quality oracle，不是 Sentaurus/PC1D 外部验证。

bulk-trap protocol 固定 homogeneous MB p/n、fully-ionized dopants、一个
truncated-Gaussian acceptor-like energy distribution、trap-aware semiconductor
work-function contacts，以及每格 16/32/64 的共享 occupancy/recombination/charge
quadrature。默认 MoL 必须拒绝该配置；energy doubling、mass action、face current、
Poisson 和离散 Gauss balance 分列 gate。旧 SCAPS Gaussian metadata 不进入该闭合。

CIGS graded-optics protocol 固定 Minoura composition-resolved dielectric
model、与 electrical Eg/chi 共用的 GGI coordinate、8/16/32 optical slices、
96/192/384 KK quadrature、固定 electrical observation mesh 与 AM1.5G wavelength
grid。Carron alpha 在 front/mid/back 三个 GGI 上作为独立 benchmark；每格另检查
default-off、uniform-composition、causal n/k、reflectance 和 photon budget。该 lane
不执行 transport，因此不产生 J-V/PCE 或外部器件认证。

manifest 保存去重后的完整 protocol document 和 hash。certificate 只有在所有完成 cell 的 protocol 内容自校验通过且 hash 跨 grid/tolerance 一致时才记录 `protocol_sha256`。protocol provenance 不替代 config、source、environment、grid 或 tolerance provenance。

## Content-addressed outputs

默认输出为 `outputs/numerical-refinement/<lane>/<run_id>/`：

```text
cells/<file_sha256>.json
manifests/<file_sha256>.json
certificates/<certificate_content_id>.json
state.json
```

cell、manifest 和 certificate 均为 immutable artifact；`state.json` 只是原子更新的 resume pointer。runner 在恢复时重新验证 manifest、artifact hash、run ID、lane definition 和 matrix completion state。`--retry-failed` 产生新 cell artifact，旧失败 artifact 保留。

runner 拒绝把输出写到 `reproducibility/baselines` 或 `perovskite_sim/data/references` 下，因此默认不会覆盖历史 reference。把新证书晋级为长期 reference 是独立的人工审查步骤。

source fingerprint 覆盖当前 scope 内的 staged、unstaged、deleted 和
untracked 内容；environment identity 记录会改变求解方程或路径的
`SOLARLAB_*`/`PEROVSKITE_RHS_FINITE_CHECK` effective value。certificate、
registry、manifest、state 和 artifact reference 均拒绝未知字段，resume
还会逐项核对完整 canonical identity。这样 staged-only 修改或环境开关
不能复用旧 cell。

## 执行命令

列出所有 lane：

```bash
python scripts/run_numerical_refinement.py --list-lanes
```

逐 lane 做只读计划验证；dry-run 不创建输出目录：

```bash
python scripts/run_numerical_refinement.py scaps-mirror-frozen-ion-ss --dry-run
python scripts/run_numerical_refinement.py ionmonger-mobile-ion-transient --dry-run
python scripts/run_numerical_refinement.py ionmonger-ion-aware-dc-v1 --dry-run
python scripts/run_numerical_refinement.py ionmonger-ion-aware-dc-resolved-v2 --dry-run
python scripts/run_numerical_refinement.py ionmonger-ion-aware-impedance-resolved-v1 --dry-run
python scripts/run_numerical_refinement.py csi-qf-frequency-domain --dry-run
python scripts/run_numerical_refinement.py csi-qf-frequency-domain-resolved-v2 --dry-run
python scripts/run_numerical_refinement.py twod-uniform-limit --dry-run
python scripts/run_numerical_refinement.py twod-mobile-ion-interface-srh-v1 --dry-run
python scripts/run_numerical_refinement.py external-series-shunt-dc-v1 --dry-run
python scripts/run_numerical_refinement.py external-series-shunt-dc-operating-quadrant-v2 --dry-run
python scripts/run_numerical_refinement.py electrothermal-terminal-mpp-v1 --dry-run
python scripts/run_numerical_refinement.py electrothermal-terminal-mpp-resolved-v2 --dry-run
python scripts/run_numerical_refinement.py electrothermal-terminal-mpp-grid-resolved-v3 --dry-run
python scripts/run_numerical_refinement.py interface-recombination-charge-off --dry-run
python scripts/run_numerical_refinement.py interface-charge-equilibrium-referenced-v1 --dry-run
python scripts/run_numerical_refinement.py interface-charge-device-stress-v1 --dry-run
python scripts/run_numerical_refinement.py interface-charge-device-stress-resolved-v2 --dry-run
python scripts/run_numerical_refinement.py no-ion-dae-transient-v1 --dry-run
python scripts/run_numerical_refinement.py single-positive-ion-dae-transient-v1 --dry-run
python scripts/run_numerical_refinement.py dual-mobile-ion-dae-transient-v1 --dry-run
python scripts/run_numerical_refinement.py algebraic-interface-state-dae-transient-v1 --dry-run
python scripts/run_numerical_refinement.py single-ion-algebraic-interface-dae-transient-v1 --dry-run
python scripts/run_numerical_refinement.py single-ion-algebraic-interface-dae-transient-resolved-v2 --dry-run
python scripts/run_numerical_refinement.py degenerate-pn-equilibrium-v1 --dry-run
python scripts/run_numerical_refinement.py incomplete-ionization-temperature-equilibrium-v1 --dry-run
python scripts/run_numerical_refinement.py incomplete-ionization-bgn-temperature-equilibrium-v1 --dry-run
python scripts/run_numerical_refinement.py bulk-energy-distributed-trap-equilibrium-v1 --dry-run
python scripts/run_numerical_refinement.py cigs-graded-optics-v1 --dry-run
```

建议固定 BLAS 线程后逐 lane 执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py scaps-mirror-frozen-ion-ss
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py ionmonger-mobile-ion-transient
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py ionmonger-ion-aware-dc-v1 --allow-noncertified-exit-zero
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py ionmonger-ion-aware-dc-resolved-v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py ionmonger-ion-aware-impedance-resolved-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py csi-qf-frequency-domain
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py csi-qf-frequency-domain-resolved-v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py twod-uniform-limit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py twod-mobile-ion-interface-srh-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py external-series-shunt-dc-v1 --allow-noncertified-exit-zero
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py external-series-shunt-dc-operating-quadrant-v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py electrothermal-terminal-mpp-v1 --allow-noncertified-exit-zero
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py electrothermal-terminal-mpp-resolved-v2 --allow-noncertified-exit-zero
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py electrothermal-terminal-mpp-grid-resolved-v3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py interface-recombination-charge-off
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py interface-charge-equilibrium-referenced-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py interface-charge-device-stress-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py interface-charge-device-stress-resolved-v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py no-ion-dae-transient-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py single-positive-ion-dae-transient-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py dual-mobile-ion-dae-transient-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py algebraic-interface-state-dae-transient-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py single-ion-algebraic-interface-dae-transient-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py single-ion-algebraic-interface-dae-transient-resolved-v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py degenerate-pn-equilibrium-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py incomplete-ionization-temperature-equilibrium-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py incomplete-ionization-bgn-temperature-equilibrium-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py bulk-energy-distributed-trap-equilibrium-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py cigs-graded-optics-v1
```

先执行一个 cell 并留下可恢复状态：

```bash
python scripts/run_numerical_refinement.py twod-uniform-limit --max-cells 1 --allow-noncertified-exit-zero
python scripts/run_numerical_refinement.py twod-uniform-limit
python scripts/run_numerical_refinement.py twod-uniform-limit --retry-failed
```

source、executor 或 environment 变化会产生新的 `run_id`，不会误接到旧 run。相同 run 重复执行只复用已验证的 cell。

## 真实矩阵成本和证据边界

当前 registry 共 29 条 lane；二十七条 lane 各 9 个 cell，device-stress v1/v2 分别为 4/6 个 cell，合计 253 个 content-addressed cell。其内部工作量至少包括：

- frozen SCAPS：279 个 residual-certified steady voltage points；
- mobile IonMonger：9 次 forward/reverse transient J-V，每次 40 个采样点，且可能触发 recovery/bisection；
- ion-aware DC：18 次 fixed-bias endpoint ladder；每段保留独立 DC state
  certificate，Radau 失败时再走 BDF，原始及 resolved 网格证据分别保留；
- ion-aware impedance：9 次 certified DC anchor 加完整 frequency/stencil
  response，并逐频率检查 backward error、库存响应和 current decomposition；
- c-Si minimum-roadmap lane：54 个 bias/FD-step operating-point calls，共 162 个 frequency points；resolved-v2 companion 另有同等工作量；
- frozen 2D uniform lane：117 个 2D finite-time voltage settles，外加 9 个 matched 1D forward/reverse sweeps 和 2D seed；
- combined 2D lane：9 个 strict public J-V runs，各有 3 个 fixed-voltage dwell，并逐点重建 complete current、ion/interface diagnostics 与 GB geometry evidence；
- external circuit：v1/v2 各 9 个 strict intrinsic J-V cells，分别检查完整
  junction trace 与器件发电象限上的 terminal mapping、Kirchhoff balance、
  zero-coupling 和 source/protocol provenance；
- electrothermal：v1/v2/v3 各 9 个 cells；每个 cell 在 ambient、maximum 和
  Brent root trial temperatures 上重新运行 fresh-state strict J-V，应用 frozen
  Rs/Rsh，选取 sampled terminal MPP，并重建 first-law ledger；
- interface charge-off：9 个 certified dark occupancy references、225 个
  requested two-sided QF illuminated voltage points及其 illumination ladders，
  并逐点计算 capture flux、local QSS、continuity 与 current-spread evidence。
- equilibrium-referenced interface charge：9 个重新认证的 charge-off dark
  anchors、9 个 dark-bias state 和 9 个 illuminated state，并逐点保存
  `f_eq/f/Delta sigma`、trace shift、Gauss、IFT condition 与状态哈希。
- interface-charge device stress：v1/v2 每个 refinement cell 内执行 9 个
  one-factor device variants；两条 lane 合计重建 90 个 contact-certified dark
  anchors，并求 90 个 dark-bias 与 90 个 illuminated target states。
- no-ion DAE：9 个 cell 各执行同网格高精度 Radau/MoL 参考、dense-central
  backward Euler 和 structured-analytic backward Euler；N=8/16/32 的时间步
  按 N 平方缩放，最细 cell 为 512 步。
- single-positive-ion DAE：9 个 cell 各执行同网格 strict Radau/MoL、
  dense-central BE 与 structured-analytic BE；分列 n/p/P differential residual、
  Poisson/边界 algebraic residual、blocking-ion inventory 和 finite-site bounds，
  最细 cell 为 128 步。
- dual-mobile-ion DAE：9 个 cell 各执行同网格 strict Radau/MoL、
  dense-central BE 与 structured-analytic BE；分列 n/p/P+/P- differential
  residual、逐 species blocking inventory、shared-site vacancy 与 Poisson/边界
  algebraic residual，最细 cell 为 128 步。
- algebraic-interface-state DAE：9 个 cell 各执行两层同网格的 strict
  locally-eliminated-QSS Radau/MoL、dense-central BE 与 structured-analytic BE；
  分列 carrier/interface/algebraic residual、三类 continuity balance、interface
  occupation/bounds、clamp margin 和 structured work，最细 cell 为 128 步。
- degenerate PN：9 个 cell 各重建 statistics-aware contacts，并在同一冻结
  high-doping homojunction 上求 fully-ionized FD-Poisson equilibrium、generalized-SG
  零流和 abrupt-depletion analytic oracle。
- CIGS graded optics：9 个 cell 各执行 photon-conserving AM1.5G TMM、
  uniform-composition companion 和 front/mid/back Carron comparison；matrix
  分别加密 absorber slices 与 Kramers-Kronig quadrature，不执行 transport。

因此 CI 只运行 schema、resume 和注入式 adapter smoke；完整物理矩阵应作为受控的长时任务逐 lane 运行。当前 infrastructure、dry-run 或 adapter smoke 不能替代这 253 个 cell 的真实结果。

`certified` 只表示该冻结代码/config/protocol/environment 下的内部数值收敛。它不等于 SCAPS/IonMonger 外部 solver parity，不等于实验验证，也不证明材料参数或模型闭包唯一正确。

### 2026-08-24 no-ion DAE first-slice certificate

`no-ion-dae-transient-v1` 在 source commit `985a234`、source changes 为空且
BLAS/OpenMP 单线程的环境下完成 9/9 cell，0 failed、0 missing、0 reused：

- run ID：`b3030c044d4753f17aed89216983ff2ac7013f54a5865fc6fa1647200c05931c`；
- certificate SHA-256：`44807d654d815bc0daa4aeb7d3ecc48ecbcb17436209f4d56c62f64d14cf3c0b`；
- protocol SHA-256：`65a6a01d80078146bf01a7e94b071eaa2cb24ba94ee812a9304cd97aed660d67`；
- terminal grid differences：carrier inventory response `3.38764e-2`、MoL
  terminal log-density error `3.18170e-6`、potential error
  `5.38198e-15 V`、dense/structured terminal log-density difference
  `8.88178e-16`、potential difference `1.64957e-13 V`；
- terminal time-step differences：carrier inventory response `6.26970e-6`、
  MoL terminal log-density error `9.16464e-5`、potential error
  `7.39191e-14 V`、dense/structured terminal log-density difference
  `1.11022e-16`、potential difference `4.91740e-14 V`；
- 全矩阵最大 MoL terminal log-density error `3.79255e-4`、normalized
  differential/algebraic residual `9.32738e-10 / 1.97891e-16`、electron/hole
  balance defect `1.03260e-8 / 4.06689e-13 A/m2`、dense/structured trajectory
  log-density difference `1.99840e-15`；structured RHS-work fraction 最大
  `2.54512e-2`，CSR nonzeros/node 最大 `17.9091`。

该证书关闭的是单层、ohmic、无离子、无 interface/`InterfaceDefect` 的
research DAE first-slice entry gate。它不认证移动离子、代数界面态、选择性
接触、cross-node interface sampling、生产 transient/experiment/backend route，
也不是外部 c-Si device validation。

### 2026-08-24 single-positive-ion DAE certificate

`single-positive-ion-dae-transient-v1` 在 source commit `6e9a274`、source
changes 为空且 BLAS/OpenMP 单线程的环境下完成 9/9 cell，0 failed、0 missing：

- run ID：`d71d74acc5574e920edf8e0edb05020c043f3e8a87b1c95c926aa14556658dab`；
- certificate SHA-256：`7538fa4acea081c1c51c5f75201dfec90ebc64bca9e4d62068cb440f97b627e8`；
- protocol SHA-256：`3f0b24b96136e42972b689258aea16c547ac4e18e948ec7aa269ca23af1f989c`；
- terminal grid/time-step differences：log-density error
  `1.95655e-9 / 2.30604e-9`、positive-ion relative error
  `5.53590e-11 / 9.84279e-11`、potential error
  `5.00667e-11 / 5.94353e-11 V`，全部通过预注册门限；
- 全矩阵最大 carrier/positive-ion/algebraic normalized residual
  `4.31830e-10 / 1.33230e-16 / 1.34261e-16`，positive-ion inventory drift
  `3.14321e-16`，electron/hole balance defect
  `3.35000e-18 / 7.79983e-18 A/m2`；
- 正离子最小相对运动 `2.94123e-6`，证明测试不是静态零变化；全矩阵
  dense/structured trajectory 的 log-density、positive-ion、potential 差异至多
  `5.55112e-16 / 0 / 4.33681e-19 V`；structured RHS-work fraction 最大
  `2.55639e-2`，CSR nonzeros/node 最大 `24.7273`。

该证书关闭的是单层、ohmic、blocking single-positive-ion、无 interface 的
research DAE topology gate。它不是外部 IonMonger solver parity，不认证双离子、
algebraic interface state、选择性接触、生产 transient/experiment/backend route。

### 2026-08-24 dual-mobile-ion DAE certificate

`dual-mobile-ion-dae-transient-v1` 在 source commit `2d6b32f`、source changes
为空且 BLAS/OpenMP 单线程的环境下完成 9/9 cell，0 failed、0 missing；相同
环境重复执行为 0 executed、9 reused，run/certificate hash 不变：

- run ID：`a5e50a9f6522bf1229e1f2b416caf9b5ba914e574ceca2a08a363e1627581cc2`；
- certificate SHA-256：`15a6a4dcf38db26e2fa78f41ede0d908ad154c267ce86de13eeaf7e6c1f050ab`；
- protocol SHA-256：`8693f4c2009eb4487978671e5452db8c5ba596e3b365d12c13663a9257af348d`；
- terminal grid/time-step differences：log-density error
  `2.10299e-9 / 2.48429e-9`、positive-ion relative error
  `5.49541e-11 / 9.77153e-11`、negative-ion relative error
  `4.75563e-12 / 8.46220e-12`、potential error
  `5.38119e-11 / 6.40295e-11 V`，全部通过预注册门限；
- 全矩阵最大 carrier/positive-ion/negative-ion/algebraic normalized residual
  `3.44856e-10 / 4.67293e-16 / 1.67140e-16 / 8.19753e-16`，positive/negative
  inventory drift `2.95472e-16 / 3.08323e-16`，electron/hole balance defect
  `3.14035e-18 / 7.81589e-18 A/m2`；
- 正/负离子最小相对运动 `2.94124e-6 / 9.31920e-7`，shared-site vacancy
  fraction 最小 `0.9799999`；全矩阵 dense/structured trajectory 的 log-density、
  positive-ion、negative-ion、potential 差异至多
  `5.55112e-16 / 0 / 0 / 8.67362e-19 V`；structured RHS-work fraction 最大
  `2.05562e-2`，CSR nonzeros/node 最大 `37.4242`。

该证书关闭的是单层、ohmic、blocking shared-site dual-mobile-ion、无 interface
的 research DAE topology gate。负离子参数是合成 protocol 输入；这不是外部
IonMonger parity，也不认证 algebraic interface state、选择性接触、生产
transient/experiment/backend route。

### 2026-08-24 algebraic-interface-state DAE certificate

`algebraic-interface-state-dae-transient-v1` 在 source commit `008aef3`、source
changes 为空且 BLAS/OpenMP 单线程的环境下完成 9/9 cell，0 failed、0 missing；
相同环境重复执行为 0 executed、9 reused，run/certificate hash 不变：

- run ID：`407927842820d9360b132aaad50fa97c5bec55b146bef75f86811edd06845cad`；
- certificate SHA-256：`21bb12e4655c60ce7c97ce2a3cf57617fc3c2cb667990dab244fc04dc4a53c89`；
- protocol SHA-256：`9ffcc7e0cf2adfa52192686563455558f9fc4afa3a1982736d87f1c03af75efd`；
- terminal grid/time-step interface-occupation differences
  `4.92480e-3 / 5.57687e-13`；terminal MoL interface/log-density error changes
  `7.60700e-10 / 5.57733e-13` 与 `7.60825e-10 / 5.59330e-13`；
- 全矩阵最大 carrier/interface/algebraic normalized residual
  `4.97175e-8 / 2.55161e-14 / 2.55161e-14`，electron/hole/interface balance
  defect `8.47196e-15 / 1.50341e-15 / 1.45516e-15 A/m2`；
- 全轨迹 dense/structured log-density、interface-state relative density、potential
  差异至多 `6.61882e-12 / 6.61504e-12 / 2.01228e-16 V`；structured RHS-work
  fraction 最大 `2.18447e-2`，4/8/16 每层区间对应 CSR nonzeros/node
  `17.4444 / 18.1765 / 18.5758`；
- 全部 clamp-inactive、bounded-interface-state、positive-terminal-density 与
  strict MoL numerical-health gate 通过。

该证书只关闭 dark 10 mV/10 ns、两层 ohmic、单个 uncharged interface、四个
DOS-bounded algebraic Fermi-Richardson states 的 research DAE topology gate。
`InterfaceDefect`、可配置 cross-node carrier sampling、dynamic states、interface
charge、two-sided trace、ions、selective contacts、field mobility、photon recycling
及 clamp-active operating point 均不在声明内；它也不是生产 transient、外部
solver、SCAPS 或实验验证。

### 2026-08-23 charge-off reference certificate

`interface-recombination-charge-off` 在 source commit `29c94b4`、单线程
BLAS/OpenMP 环境下完成 9/9 cell，0 failed、0 missing、0 reused：

- run ID：`d0dc822393290d892e7118bcb7fabd4214b5584815f51ff9ff24f663822687e4`；
- certificate SHA-256：`0a4fdebdf18eb0237eaa1a4bef599872745697d148461f6de25d10a6985a950b`；
- protocol SHA-256：`d423c42dcf486d40b2bc84c930a806de7aa838810f84885b10b7a4a203755048`；
- terminal grid differences：interface flux `0.331079 A/m2`、normalized J-V
  `9.73989e-4`、Voc `4.85196e-6 V`；
- terminal tolerance differences：interface flux `9.41419e-7 A/m2`、
  normalized J-V `4.17758e-7`、Voc `4.16675e-9 V`；
- 全矩阵最大 continuity bound `8.07169e-5 A/m2`、current spread
  `7.96574e-5 A/m2`、interface carrier imbalance `1.09200e-12 A/m2`、
  normalized local interface residual `2.20496e-8`、Poisson residual
  `3.69425e-14`。

该证书只关闭 charge-off reference entry gate；它本身不证明 charged outer
Poisson closure。后续 charged research certificate 见下一节。

### 2026-08-23 equilibrium-referenced interface-charge certificate

`interface-charge-equilibrium-referenced-v1` 在 source commit `23783a3`、
source changes 为空且 BLAS/OpenMP 单线程的环境下完成 9/9 cell，0 failed、
0 missing、0 reused：

- run ID：`f94831ce5f26b6d4aafa702313846aaf717a6d91b58b99ade72481e77f1ae5c4`；
- certificate SHA-256：`1691eaee87208f2494207c94a6f8c484299e34c4ac99c952b6c8df7915cf1921`；
- protocol SHA-256：`63b646172ca135f58227000cdcb5f35a07e9a4b70387a5d197a0498592c605b3`；
- terminal grid differences：charged current `7.59426e-4`、occupancy
  `7.62673e-7`、sheet charge `8.43618e-4`、trace shift
  `8.91838e-6 V`；
- terminal tolerance differences：charged current `1.60532e-10`、occupancy
  `8.49820e-14`、sheet charge `3.25563e-10`、trace shift
  `7.00828e-16 V`；
- 全矩阵最大 normalized Gauss residual `1.74252e-16`、local interface
  residual `1.86582e-12`、continuity bound `4.57098e-9 A/m2`、current
  spread `4.55371e-9 A/m2`、normalized cell residual `4.55619e-9`、Poisson
  residual `4.76324e-16`、scaled local Jacobian condition `4.87423e4`；
- dark incremental charge 与 trace shift 全矩阵严格为零，charge-on/off
  暗态数组 bit-identical；最大 `|Delta sigma|/(q*N_t)` 为
  `9.04050e-4`。

该证书将 purpose-built steady-state Python research lane 升级为内部数值
认证。它不启用 production material assembly、backend experiment、
transient、impedance 或 2D，也不是 SCAPS parity、实验验证或绝对 trap
charge/全器件电中性证明。

### 2026-08-23 interface-charge device-stress certificates

`interface-charge-device-stress-v1` 在 source commit `56cd1bb` 完成 4/4
cell，0 failed、0 missing、0 reused，但证书保持 `partial`：N30 到 N60 的
`stress_sheet_charge_C_m2` pointwise relative difference 为 `1.47351e-2`，
超过固定 `1e-2` 门限。run ID 为
`250a3a8d934f327ab0f73f5197113d2a1de9d57be07f58d71bbd89904fd025a6`，
partial certificate 为
`000fd33d2f162de00ac97fe15e174c27d0d3b995c4aabb9b22e5e62e687df657`。

该结果未被删除、重试或通过放宽门限覆盖。

独立的 `interface-charge-device-stress-resolved-v2` 保留同一 9 个设备点、
observable、quality gate 与 protocol，仅把 terminal grid 扩展为 N90。在
source commit `a3c6b30`、source changes 为空且 BLAS/OpenMP 单线程环境下，
它完成 6/6 cell，0 failed、0 missing、0 reused：

- run ID：`30b146b7f95934fd4353890916d8318f8847e3bb8cb7f556f61afa02223a7b55`；
- certificate SHA-256：`f6e214307fe73fbc9d866d5e2537658cdb563134df78a419ecb6f4f873bd0844`；
- protocol SHA-256：`ff0d4f385ef67bfc749045be955004979d925e2436dca22d3495265951d865f3`；
- terminal grid differences：current `1.00579e-3`、equilibrium occupancy
  `9.87823e-7`、target occupancy `9.87955e-7`、sheet charge
  `1.54775e-3`、trace shift `4.26550e-5 V`；
- terminal tolerance differences：current `3.15183e-10`、equilibrium
  occupancy `1.11022e-16`、target occupancy `4.19165e-12`、sheet charge
  `4.02521e-8`、trace shift `4.73996e-14 V`；
- 全矩阵最大 continuity bound `8.49714e-7 A/m2`、current spread
  `2.36611e-7 A/m2`、local interface residual `2.99688e-10`、normalized
  cell residual `2.36611e-7`、normalized Gauss residual `8.58907e-15`、
  Poisson residual `1.16539e-14`、scaled local Jacobian condition
  `1.04812e7`；
- dark incremental charge 与 trace shift 全部严格为零，所有 device point
  均通过 contact、charge-law、occupancy、sign 与 stack-identity gate。

该证书只把冻结的 two-layer one-factor `E_t/CBO/N_D/N_t` 设备包络升级为
内部数值认证。N120/factor=0.5 在 `N_D=2e15 cm^-3` dark-bias target 上
fail-close；`N_D>=5e15 cm^-3`、多参数交互、历史三层 SCAPS-derived illuminated
case、transient、impedance、2D、绝对 trap charge 与全器件电中性仍不在声明内。

### 2026-08-24 degenerate PN equilibrium certificate

`degenerate-pn-equilibrium-v1` 在 source commit `d756c76`、source changes 为空且
BLAS/OpenMP 单线程的环境下完成 9/9 cell，0 failed、0 missing、0 reused：

- run ID：`3c7c98f9d67bbb2ff2864f946183af9db2b008cf0971a76945e6fdaa7a602eb9`；
- certificate SHA-256：`968ad3bb67dc696b841a6bb8544c16eba3f9d5748b2fe737e20b8c7e30f8373f`；
- protocol SHA-256：`96dd6e56aeb90ad458c5ce86ad31c7aa61eea7e4737573fb7cc30852c5ac91e9`；
- terminal grid difference：depletion ratio `1.24453e-3`、peak-field ratio
  `8.10282e-4`、space-charge balance error `4.22650e-3`；三项 terminal
  tolerance difference 均为 0；
- 全矩阵最大 normalized Poisson residual `1.31861e-12`、normalized carrier
  rate `3.90497e-14`、relative face current `4.02601e-14`、space-charge
  balance error `1.75650e-2`；
- depletion-width / peak-field analytic error 最大 `3.37788e-2 / 2.92171e-2`，
  所有 cell 均保持正载流子、statistics-aware contact certificate 和
  recombination-off topology gate。

该证书只覆盖 fully-ionized、dark、homogeneous c-Si p+/n+ equilibrium。
recombination、incomplete ionization、BGN、bias/illumination、heterojunction 和
production experiment routes 均不继承此证书；解析 depletion approximation 也不等于
Sentaurus/PC1D 外部验证。

### 2026-08-24 incomplete-ionization temperature certificate

`incomplete-ionization-temperature-equilibrium-v1` 在 source commit `1090354`、
source changes 为空且 BLAS/OpenMP 单线程的环境下完成 9/9 cell，0 failed、
0 missing、0 reused：

- run ID：`ad690a7e5398a6e3829f7f04d470a59fe20144e5c863a896f681a87fa3ac8008`；
- certificate SHA-256：
  `902ae0f91b77cf7403349d4d54553c2d43c3c774b1f2a62530e2a27c9fbc0254`；
- protocol SHA-256：
  `33b421757a1521f214047ae58b3ad0cfd412570b65f6b7103e3fe3ed1c99d779`；
- terminal grid difference：归一化积分电荷宽度 `2.05168e-4`、归一化峰值场
  `1.13830e-3`、电荷平衡温度曲线 `4.84260e-4`；对应 terminal tolerance
  difference 为 `2.27381e-7 / 4.26088e-8 / 4.57735e-7`；
- 全矩阵最大 normalized Poisson residual `9.12692e-9`、normalized carrier
  rate `1.07050e-13`、relative face current `1.08413e-13`、space-charge
  balance error `1.96418e-3`，最多 9 次 Newton iteration；
- terminal grid 的 acceptor 离化率从 100 K 的 `0.09027` 上升到 300 K 的
  `0.68718`，donor 从 `0.19690` 上升到 `0.90058`；所有 cell 的离化率有界、
  载流子为正、接触热力学证书通过。

该证书只覆盖 dark、homogeneous、recombination-off、noninteracting discrete
donor/acceptor equilibrium。低温耗尽区内离化电荷随势变化，因此不继承
fully-ionized abrupt-depletion analytic oracle。impurity band、Mott transition、
BGN、dopant kinetics、bias/illumination、recombination、heterojunction、生产
J-V/C-V/impedance，以及 Sentaurus/PC1D 或实验验证均不在声明内。

### 2026-08-24 incomplete-ionization plus BGN temperature certificate

`incomplete-ionization-bgn-temperature-equilibrium-v1` 在 source commit
`d34ef7f`、source changes 为空且 BLAS/OpenMP 单线程环境下完成 9/9 cell，
0 failed、0 missing、0 reused：

- run ID：`bc654a0c76f2d13cdbf64256160fc22b8c7079388aa4babd76998e866e3c3557`；
- certificate SHA-256：
  `bc1285a3ef9d42e1e6346aedad33ad3871b5597a7b409f4ea002bf029f024ce6`；
- protocol SHA-256：
  `58c95444ae1d35ad819b53a238cc266c18699d984cb700ee28a89db76b992e3a`；
- terminal grid difference：归一化积分电荷宽度 `2.16407e-4`、归一化峰值场
  `1.17771e-3`、电荷平衡温度曲线 `4.90624e-4`；对应 terminal tolerance
  difference 为 `1.85145e-7 / 5.69642e-8 / 2.74987e-7`；
- BGN、effective-gap 和两条 contact ionization 温度曲线的 terminal grid 与
  tolerance differences 均为 0；`Delta E_g=0.0216460 eV`，effective gap 从
  100 K 的 `1.141408 eV` 降至 300 K 的 `1.102354 eV`；
- 全矩阵最大 normalized Poisson residual `7.30886e-9`、normalized carrier
  rate `1.72625e-13`、relative face current `2.17846e-13`、space-charge
  balance error `1.99296e-3`，最多 9 次 Newton iteration；
- terminal grid 的 acceptor/donor 离化率从 `0.09027/0.19690` 上升到
  `0.68718/0.90058`，所有 cell 的 BGN chemical-doping law、组合拓扑、接触
  热力学、载流子正性和离化率有界 gate 均通过。

该证书只认证 dark、homogeneous、recombination-off 的 frozen Slotboom BGN 与
noninteracting discrete dopant equilibrium 组合。它不覆盖 impurity band、Mott
transition、其他 BGN 参数化、degenerate recombination、bias/illumination、
heterojunction、生产实验路由、Sentaurus/PC1D 外部验证或实验验证。

### 2026-08-24 energy-distributed bulk-trap equilibrium certificate

`bulk-energy-distributed-trap-equilibrium-v1` 在 source commit `d336e45`、
source changes 为空且 BLAS/OpenMP 单线程环境下完成 9/9 cell，0 failed、
0 missing、0 reused：

- run ID：`b69fa02f71039b8f1d8c753a8ee07ef8689bc261e206c58c0ea91da2b5ef07f4`；
- certificate SHA-256：
  `bd385f0a01db4ad769f6fc1591c5cd79acafef5663d60f81ddee6cb9987991fc`；
- protocol SHA-256：
  `b114187900dfe84e793031a68e360afe4aa3881218fb1df8737c253b4149af51`；
- terminal grid difference：归一化积分陷阱电荷 `9.92921e-5`、峰值场/平均
  内建场比值 `3.07036e-2`；对应 terminal tolerance difference 为
  `5.07738e-11 / 1.97698e-7`，两端 contact occupancy 在两维均严格不变；
- 全矩阵最大 energy charge/occupancy/recombination relative change 为
  `5.02648e-4 / 5.02648e-4 / 4.80659e-4`；
- 最大 normalized Poisson residual `2.70886e-9`、Gauss-law relative error
  `2.16308e-8`、relative face current `7.32418e-14`、mass-action error
  `4.48251e-15`，最多 11 次 Newton iteration；
- 所有 cell 的 contact thermodynamics、拓扑、载流子正性、occupancy 有界、
  非零绝对 trap charge 与默认 production-path rejection gate 均通过。

该证书只覆盖 dark、homogeneous、Maxwell-Boltzmann、fully-ionized 的两层
silicon p/n equilibrium，并冻结一个空间均匀 truncated-Gaussian acceptor-like
分布。它不覆盖 bias/illumination、瞬态占据、Fermi-Dirac trap kinetics、
heterojunction、ions、production experiment routes，也不是外部 TCAD 或实验验证。

### 2026-08-24 composition-graded CIGS optics certificate

`cigs-graded-optics-v1` 在 source commit `6ba9055`、source changes 为空且
BLAS/OpenMP 单线程环境下完成 9/9 cell，0 failed、0 missing、0 reused：

- run ID：`f6b38cbc8ea5dd92aacfc4141a362ddc2cd687bf9f138095098f91064259a620`；
- certificate SHA-256：
  `e14e9f6f50c958e2e18dd514984e026d430688e785a9986892d0e21dfcff9958`；
- protocol SHA-256：
  `2031a7c5111de50b0e692ab69a605acab100bd84aff94620767385046bb01708`；
- terminal grid difference：absorbed flux `4.44665e-5`、generation centroid
  `1.24744e-4`、mean reflectance `1.44788e-5`、normalized generation profile
  `2.98005e-3`；对应 terminal KK-tolerance difference 为
  `8.78859e-7 / 3.56778e-8 / 1.35178e-6 / 1.57803e-6`；
- 全矩阵最坏 Carron composition median relative error `7.09067%`，完整
  Minoura/Carron ratio 范围 `0.662386-1.057105`，electrical/optical endpoint
  gap mismatch `8.52 meV`；
- photon-budget excess 与 reflectance-bound violation 均为 0，uniform
  composition reflectance difference 最大 `7.22e-16`；所有 causal `n,k`、
  topology、default-off、positive-flux、shared-coordinate 和 453-point
  Carron-completion gate 均逐 cell 通过。

该证书只认证 frozen research preset 的 composition-resolved CIGS absorber
optics、TMM slice/KK 收敛和列出的内部物理门。ZnO/CdS 仍是 nominal scalar
fallback；它不包含 transport solve，也不是 measured-device、外部 SCAPS/Setfos、
J-V/PCE 或实验验证。

### 2026-08-24 combined 2D mobile-ion/interface-SRH certificate

`twod-mobile-ion-interface-srh-v1` 在 source commit `0c9eb26`、source changes
为空且 BLAS/OpenMP 单线程环境下完成 9/9 cell，0 failed、0 missing、0 reused：

- run ID：`89d108b8817fb4af5d0749bd5848efada9dda99a1b559ed395a8cb0603eaa55b`；
- certificate SHA-256：
  `b02bc4f8b3b5d470d599f6dacde746b26c263591aafecd14cc6c890a94b677dd`；
- protocol SHA-256：
  `2b5371b8f89c2c4a749250fe13844495ca6750181fc4232579ed7c82d1775eee`；
- terminal grid differences：complete current `6.095485e-3 A m-2`、interface
  current `4.278045e-9 A m-2`、lateral variation `4.980758e-5`、maximum ion
  site fraction `6.800086e-5`、ion redistribution `3.956838e-4`；对应
  terminal tolerance differences 分别为 `1.55e-15 / 4.76e-22 /
  2.73e-15 / 3.12e-17 / 2.10e-15`；
- 全矩阵最大 ion inventory drift `7.49e-16`、all-face complete-current
  spread `5.71e-14 A m-2`、current-decomposition relative error `2.93e-16`、
  physical GB width relative error `4.97e-16`；
- 所有 cell 的 active-ion/carrier positivity、site occupancy、ion diagnostics、
  clamp-inactive、finite positive interface rates、combined topology、explicit
  protocol 和三电压完成 gate 均通过；minimum lateral response `1.95e-3`、
  minimum ion redistribution `1.18e-2`，证明未静默退化到 uniform/frozen lane。

该证书只认证一个 0.0/0.05/0.10 V、每点 10 ns 的 synthetic Neumann-x、ohmic、
blocking single-positive-ion、finite-width GB、clamp-inactive cross-node
`InterfaceDefect` slice。它不覆盖 dual ions、selective contacts、interface
charge/state、field mobility、long-time hysteresis、一般 2D microstructure、
外部 solver 或实验验证。完整边界见
`docs/twod-combined-numerical-certificate.md`。

### 2026-08-24 electrothermal terminal-MPP certificates

三条 lane 均固定同一 6-point/branch、20 V/s fresh-state strict J-V、同一
Rs/Rsh、800 W/m2 absorbed optical power、300--380 K thermal envelope、
observable、quality gate 与 componentwise tolerance ladder。前两条完整执行但
未通过预注册 grid gate，证据没有删除或改写：

- `electrothermal-terminal-mpp-v1` 在 commit `4d1273f` 的 10/15/20 ladder
  完成 9/9 cells，run ID
  `47798511b2cd28268f6dc1f9ba1c12693b4cef91b705277fb2b49e8fa1fa9a90`，
  certificate
  `8cb1a02d11efc7a5622d6d806a17be8c7ab6df0c4c6b7a40d8b2c1be2031ef7a`
  为 `partial`；300--380 K MPP-power response terminal-grid difference
  `3.26741% > 2%`。
- `electrothermal-terminal-mpp-resolved-v2` 在 commit `5605f6d` 的 20/30/40
  ladder 完成 9/9 cells，run ID
  `3ae21a1e4d09868fce73f840888739a8cff3a9d2877437e4f8d847e90bf6b42b`，
  certificate
  `533471fa0c64897ffde64ee7a9546657bf011fbff5529ddb79dddbfe93602578`
  仍为 `partial`；同一 response difference 为 `13.7361% > 2%`。

`electrothermal-terminal-mpp-grid-resolved-v3` 只将 ladder 提高到 40/60/80，
在 source commit `6afb1a6`、source changes 为空且 BLAS/OpenMP 单线程环境下
完成 9/9 cells，0 failed、0 missing、0 reused：

- run ID：`6056c1fb843c563b8de07c0a060713510cbeadf4134883641d40d1237ff1d8b7`；
- certificate SHA-256：
  `67d18979c9ce2d23a0e2d0e848513f50d8e1e4880819c92a5ff4da8719682cf6`；
- protocol SHA-256：
  `ecad8af783b4bd14c011b31780aec32f325bb9de72e515acf1eb2c031d3d7f01`；
- terminal grid differences：300--380 K MPP-power response `0.707880%`、
  operating temperature/rise `0.00150753 K`、terminal MPP current
  `0.0160342%`、power `0.0141103%`、voltage `1.23697e-5 V`；
- terminal tolerance differences：response `2.78525e-5`、temperature
  `1.13998e-7 K`、current `7.46909e-11`、power `6.57277e-11`、voltage
  `5.76206e-12 V`；
- 全部 cell 的 source/external/electrothermal certification、first-law exact
  reconstruction、protocol/hash alignment、temperature envelope、power residual、
  active series drop、shunt current 与 nontrivial temperature-response gates 通过。

因此 v3 certificate 只关闭 frozen synthetic inputs 下 fresh-state、forward
sampled terminal-MPP steady root 的内部 grid/tolerance/first-law/provenance gate。
它不认证 voltage-sampling refinement、continuous MPP optimizer、joint
electrical-thermal-ion transient、spatial heat equation、measured parameters、
external solver parity、lifetime 或实验温度。
