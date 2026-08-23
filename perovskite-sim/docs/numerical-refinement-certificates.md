# Phase 1 numerical refinement certificates

本文档定义 `reproducibility/numerical_refinement_registry.yaml` 的执行和证据契约。当前 registry 有 12 条 `grid x tolerance` lane，覆盖 Phase 1 minimum/resolved、ion-aware DC/impedance、2D uniform limit，以及 Phase 3 interface charge-off/charged research closure。阈值、配置内容哈希、adapter 和矩阵都在求解前固定；修改阈值必须使用新的 lane ID，不能根据已有结果原地放宽。

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
| `interface-recombination-charge-off` | N30/60/90 | QF residual factor 1/0.5/0.25 | two-sided interface flux、归一化 J-V、Voc |
| `interface-charge-equilibrium-referenced-v1` | N30/60/120 | QF residual factor 1/0.5/0.25 | charged current、occupancy、sheet charge、trace shift |
| `interface-charge-device-stress-v1` | N30/60 | QF residual factor 1/0.5 | 9-point Et/CBO/Nd/Nt current、dark occupancy、sheet charge、trace shift |
| `interface-charge-device-stress-resolved-v2` | N30/60/90 | QF residual factor 1/0.5 | v1 同一 9-point observable 与门限，N90 为 terminal grid |

归一化 J-V/trace 使用 absolute `L_inf <= 0.5%`，Voc 使用 absolute `<= 1 mV`，Jsc 使用 relative `<= 0.2%`。离子库存漂移 gate 为 `<= 1e-10`。c-Si 的 all-face spread 和 backward error gate 分别为 `<= 5e-4` 和 `<= 1e-10`。其余 lane-specific quality gate 见 registry；它们仍是 internal candidate gate，不能解释成外部物理误差条带。

2D 的 `grid` 是同步乘数：每层竖直区间数为 `5 x multiplier`，横向区间数为 `4 x multiplier`，匹配的 1D 区间总数为三层竖直区间总数。它不是只细化 y 而固定 x。1D 和 2D 两侧都比较电子加空穴的传导电流，不混入仅 1D 路径具有的离子/位移电流；除 refinement gate 外，每个 cell 还要求绝对 2D-to-1D 归一化差不超过 0.5%，横向均匀性同时检查 `n` 和 `p`。

## Protocol provenance

每个 production adapter 在 cell metadata 中写入：

- `protocol`：canonical protocol document；
- `protocol_schema`：experiment、numerical execution 或 bundle schema；
- `protocol_hash`：canonical document 的 SHA-256。

mobile-ion J-V 使用真实 `ExperimentProtocol`；c-Si 将每个 DC bias 的 QF impedance protocol 明确组成 bundle；frozen steady ladder 使用不虚构 scan rate 的 numerical protocol；charge-off interface lane 使用专用 two-sided QF protocol，绑定 contact certificate、暗态 occupancy reference 与 illumination ladder；charged interface lane 进一步绑定 `-q*N_t*(f-f_eq)`、暗态 bit identity、dark-bias/light targets 与 local IFT gates；device-stress companion 将 baseline 加 `E_t/CBO/N_D/N_t` 各两个 one-factor 端点、逐设备暗态参照和 charge/barrier 符号门限写入同一 canonical protocol；2D bundle 分列实际执行的 1D forward/reverse protocol、被比较的 forward branch 和 2D ascending finite-time protocol。2D 的 1D `v_rate` 显式设为 `V_step / settle_time`，使两条路径的每点 dwell 相同。

ion-aware DC 使用专用 frozen physical protocol，记录固定偏压、有效温度、
明暗历史、初态来源/可选初态 SHA-256、blocking ion 边界、ordered endpoint
ladder、连续通过次数和全部物理门槛。其外层 numerical execution protocol
另行记录 grid/tolerance 来源、componentwise `atol`、`rtol`、`max_nfev`
及 `Radau -> BDF` 方法阶梯。

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
python scripts/run_numerical_refinement.py interface-recombination-charge-off --dry-run
python scripts/run_numerical_refinement.py interface-charge-equilibrium-referenced-v1 --dry-run
python scripts/run_numerical_refinement.py interface-charge-device-stress-v1 --dry-run
python scripts/run_numerical_refinement.py interface-charge-device-stress-resolved-v2 --dry-run
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
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py interface-recombination-charge-off
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py interface-charge-equilibrium-referenced-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py interface-charge-device-stress-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py interface-charge-device-stress-resolved-v2
```

先执行一个 cell 并留下可恢复状态：

```bash
python scripts/run_numerical_refinement.py twod-uniform-limit --max-cells 1 --allow-noncertified-exit-zero
python scripts/run_numerical_refinement.py twod-uniform-limit
python scripts/run_numerical_refinement.py twod-uniform-limit --retry-failed
```

source、executor 或 environment 变化会产生新的 `run_id`，不会误接到旧 run。相同 run 重复执行只复用已验证的 cell。

## 真实矩阵成本和证据边界

当前 registry 共 12 条 lane；十条既有 lane 各 9 个 cell，device-stress v1/v2 分别为 4/6 个 cell，合计 100 个 content-addressed cell。其内部工作量至少包括：

- frozen SCAPS：279 个 residual-certified steady voltage points；
- mobile IonMonger：9 次 forward/reverse transient J-V，每次 40 个采样点，且可能触发 recovery/bisection；
- ion-aware DC：18 次 fixed-bias endpoint ladder；每段保留独立 DC state
  certificate，Radau 失败时再走 BDF，原始及 resolved 网格证据分别保留；
- ion-aware impedance：9 次 certified DC anchor 加完整 frequency/stencil
  response，并逐频率检查 backward error、库存响应和 current decomposition；
- c-Si minimum-roadmap lane：54 个 bias/FD-step operating-point calls，共 162 个 frequency points；resolved-v2 companion 另有同等工作量；
- 2D：117 个 2D finite-time voltage settles，外加 9 个 matched 1D forward/reverse sweeps 和 2D seed；
- interface charge-off：9 个 certified dark occupancy references、225 个
  requested two-sided QF illuminated voltage points及其 illumination ladders，
  并逐点计算 capture flux、local QSS、continuity 与 current-spread evidence。
- equilibrium-referenced interface charge：9 个重新认证的 charge-off dark
  anchors、9 个 dark-bias state 和 9 个 illuminated state，并逐点保存
  `f_eq/f/Delta sigma`、trace shift、Gauss、IFT condition 与状态哈希。
- interface-charge device stress：v1/v2 每个 refinement cell 内执行 9 个
  one-factor device variants；两条 lane 合计重建 90 个 contact-certified dark
  anchors，并求 90 个 dark-bias 与 90 个 illuminated target states。

因此 CI 只运行 schema、resume 和注入式 adapter smoke；完整物理矩阵应作为受控的长时任务逐 lane 运行。当前 infrastructure、dry-run 或 adapter smoke 不能替代这 100 个 cell 的真实结果。

`certified` 只表示该冻结代码/config/protocol/environment 下的内部数值收敛。它不等于 SCAPS/IonMonger 外部 solver parity，不等于实验验证，也不证明材料参数或模型闭包唯一正确。

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
