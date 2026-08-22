# SolarLab 物理与数值完备性加固实施/测试路线图（2026-08-14）

## 0. 文档定位

本文把 `docs/audit/2026-08-13-physics-numerics-audit-and-roadmap.md` 的审计结论转换为可执行的开发、测试和证据升级计划。目标不是一次性增加所有公式，而是按以下顺序建立可信度：

1. 先让默认行为、求解接受原因、接触相容性和阻抗前置条件可见；
2. 再完成网格、容忍度、残差和协议的内部数值认证；
3. 然后建设离子可见的小信号阻抗；
4. 最后才研究界面陷阱电荷、DAE 和更完整的材料本构。

本文同时是实施计划和 2026-08-14 工作树的 first-slice 状态快照，并由
2026-08-20 的 Phase 1 闭环审计和 Phase 2 step 1 DC checkpoint 更新。
Phase 1 聚焦 Python 回归为 `359 passed, 17 deselected`；DC checkpoint
后的仓库默认 Python 套件为 `1999 passed, 2 skipped, 263 deselected`，
其中默认 pytest 配置排除 `slow` 标记。前端
29 个测试文件为 `399 passed`，`tsc --noEmit` 与 Vite production build
通过；Python `compileall`、聚焦 Ruff 与 `git diff --check` 通过。先前
OneDrive dataless Git object 已仅做对象水合，未改源码或 manifest，P0
复现节点及最终全套均通过。这些结果把四个 P1 工作包的实现提升为
`INTERNAL_TESTED`；只有冻结 certificate 明确通过的 lane 才是
`INTERNAL_CERTIFIED`。当前 c-Si resolved-v2 与三条 regularization ladder
通过内部认证，有限速率 IonMonger J-V 仍为 `partial`，SCAPS/interface/2D
lane 为 `failed`。2026-08-20 新增的 Phase 2 step 1 DC lane 中，原始
N30/60/90 矩阵因 occupancy 网格差为 `partial`，versioned N60/90/120
resolved-v2 为 `INTERNAL_CERTIFIED`。这只清除了 ion-aware impedance 的
DC 数值前置条件；接触仍为 `compatible_unverified`，频域离子线性化尚未
实现，因此 Phase 2 仍未整体退出，更不构成外部或实验验证。

### 0.1 本轮边界

- 默认数值和默认物理不改变；新策略优先采用显式 opt-in。
- 保留 `compute_V_bi()` 的有符号约定：`phi(right)-phi(left)=W_left-W_right`。
- 不批量改写 shipped YAML 的 manual/legacy `V_bi`，不把校准兼容 deck 自动迁移为物理接触 deck。
- `iface_state_charge` 保持 `PARKED`，本轮不接通界面陷阱电荷。
- 不把内部残差、网格收敛、cross-code 校准或 SCAPS 趋势对齐写成实验验证。
- 不以一次单网格拟合替代容忍度阶梯、网格阶梯、独立参考和协议复现。

### 0.2 证据标签

审计结论的 `CONFIRMED / PLAUSIBLE / REFUTED` 与实现结果的证据级别是两条不同轴。后续 PR、API 和论文表格统一使用下列结果标签：

| 标签 | 含义 | 允许的表述 | 不允许的外推 |
|---|---|---|---|
| `PARKED` | 代码脚手架存在，但活动路径禁用或不公开 | “未启用；物理闭环尚未完成” | “已支持该物理” |
| `FIRST_SLICE` | 接口/元数据/诊断骨架已实现或正在实现 | “实现切片已出现，等待验收” | “数值已认证” |
| `INTERNAL_TESTED` | 单元/集成测试验证实现契约 | “实现按测试契约工作” | “物理模型已验证” |
| `INTERNAL_CERTIFIED` | 残差、守恒、网格与容忍度门槛均通过 | “在声明的模型/协议域内数值认证” | “与真实器件一致” |
| `CROSS_CODE_CALIBRATED` | 与外部代码在已披露校准下对齐 | “calibrated reproduction” | “无拟合外部验证” |
| `EXTERNAL_SOLVER_VALIDATED` | 冻结 deck、版本、协议和原始输出的独立交叉验证 | “外部求解器验证” | “实验验证” |
| `EXPERIMENT_VALIDATED` | 独立器件数据、误差和不可调参数协议均可追溯 | “实验验证” | “普适材料定律” |

每个可发布结果还必须携带：Git commit、环境、配置 SHA-256、网格、容忍度策略、求解接受类型、接触证书、协议、证据标签和已知限制。

## 1. Phase 0：可信度 first slice 与默认保持（当前至 2 周）

### 1.1 Phase 0 总览

| 工作项 | 2026-08-14 状态 | 默认行为 | 本阶段目标证据 |
|---|---|---|---|
| P0.1 componentwise absolute tolerance | `INTERNAL_TESTED`：opt-in 实现与 24 个聚焦测试 | scalar `atol` 不变 | 待 tolerance x grid 认证 |
| P0.2 steady-state acceptance metadata | `INTERNAL_TESTED`：接受原因、逐点 API 状态与 fail-closed chain | 旧 `converged` 兼容字段保留 | 待研究导出的 acceptance policy gate |
| P0.3 contact thermodynamic certificate | `INTERNAL_TESTED`：固定门槛、绝对 metal 对齐、诊断和 strict helper | 不自动拒绝 legacy deck | 待 shipped-deck 分类清单与外部接触证据 |
| P0.4 impedance diagnostics | `INTERNAL_TESTED` first slice：连续 AC、共点电流、频带/网格/DC 证据 | 旧方法别名和非 strict 路径保留 | 待幅值/循环/网格/DC 联合认证 |
| P0.5 interface trap charge policy | `PARKED` | 电静力电荷关闭 | 明确禁用和能力说明 |
| P0.6 claim/evidence wording | README、UI 和开发说明 first slice 已收缩 | 不改数值 | 待 registry 全量扫描 |

### 1.2 P0.1：opt-in componentwise tolerance

#### 实现范围

当前 first slice 以 `ComponentwiseAtol` 显式对象触发向量绝对容忍度，历史 scalar 路径保持原样。涉及文件：

- `perovskite_sim/solver/tolerances.py`：策略、1D/ion-only/2D 向量构造器；
- `perovskite_sim/solver/mol.py`：`run_transient()` 与 `split_step()` 接入；
- `perovskite_sim/solver/illuminated_ss.py`：预条件路径传播；
- `perovskite_sim/experiments/degradation.py`：快照测量传播与 refinement；
- `perovskite_sim/twod/solver_2d.py`：2D `(n,p)` 状态接入；
- `perovskite_sim/_compat/scipy_shim.py`：兼容求解器接受向量 `atol`；
- `docs/componentwise-tolerance-policy.md`：策略语义；
- `tests/unit/solver/test_tolerances.py`：实现契约。

策略应满足：

```text
atol_i = refinement_factor * max(minimum_atol,
                                 fraction_species * reference_i)
```

载流子参考量由局域 `ni^2`、`N_A`、`N_D` 的暗平衡质量作用关系得到；离子参考量由 `P_ion0`/`P_ion0_neg` 得到；显式界面状态暂以初态幅值构造误差地板。该构造是误差控制策略，不是载流子正性证明。

#### 测试矩阵与通过门槛

| 层级 | 用例 | 通过门槛 |
|---|---|---|
| 单元 | intrinsic、p-type、n-type、极端补偿掺杂 | 参考尺度有限、非负，且满足质量作用关系到浮点误差 |
| 单元 | single ion、dual ion、4-state/interface、2D flatten 顺序 | 向量长度和状态块顺序与 state layout 完全一致 |
| 单元 | NaN、Inf、负密度参考、shape mismatch、非法 refinement | fail-close，错误信息含字段名和预期形状 |
| 兼容 | scalar `atol` 经 1D、split-step、2D、degradation | 传给求解器的对象和值不变；默认结果数组 bit-identical |
| 集成 | 1D 无离子、单离子、双离子、界面状态；2D uniform | policy 到达实际 `solve_ivp`，不被中间包装器降回 scalar |
| refinement | factor `1, 0.1, 0.01`，固定 `rtol` 和网格 | 最后两级 `Voc <= 1 mV`、`Jsc <= 0.2%`、归一化 J-V `L_inf <= 0.5%` |
| 守恒 | 离子 transient/split-step | 每种离子库存相对漂移 `<= 1e-10`，且不因收紧容忍度恶化 |
| 暂态 | TPV/hysteresis/阻抗预条件代表点 | 最后两级关键 trace 归一化 `L_inf <= 1%`；未达门槛则只报告 sensitivity envelope |

上述观测量阈值是进入 benchmark registry 的候选门槛，必须按 lane 预注册；不得在看过结果后放宽。若某 lane 无法满足，证据标签停留在 `INTERNAL_TESTED`，不能升级为 `INTERNAL_CERTIFIED`。

#### Phase 0 退出条件

- scalar 默认路径的兼容性测试存在；
- 所有公开 transient 入口要么接受 `ComponentwiseAtol`，要么明确拒绝，不得静默忽略；
- 测试覆盖状态布局、非法输入和 refinement 组合；
- 文档不再把向量 `atol` 称为 positivity mechanism。

### 1.3 P0.2：steady-state acceptance metadata

#### 实现范围

`perovskite_sim/experiments/steady_state.py` 的 first slice 将求解终点区分为：

- `residual_converged`：严格非线性 residual gate；
- `current_bounded_stall`：Newton/line-search stall，但连续性电流上界通过；
- `transient_assisted`：J-V 点由 transient fallback 得到。

`SteadyStateResult` 保留兼容字段 `converged`，新增 `acceptance` 和 `residual_converged`；`JVSweepSSResult` 新增逐点 `point_acceptance`、`point_residual`、`point_continuity_current_bound`。涉及测试为 `tests/unit/experiments/test_steady_state.py`。

#### 测试矩阵与通过门槛

| 用例 | 预期接受类型 | 通过门槛 |
|---|---|---|
| 人工严格残差收敛 | `residual_converged` | residual 小于 strict gate；属性 `residual_converged=True` |
| 人工 step/line-search stall 且电流有界 | `current_bounded_stall` | residual 未伪装为 strict；电流上界有限且低于 gate |
| transient fallback | `transient_assisted` | residual 来源明确；不伪造 continuity-current bound，未知值用 NaN |
| fallback 边界 | Dirichlet 与 Robin/selective | 只排除真实 algebraic pin；Robin 边界残差必须通过 guard |
| 完整 J-V | 混合逐点状态 | 三个 metadata 数组与 `V/J` 等长且逐点一一对应 |
| 旧调用者 | 任一成功状态 | `converged` 兼容语义不破坏现有调用；数值输出不因新增元数据改变 |
| API/序列化 | 三种状态 | label、residual、bound 均可 JSON 序列化，不丢失 NaN 语义或改成 0 |

Phase 1 起，研究级导出默认要求用户显式选择接受策略：`strict_residual_only` 或 `allow_current_bounded_with_label`。`transient_assisted` 点可用于连续曲线展示，但不能自动继承稳态 Newton 的残差证书。

### 1.4 P0.3：contact thermodynamic certificate

#### 物理检查

接触证书必须使用求解器实际消费的端点 band/DOS、载流子库和 Poisson 电势差，而不是重新推导一套平行参数：

```text
E_C = -phi - chi
E_V = E_C - E_g
E_Fn = E_C + V_T * ln(n / N_C)
E_Fp = E_V - V_T * ln(p / N_V)
```

零偏热平衡的内部必要条件是左/右两端 `E_Fn` 和 `E_Fp` 共用同一个 Fermi level。first slice 位于：

- `perovskite_sim/physics/contacts.py`：`ContactThermodynamicCertificate`、`assess_contact_thermodynamics()`、`require_contact_thermodynamic_certificate()`；
- `tests/unit/physics/test_contacts.py`：legacy/manual、semiconductor work-function 和 metal work-function 用例。

当前 tolerance 固定上限为 `5 meV`，调用者只能收紧，不能放宽后获得 `certified`。它是版本化内部 evidence gate，不是普适接触物理误差条；精确构造的合成平衡测试仍应达到接近机器精度。状态区分 `certified`、`inconsistent`、`compatible_unverified` 和 `not_assessable`。

#### 测试矩阵与通过门槛

| 维度 | 用例 | 通过门槛 |
|---|---|---|
| 极性 | p-i-n、n-i-p、正/负 signed `V_bi` | 不在源头取 `abs()`；合成自洽态 QFL span `< 1e-10 eV` |
| 模式 | legacy/manual、`semiconductor_work_function`、`metal_work_function` | 只按实际 reservoir/potential 认证；模式名不自动授予证书 |
| 温度 | 250/300/350 K，含 DOS 温标和 Varshni | 使用 device `V_T` 与温度后 DOS；无 300 K 硬编码泄漏 |
| 端点 | graded Eg/chi、不同 Nc/Nv、补偿掺杂 | 使用真实端点数组而非 layer nominal 值 |
| `ni` 一致性 | `ni_300` 与 Nc/Nv/Eg 一致/故意不一致 | 不一致时 strict path fail-close |
| metal reservoir | metal work function 匹配/不匹配 doping-derived reservoir | 同时检查 QFL span 与两端绝对 metal/semiconductor work function；共同平移也不能获证 |
| flat-band floor | metal carrier floor on/off、T 不等于 300 K | floor 与温标一致；证书使用 floor 后实际密度 |
| 数据缺失 | band/DOS 完整、仅 band、band-less legacy | 不把“算不出来”标成 `certified` |
| 边界类型 | Dirichlet、Robin/selective contact | 相同平衡库给出相同 thermodynamic report；动力学参数另行验证 |
| shipped deck | SCAPS、IonMonger、Driftfusion、CIGS、c-Si | 生成分类清单；不自动修改 deck 或因 legacy 身份一刀切报错 |

`tests/unit/physics/test_contacts.py` 中的 SCAPS fixture 已量化并通过：
legacy tilt 约 `6.025 meV`，显式 semiconductor-work-function 模式接近
零，任意 metal-work-function 组合暴露约 `0.194 eV` 的 reservoir
mismatch；把两端已匹配 metal work function 共同平移 `+1 eV` 时，跨端
QFL span 仍接近零，但绝对功函数 mismatch 为 `1 eV`，因此保持
`inconsistent`。250/350 K 的 physical mode 也使用活动温度下的接触势，
不再用 300 K legacy 估计制造假 mismatch。这证明诊断计算契约，不证明
`5 meV` 是普适物理误差界。

#### 运行政策

- 默认：返回/导出证书，但保持 legacy 兼容行为；
- `report`：所有实验结果带证书和原因；
- `strict`：研究工作流要求 `certified`，否则在求解前 fail-close；
- `not_assessable` 不能升级为 `certified`，但可在显式兼容 lane 中继续运行；
- 不把接触证书解释成 surface recombination velocity、Schottky barrier、tunneling 或交换电流的外部验证。

外部升级至少需要温度依赖暗 J-V、已知 barrier/injection 数据或冻结版本的独立求解器 deck，同时约束 work function、DOS、接触选择性和传输模型。

### 1.5 P0.4：impedance diagnostics first slice（已实现，未完成数值认证）

#### 当前实现

`perovskite_sim/experiments/impedance.py` 已加入：

- `ImpedanceProtocol`：方法、`V_dc`、`delta_V`、明暗、DC settle、cycles、extract window 和 `points_per_cycle`；
- `IonicTimescale` / `FrequencyWindowAssessment`：Debye、blocking-charge 和 diffusion 频率估计；分别报告特征频率是否 bracketed，以及具有两侧 decade margin/最大 0.5-decade 采样间隙的 ionic branch coverage；
- `OperatingPointCertificate`：载流子/离子面积残差、离子面电流、DC 全面电流 spread、接触证书；
- `GridAssessment`：guarded cells、offenders、最大 cell/Debye 比、override 和独立网格状态；
- `ImpedanceDiagnostics`：admittance、all-face spread、reciprocal condition、backward error、载流子 storage response；
- `require_operating_point_certificate`：默认关闭的 strict gate；
- 明确方法名 `transient_ion_aware` 与 `qf_frequency_ion_free`，保留旧 alias。

Transient AC 不再把每个小区间的正弦电压冻结为常数并反复重启
`solve_ivp`。每个频率只运行一次接受连续 `V(t)` 的 Radau 积分，同时保留
edge/midpoint 状态；centered displacement current 与 midpoint conduction
current 在同一 lock-in 时间戳组合。生产循环的纯电容回归在 40 点/周期下
给出 `-90 deg` 相位并通过幅值误差 `<0.2%` 门，消除了原固定 `4.5 deg`
时间错位偏差。非有限 DC/QF 证据无条件停止，动态 interface-state block
在 residual 与电荷共同认证前显式 capability-fail；证书/能力失败在 blocking
API 映射为 HTTP 422。

`backend/main.py` 已透传 method、amplitude、cycles、extract、DC settle、
illumination 和 strict gate，并保留多维 complex diagnostics 的形状；两个
前端 impedance pane 已暴露这些控件，结果图会显示 DC 未认证和频带遗漏
告警。IonMonger N30 的筛选尺度为 Debye length `1.467 nm`、dielectric
`0.747 Hz`、blocking charge `5.49 mHz`、diffusion `1.01e-5 Hz`，所以
默认 `10 Hz--100 kHz` 明确标记为未 bracket ionic blocking scale，且不覆盖
ionic branch。单个恰好落在 blocking scale 的频点、以及只给出极宽的两个
端点，都不能获得 branch coverage。

这仍只是 Phase 0 first slice。聚焦单元/API 契约已进入回归，但有限时间
DC settle 不称为真正稳态。`qf_frequency_ion_free`
已有的线性代数证书不能被移植成“离子阻抗证书”；
`transient_ion_aware` 能看见离子也不等于低频已收敛。

#### 已覆盖与必须继续补齐的测试矩阵

| 层级 | 用例 | 通过门槛 |
|---|---|---|
| 单元 | 无离子、单离子、双离子、多个不连续活性区 | timescale 有限、单位正确、region segmentation 稳定 |
| 单元 | 单点、稀疏端点、dense sweep | bracket 与 coverage 分列；一侧 decade margin、最大 log-gap `<=0.5 decade`；warning 不静默消失 |
| 单元 | 非法频率、`delta_V >= 20 mV`、非法 settle/gate | fail-close，错误信息标出协议字段 |
| 单元 | 人工零 RHS、人工未 settle RHS | operating certificate 分别通过/失败，原因列表可判定 |
| 接触组合 | contact `certified/inconsistent/not_assessable` | 数值证书与热力学证书分列；strict 需要两者同时通过 |
| QF adapter | `Y`、face spread、rcond、backward error、storage | 所有底层诊断原样进入统一结果，不只返回 `Z` |
| transient 时间离散 | 纯电容生产循环；40/80/160 points-per-cycle | 40 点幅值误差 `<0.2%`、phase `-90 deg`；后续器件阶梯最后两级 phase 差 `<0.5 deg` |
| transient 线性 | `delta_V = 10, 5, 2.5 mV` | 最后两级 `|Z|` 相对差 `< 1%`、phase 差 `< 0.5 deg` |
| transient | cycles/extract/settle 加倍 | 最后两个 extraction window 的复导纳差 `< 1%`；否则不得认证 |
| 网格 | N30/N60/N90 或预注册 grid ladder | 最后两级 `|Z| < 2%`、phase `< 1 deg`，每个频点均检查 |
| 频域线性代数 | 每个频点 | QF lane all-face relative spread `<= 5e-4`、backward error `<= 1e-10` |
| 解析数值参考 | Randles/RC lock-in | 只认证 lock-in 数值和符号，不作为器件物理验证 |
| API | dataclass/complex array/non-finite/warnings | 保留多维 complex shape；非有限证据 fail-close；证书/能力失败为 HTTP 422 |

first slice 的残差阈值（载流子面积率、离子面积率、DC face spread）必须在 Phase 1 通过 refinement 数据定标并写入 registry；在此之前只能称 candidate internal gates。

### 1.6 P0.5：界面陷阱电荷明确 `PARKED`

当前代码状态：

- `perovskite_sim/solver/mol.py::_charge_density()` 的主 `rho` 不含 bulk trap occupancy charge；
- `MaterialArrays.iface_state_charge` 默认 `0.0`；
- `compute_interface_trap_charge()` 与 shared-node Poisson 加项存在，但活动 steady-state 路径未赋予非零 charge；
- 动态 MoL interface-state block 在 impedance 入口显式拒绝；QF 的局部 interface lane 即使可求 occupancy，也尚未把 charge 放回 outer Poisson；
- `two_sided_interface.py` 有固定片电荷的 Gauss 跳跃数学脚手架，但生产外层 Poisson 未把占据依赖片电荷闭合进去。

因此 Phase 0 必须统一表述为：界面平面态当前是 recombination-only；trap electrostatics 未启用。API/UI/config 不得暴露 `-1/+1` 的自由符号旋钮。

#### donor/acceptor 的平衡参考增量符号

令 `f` 为电子占据率。绝对电荷可写为：

```text
acceptor-like: sigma_A(f) = -q * N_t * f
donor-like:    sigma_D(f) = +q * N_t * (1 - f)
```

但若 Poisson 加的是相对于同一暗平衡参考态的增量，则：

```text
Delta sigma_A = sigma_A(f) - sigma_A(f_eq)
              = -q * N_t * (f - f_eq)

Delta sigma_D = sigma_D(f) - sigma_D(f_eq)
              = -q * N_t * (f - f_eq)
```

两者同号。当前“acceptor `-1`、donor `+1` 乘以 `q N_t(f-f_eq)`”的接口若直接开放，会使 donor-like 的平衡参考增量符号错误。若未来选择绝对电荷语义，donor/acceptor 才有不同绝对符号，但必须同时定义 neutral occupancy、固定补偿电荷和全器件电中性，不能只换符号。

#### PARKED 原因和解锁条件

| 风险 | 当前问题 | 解锁门槛 |
|---|---|---|
| `f_eq` | 未存储与同一拓扑、同一接触证书、同一参数求得的残差认证暗态占据 | charge-off 暗态计算并冻结逐界面 `f_eq`；reference-on 暗态 `Delta sigma=0` 到机器精度 |
| gauge | heterojunction 两侧 band/QFL 投影和 trap energy 参考可能不在同一能量 gauge | 显式 energy-reference 契约；左右投影在热平衡给出相同 occupancy |
| 符号 | raw donor/acceptor multiplier 混淆绝对电荷与增量电荷 | helper 直接返回带物理符号的 `Delta sigma`；移除任意 scalar sign |
| 空间离散 | 全片电荷沉积到 shared node，结果可依赖 node ownership、epsilon 跳变和网格 | 使用 two-sided trace/Gauss jump 或严格有限体积 sheet source；通过积分电荷和 D-field jump 测试 |
| outer Poisson | QF 局部状态当前在外层 Poisson 之后求解；事后算电荷不是自洽闭环 | occupancy/charge 进入每次外层 Poisson residual，并提供一致 Jacobian/IFT 导数 |
| transient | occupancy 没有统一的守恒动态/代数 DAE 语义 | Phase 3 仅 steady-state research opt-in；transient 在 Phase 4 DAE 前 fail-close |
| calibration | 打开电荷会改变既有 `iface_state_calibration_factor` 和 SCAPS 幅值 | 原校准证据降级；重新做无调参内部和独立外部比较 |

Phase 0 测试只验证 parked policy：默认 charge 为零、公开入口不能误启用、所有不支持组合显式报错、charge-off 数组与历史基线 bit-identical。

### 1.7 P0.6：收缩对外主张

必须在 README、API capability response、benchmark registry 和报告模板中统一：

- 2D：仅“横向均匀性/寿命图样研究”，不是完整 perovskite 2D 微结构物理；
- CIGS grading：仅“graded transport”，在 graded `alpha(lambda,x)`/`n,k(x)` 落地前不主张 notch PCE 优化；
- SCAPS/Driftfusion/IonMonger：披露校准项，使用 `calibrated reproduction` 或 `trend-aligned lane`；
- interface states：recombination-only、calibration scaffold disclosed；
- impedance：分别标记 ion-aware transient 与 ion-free certified frequency-domain，不合并为一个“已认证离子阻抗”功能；
- contact certificate：只证明内部边界热力学算术相容，不证明真实接触参数。

通过门槛：`reproducibility/config_benchmark_matrix.yaml` 中每个 benchmark 有 `claim_level`、`evidence_tier`、`limitations`、协议和配置；API 与文档的 capability label 快照测试一致。

## 2. Phase 1：内部数值认证与协议可复现（2 至 6 周）

### 2.1 P1.1 tolerance × grid 联合阶梯

#### 文件计划

- 新增 `perovskite_sim/validation/numerical_certificate.py`：统一 certificate schema；
- 新增 `scripts/run_numerical_refinement.py`：可恢复、带 manifest 的网格/容忍度矩阵；
- 新增 `tests/regression/test_tolerance_grid_refinement.py`；
- 修改 `reproducibility/config_benchmark_matrix.yaml` 和 schema registry；
- 结果写入新的、可哈希的 reference YAML/JSON，不覆盖历史结果。

#### 最小 benchmark 矩阵

| lane | 网格 | tolerance factor | 关键输出 |
|---|---|---|---|
| SCAPS mirror frozen-ion SS | N30/N60/N90 | 1/0.1/0.01 | J-V、Voc、Jsc、FF、residual、acceptance |
| IonMonger mobile-ion transient | N30/N60/N90 | 1/0.1/0.01 | ion inventory、hysteresis、terminal current trace |
| c-Si QF/frequency | N100/N200/N300 | FD step ladder | C-V、face spread、backward error |
| 2D uniform limit | matched x/y ladders | 1/0.1/0.01 | 2D-to-1D envelope、charge/current conservation |
| interface recombination charge-off | N30/N60/N90 | 1/0.1/0.01 | interface flux residual、J-V trend |

退出门槛：每条认证 lane 同时通过最后两级网格与容忍度门槛；否则 registry 状态为 `partial` 并存储未收敛维度。禁止只挑一个收敛的标量报告。

### 2.2 P1.2 positivity 与非物理 trial-state 诊断

#### 实施顺序

1. 在 `solver/mol.py`、`physics/recombination.py` 和实验结果中记录 final/minimum density、negative trial count、最小 SRH denominator；默认先诊断，不改变 RHS。
2. 为负 terminal state、非有限 RHS、SRH denominator 接近零建立 fail-close 研究模式。
3. 在独立 opt-in lane 原型比较 log-density 状态或 positivity-preserving update；不在未验证时替换 Radau 默认状态。
4. 检查载流子、单/双离子和 interface state 的守恒/正性是否同时成立，不能以 clip 掩盖库存变化。

测试覆盖暗耗尽、强注入、深 cliff/spike、低温、极低 `ni`、高 trap density 和快速偏压阶跃。进入 `INTERNAL_CERTIFIED` 的门槛是：输出状态严格正且有限；无 silent terminal clipping；连续性/库存门槛保持；与原方程的 observable refinement envelope 闭合。

### 2.3 P1.3 RHS 非光滑项的可控正则化

文件范围：`physics/interface_plane.py` 的限幅/状态容量、`physics/field_mobility.py` 的 Poole-Frenkel `sqrt(|E|)`、相关 TE cap 路径和 solver diagnostics。

每个正则化必须：

- 默认关闭或保持零宽度极限；
- 有明确过渡宽度和单位；
- 在远离 kink 时恢复原公式到机器精度；
- 做 regularization-width ladder；
- 同时比较 Newton/Radau evaluations、残差和物理输出，不能只以“更快”作为通过。

候选门槛：宽度减半后关键 observable 变化 `< 0.5%`，残差不恶化，守恒不变；未收敛时保留原模型并报告 solver sensitivity。

### 2.4 P1.4 统一实验协议对象

新增 `perovskite_sim/experiments/protocol.py`，至少包含：初态来源、pre-bias、soak/dwell、明暗历史、温度、扫描方向/速率、AC amplitude、DC settle 判据和输出采样。J-V hysteresis、TPV、EQE、Suns-Voc、impedance 逐步采用同一序列化 schema。

测试门槛：相同 protocol round-trip 得到相同 hash；缺失历史字段的 legacy 调用被标为 `implicit_legacy_protocol`；研究 strict 模式拒绝隐式历史；不得因引入 protocol 对象改变旧默认数值。

## 3. Phase 2：Certified ion-aware small-signal impedance（1 至 3 个月）

### 3.1 目标架构

将当前两条路径的优势合并，但不混淆证据：

```text
residual-certified ion/electron/hole DC state
        -> state/voltage linearization
        -> (j*omega*M - J) delta_y = b*delta_V
        -> conduction + displacement + ionic all-face current
        -> per-frequency numerical certificate
```

主要文件：

- `perovskite_sim/experiments/impedance.py`：协议、统一结果和路由；
- `perovskite_sim/solver/small_signal.py`：通用频域线性代数；
- `perovskite_sim/experiments/quasi_fermi_impedance.py`：ion-free 基准 lane；
- 新增 `perovskite_sim/experiments/ion_aware_impedance.py`：离子线性化适配器；
- `perovskite_sim/solver/mol.py`：提供一致 residual/storage/current evaluation；
- `backend/main.py`：暴露协议、strict flag、warnings 和 certificate；
- 新增 `tests/unit/experiments/test_ion_aware_impedance.py`、`tests/integration/test_ionic_impedance_crosscheck.py`。

### 3.2 实现步骤

1. **`INTERNAL_CERTIFIED` (2026-08-20)**：使用显式 canonical protocol
   生成并认证 DC state；有限时间 settle 只有在独立载流子/离子残差、
   逐物种离子面电流、全器件 current spread、离散库存、正性和 occupancy
   门槛连续两次通过后才晋级。N30/60/90 v1 保留 occupancy 网格失败，
   N60/90/120 resolved-v2 在原门槛下通过。接触证书仍独立为
   `compatible_unverified`。详见
   [ion-aware-dc-certification.md](../ion-aware-dc-certification.md)。
2. **`INTERNAL_TESTED_REFERENCE` (2026-08-21)**：电子、空穴、正/负离子
   的真实动态自由度以 log-density increment 进入 `M`；固定接触和结构性
   零离子节点排除。Poisson 继续全局消去，但每个 state/voltage stencil
   都重新求解，因此其全局导数进入 `J`、forcing 和 displacement。
3. **`INTERNAL_TESTED_ANALYTIC_TRANSPORT` (2026-08-21)**：reference central
   finite difference 保留 `1/0.5/0.25` 三层；structured comparison 已实现
   精确离散 Poisson 隐式灵敏度、解析 mass tangent、载流子及单/双离子 SG
   面通量导数和守恒一致的 continuity-divergence 修正，并对 `M/J/b`、逐分量
   电流和阻抗幅相 fail-close。interface recombination、contact 和
   field-mobility 导数仍未解析化，因此尚不是 full analytic Jacobian。详见
   [ion-aware-structured-jacobian-comparison.md](../ion-aware-structured-jacobian-comparison.md)。
4. **`INTERNAL_TESTED_ANALYTIC_BULK_REACTION` (2026-08-21)**：局域 bulk
   SRH、radiative、Auger 生产公式已有精确 `dR/dn`、`dR/dp`，并按每列
   scaled log-density tangent 同时进入 electron/hole continuity 行。独立
   central stencil、完整 rate Jacobian、N13/N61/N91 阻抗幅相共同 fail-close；
   radiative reabsorption 和 heterojunction de-spike 非局域分支尚未解析时
   拒绝进入该 lane。
5. 每个频点返回 `rcond`、componentwise backward error、all-face admittance spread、storage decomposition 和 perturbation-step sensitivity。
6. 根据物理 timescale 自动建议频带，但只 warning，不偷偷改用户频率。
7. 用 transient lock-in 在少量频点独立交叉检查；两条方法必须共享 DC state 和协议。

### 3.3 测试矩阵与通过门槛

| 维度 | 矩阵 | 通过门槛 |
|---|---|---|
| 解析系统 | RC、Randles、单 Debye relaxation | `Z` 幅相误差 `< 0.1%` |
| 物种 | no ion、single positive、dual positive/negative | no-ion 极限回到 QF/电子 lane；交换离子标号不改变对称系统结果 |
| 边界 | blocking ion、显式 reservoir（若支持） | blocking 总离子数守恒；reservoir 必须显式 Dirichlet 协议，不得混用 |
| 频率 | `1e-4` 至 `1e6 Hz` 的设备相关子区间 | blocking/diffusion/electronic plateaus 均有覆盖证据；不覆盖则 warning |
| 扰动 | FD state/voltage step 各减半两次 | 最后两级 `|Z| < 1%`、phase `< 0.5 deg` |
| 网格 | N30/N60/N90 或更严格预注册 ladder | 最后两级 `|Z| < 2%`、phase `< 1 deg` |
| 线性求解 | 每个频点 | backward error `<= 1e-10`，all-face relative spread `<= 5e-4` |
| 跨方法 | frequency solver vs transient lock-in | 选定频点 `|Z| < 3%`、phase `< 2 deg`，使用相同 DC/protocol |
| 被动性 | 纯被动无 generation benchmark | 不出现无物理来源的负耗散；允许的 inductive loop 必须有状态分解证据 |
| 性能 | N60、30 frequencies | 记录 wall time/peak memory/Jacobian evaluations；性能回归不超过基线 25% |

### 3.4 外部验证边界

- 解析 RC/Randles 只验证数值提取和符号；
- 与 transient lock-in 一致只属于内部交叉认证；
- IonMonger/Driftfusion 比较必须冻结代码版本、deck、preconditioning、frequency window 和原始复数输出；
- 真实器件验证必须提供面积、温度、bias/light history、AC amplitude、接触/寄生参数和测量误差；
- equivalent-circuit 拟合不能反向证明微观离子参数唯一。

Phase 2 退出时可使用“internally certified ion-aware impedance”；在独立外部产物完成前不能使用“externally validated ionic spectroscopy”。

## 4. Phase 3：界面态电静力闭环研究 lane（2 至 5 个月，继续默认关闭）

### 4.1 进入条件

只有同时满足以下条件才从 `PARKED` 进入 research opt-in：

- P0 parked policy 和符号测试完成；
- 接触 strict certificate 可用于参考暗态；
- charge-off steady state 有 residual/grid/tolerance 证书；
- two-sided Gauss jump 在 epsilon 不连续网格上已单独认证；
- 研究配置明确接受原 SCAPS calibration 失效并重新建基线。

### 4.2 数据模型和 API

弃用 `iface_state_charge: float`，改为显式枚举：

```text
interface_charge_closure:
  off                       # default
  equilibrium_referenced   # steady-state research only
```

每个界面存储：defect type、`N_t`、trap energy reference、`f`、同一暗态得到的 `f_eq`、`Delta sigma [C/m2]`、trace potential shift、Gauss residual 和 evidence status。绝对电荷模式暂不实现。

目标文件：

- `models/device.py` / `models/config_loader.py`：显式 schema 与不支持组合检查；
- `physics/interface_plane.py`：occupancy 与带符号 `Delta sigma`，不返回“正 magnitude + 外部符号”；
- `physics/two_sided_interface.py`：占据依赖 sheet charge 和导数；
- `experiments/quasi_fermi_steady_state.py`：将 interface QSS 放入 outer Poisson iteration；
- `solver/mol.py`：保持 legacy charge-off，研究 lane 不再向 shared node 任意 lump；
- `backend/main.py`：只对 steady-state research endpoint 暴露；
- 新增 `tests/unit/physics/test_interface_trap_charge.py`、`tests/integration/test_interface_charge_poisson_closure.py`。

### 4.3 求解顺序

```text
contact-certified dark state, charge off
        -> store f_eq per physical interface
biased/light outer Newton iterate
        -> carrier/QF projections at two traces
        -> local QSS occupancy f
        -> Delta sigma = -q*N_t*(f-f_eq)
        -> two-sided Gauss jump + outer Poisson residual/Jacobian
        -> carrier/interface continuity residual
        -> joint convergence certificate
```

不能采用“先解 Poisson、再解 local QSS、最后只在输出中计算 charge”的顺序；那不是自洽电静力。若用 IFT 消去 local interface state，必须验证 `d sigma / d state` 对 central FD 的相对误差。

### 4.4 测试矩阵与通过门槛

| 层级 | 用例 | 通过门槛 |
|---|---|---|
| occupancy | 对称 SRH、不同 capture velocity、极端 n/p | `0 <= f <= 1`；共同缩放速度不改变稳态 f |
| reference | certified dark state | `Delta sigma=0` 到机器精度；charge-on/off 暗态 bit-identical |
| 符号 | donor-like、acceptor-like 增量 | 二者均满足 `-q*N_t*(f-f_eq)`；禁止 raw `+1 donor` |
| 量纲/上界 | `N_t` 从 `1e8` 到 `1e13 cm^-2` | 线性随 `N_t`；`|Delta sigma| <= q*N_t` |
| Gauss | 正/负 sheet、相同/不同 epsilon | `D_right-D_left` 与 sheet charge 的符号/幅值一致，残差 `< 1e-10` 归一化 |
| 网格 | N30/N60/N120，界面左右非对称 clustering | integrated sheet charge 网格无关；barrier shift 最后两级 `< 1 meV` |
| outer coupling | analytic/FD Jacobian | 关键导数相对差 `< 1e-4`；outer residual 达预注册 gate |
| 设备 | dark、bias、light；Et/CBO/Nd/Nt sweep | dark reference 不变；bias/light barrier 变化方向可由 charge sign 解释 |
| 退化保护 | charge off | 所有历史配置和输出 bit-identical |
| 能力门 | transient、QF impedance、2D 未支持组合 | fail-close，不得 silent no-op |

### 4.5 证据和外部边界

`equilibrium_referenced` 只补“相对暗态”的增量电静力，不等于绝对 trap charge 或完整 equilibrium band bending。绝对闭环还需要 fixed countercharge、neutral reference 和全器件电中性。Phase 3 最多升级为 `INTERNAL_CERTIFIED`；与 SCAPS 的幅值重新接近仍只是 cross-code evidence，且任何重新调过 `iface_state_calibration_factor` 的结果只能标为 `CROSS_CODE_CALIBRATED`。

## 5. Phase 4：统一 DAE 与材料物理完备性（6 至 12 个月，分决策门推进）

Phase 4 不是一个大 PR。每个子 lane 必须单独立项、默认关闭、建立 analytic limit，并通过 Phase 1 的证书框架。

### 5.1 P4.1 transient DAE + algebraic interface states

目标：显式保留 Poisson/界面代数变量或采用可靠 Schur/IFT，使 transient ions、algebraic interface states 和 small-signal 使用同一状态拓扑。

文件候选：新增 `solver/dae.py`、`solver/jacobian.py`，重构 `solver/mol.py`、`physics/interface_plane.py`、`experiments/impedance.py`。先实现 no-interface/no-ion 极限，再加入单离子、双离子和 algebraic interface state。

通过门槛：DAE algebraic residual、differential residual、charge conservation 分列；no-ion 极限与现有 MoL 在 refinement envelope 内；consistent initial condition 可重复；analytic/AD Jacobian 与 FD 一致；N 翻倍的成本增长明显优于 dense FD 基线。未达到时不能替换默认 MoL。

### 5.2 P4.2 degenerate semiconductor closure

针对 c-Si/高掺杂 lane，依次加入：Fermi-Dirac statistics、incomplete ionization、band-gap narrowing，再评估高场/复合修正。

目标文件：`physics/statistics.py`（新增）、`physics/temperature.py`、`models/parameters.py`、`solver/mol.py`、contact certificate。测试包含 MB 稀薄极限、Fermi integral 参考值、charge neutrality、温度扫描、高掺杂 p/n 发射极和 analytic PN depletion limit。

进入 `EXTERNAL_SOLVER_VALIDATED` 前需要冻结 Sentaurus/PC1D/其他可信参考的版本与输入；进入 `EXPERIMENT_VALIDATED` 前需要独立暗 J-V/C-V/temperature 数据。实现公式本身只到 `INTERNAL_CERTIFIED`。

### 5.3 P4.3 能量分布陷阱与 bulk trap charge

`distribution: gaussian`、`E_char_eV`、`N_peak_cm3` 当前不能继续只作为“已验证但未消费”字段。路线是：

1. 明确 single-level 与 energy-distributed schema，旧字段若未消费则 capability label 为 parked；
2. 采用可控能量 quadrature，验证 single-level/delta-distribution 极限；
3. 将 recombination 与 occupancy 使用同一能量积分；
4. 只有定义 `N_t`、capture kinetics、charge transition/neutral reference 后才加入 bulk trap charge；不能由 `tau` 或空间 profile 反推电荷。

测试门槛：能量节点加倍后 recombination/charge `< 0.5%`；积分占据有界；detailed-balance equilibrium residual 通过；bulk integrated charge 与 Poisson Gauss balance 一致。

### 5.4 P4.4 graded optics、2D 和器件外部本构

按研究需求选择，不并行承诺全部完成：

- CIGS：实现 composition-dependent `alpha(lambda,x)` 和 `n,k(lambda,x)`，与 graded Eg/chi 使用同一 composition field；
- 2D：只有在加入移动离子、界面重组和 grid-independent grain-boundary area 后，才扩大“完整微结构”主张；
- 外部器件：串联电阻、并联泄漏、自热/热电耦合必须以独立 circuit/energy balance 层实现，不把它们塞进 contact calibration；
- 高通量/反演：在前向 certificate 可机器读取后，再做并行、敏感度、identifiability 和 UQ。

各 lane 的共同门槛：1D/uniform/zero-coupling analytic limit；网格与参数 quadrature 收敛；默认关闭；独立 benchmark；registry 中明确 `claim_level`。没有 graded optical data 时，CIGS 仍只能称 graded transport；没有 2D ion/interface closure 时，2D scope 不升级。

### 5.5 P4.5 identifiability，而不是直接参数拟合

优先回答 `iface_state_calibration_factor`、`het_recomb_despike`、trap density、capture velocity 和 ion parameters 是否可由 J-V/scan-rate/TPV/impedance 联合识别。先用 synthetic recovery、profile likelihood/Fisher rank 和多初值检查结构可辨识性，再考虑 Bayesian posterior。

通过门槛：synthetic truth 在无噪/有噪情况下的 recovery coverage 预注册；参数相关矩阵/秩亏显式报告；前向失败进入 likelihood 的规则固定；不能把宽 posterior 或结构不可辨识参数报告为精确材料常数。

## 6. 跨阶段统一测试与发布门

### 6.1 PR 级门槛

每个 PR 至少执行：

1. 变更文件的 targeted unit tests；
2. 相关 integration/regression lane；
3. `git diff --check`；
4. scalar/default compatibility 检查；
5. 新字段的 serialization/schema 测试；
6. benchmark registry 的 claim/evidence/limitations 校验。

任何测试结果必须记录命令、commit、环境和通过/跳过/失败数。本文不预填未来的 `N passed`。

### 6.2 慢门矩阵

| 门 | 频率 | 内容 | 失败处理 |
|---|---|---|---|
| smoke | 每 PR | unit + 小网格 integration | 阻止合并 |
| numerical | 每周 | tolerance/grid ladder、conservation、QF certificates | registry 降级，阻止证据升级 |
| protocol | 每周 | J-V/TPV/EQE/impedance protocol hashes | 阻止发布 |
| external | 每个 release candidate | frozen solver/raw-data comparisons | 不阻止内部开发，但禁止 external claim |
| experimental | 按数据集版本 | blinded or held-out device comparison | 失败保留并报告，不以重新校准覆盖 |

### 6.3 关键 observable 的统一记录

- 数值：residual、step、acceptance、RHS evaluations、Jacobian evaluations、wall time、peak memory；
- 守恒：electron/hole source balance、positive/negative ion inventory、Gauss residual、photon budget；
- 接触：四个 endpoint QFL、span、mode、data completeness；
- 界面：flux residual、occupancy、`f_eq`、sheet charge、D-field jump；
- 阻抗：DC certificate、frequency coverage、`Z/Y`、all-face spread、rcond、backward error、storage decomposition；
- 证据：config/reference hashes、calibration parameters、limitations。

## 7. 优先级、依赖和停止条件

```text
P0 metadata/default preservation
  -> P1 numerical + protocol certificates
     -> P2 certified ion-aware impedance
     -> P3 steady-state interface charge research lane
        -> P4 unified DAE / extended constitutive physics
```

P2 与 P3 可在 P1 后并行，但 P3 不得绕过 parked 解锁条件。P4 DAE 只有在 P2/P3 证明状态拓扑确有科研价值时启动。

停止或降级条件：

- tolerance/grid ladder 不闭合：只发布 sensitivity envelope；
- contact `not_assessable`：保留 compatibility label，不授予严格证书；
- impedance DC 未稳或 frequency window 不覆盖：不解释低频 ionic branch；
- interface charge 导致 dark reference 漂移：回到 `PARKED`，不得靠重新拟合掩盖；
- two-sided Gauss residual/网格收敛失败：禁止 shared-node workaround 进入生产；
- 外部结果需要重新调多个 scaffold 才对齐：标签保持 `CROSS_CODE_CALIBRATED`；
- 2D/graded optics/FD 等新 lane 无独立需求和数据：保持 scope 限制，不为“功能数量”实现。

## 8. 完成定义

本路线图的完成不是“所有 P0-P4 代码都存在”，而是以下证据链成立：

1. 默认兼容路径可复现，新物理/数值策略均可识别、可关闭；
2. 每个求解结果说明它为何被接受，不能再用一个 `converged=True` 覆盖不同终点；
3. 接触电势与载流子库具有运行时证书或明确的 compatibility/unassessable 标签；
4. 阻抗结果携带 DC、协议、频带和逐频数值诊断，离子可见与 ion-free lane 不混称；
5. 界面陷阱电荷在 `f_eq`、gauge、符号、Gauss jump 和 outer Poisson 闭环前保持 parked；
6. 内部数值认证、cross-code 校准、外部求解器验证和实验验证在 registry 与论文中严格分层；
7. 任何“validated/certified”主张都能追溯到冻结输入、测试门槛、原始产物和未通过项。

按此顺序推进，SolarLab 会先成为“能准确说明结果可信边界”的研究代码，再逐步成为具备离子谱学和界面电静力闭环的可认证平台，而不是在未闭环的物理上继续积累更多不可判定的开关。
