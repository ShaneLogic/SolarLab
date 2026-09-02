# SCAPS-1D M1–M3 多价缺陷 参数录入与导出交付单

面向执行 SCAPS 仿真的操作者。配套契约：`docs/multivalent-metastable-defect-contract.md`。
冻结 suite：`reproducibility/scaps_multivalent_defect_suite.json`。

与 S0–S2 交付单（`docs/scaps-s0-s2-collection-sheet.md`）结构平行；两份可对照阅读。

---

## 0. 本单可直接派发（2026-09-02 定案）

原先的四项阻塞全部关闭：

- importer 已存在：`scripts/import_scaps_multivalent_defect_reference.py`
  （fail-closed，与 S0–S2 的同构；6 个单测 + 10 个拒绝分支实测通过）；
- manifest schema 已定义：`solarlab.scaps_multivalent_defect_parameter_manifest`
  （协议文档 `docs/scaps-defect-m1-m3-reference-protocol.md`）；
- `charge_state_occupation_fraction_per_state` 编码已定案（§5.4，不再是提案）；
- 按 family 的符号规则已实现（M1 非负非零 / M2 非正非零 / M3 不设约束）。

配套模板：`docs/scaps-collection-templates/M{1,2,3}_profile_template.csv`
（表头逐字符即契约）与 `multivalent_parameter_manifest_skeleton.json`
（填完 `__FILL__` 即可通过 importer —— 已实测端到端验证）。

SCAPS 侧字段名已逐页对照 `docs/manual/SCAPSManual2016.pdf`（§3.6.2 多价缺陷、
§3.6.3 能级参考、§3.3 接触、§5.1.1 网格）校准，见 §3 各处标注。

---

## 1. 交付物清单

三个场景 M1 / M2 / M3，每个交三样，共十件：

| # | 文件 | 说明 |
|---|---|---|
| 1–3 | `M1.def` `M2.def` `M3.def` | SCAPS 输入定义文件，原样保存 |
| 4–6 | `M1.csv` `M2.csv` `M3.csv` | 逐位置剖面，列名见 §5.1 |
| 7 | `parameter_manifest.json` | schema `solarlab.scaps_multivalent_defect_parameter_manifest`，从 skeleton 填起 |
| 8 | SCAPS 版本号 | 精确版本串 |
| 9 | 操作者真实姓名 | 用于署名声明 |
| 10 | 导出日期 | ISO 格式 |

`derived_pn_device`（左 M1 / 右 M2 的 p-n 器件）**不需要你建 deck** —— 它由
SolarLab 从两个单层场景拼出来，不是从 SCAPS 导出的。

---

## 2. 红线

与 S0–S2 同：**不许插值 / 拟合 / 重采样 / 平滑**；**不许改数值**（只允许改表头）；
**不许改 §3 的任何参数**（改了 `config_sha256` 就对不上）；**必须用 SCAPS-1D**。

契约字段（`external_reference_contract`）逐条：

```text
independent_export_attestation_required: true
interpolation_allowed:                   false
required_solver:                         "SCAPS-1D"
source_deck_required_per_scenario:       true
status:                                  "not_supplied"   ← 交回后由我方翻转
```

---

## 3. 参数录入表

`configs/scaps_defect_m{1,2,3}_*.yaml` 是权威来源。下表已换算到 SCAPS 常用 cm 制。

### 3.1 三个场景的共用底表

**与 S0–S2 完全相同** —— 若已建过 S0–S2 的 deck，层参数可直接复用。

| 量 | 仓库值（SI） | **输进 SCAPS 的值** |
|---|---|---|
| 温度 T | 300.0 K | `300` K |
| 层厚 | 3.0e-7 m | `0.300` µm |
| 相对介电常数 εr | 20.0 | `20` |
| 电子迁移率 µn | 2.0e-3 m²/(V·s) | `20` cm²/(V·s) |
| 空穴迁移率 µp | 2.0e-3 m²/(V·s) | `20` cm²/(V·s) |
| 电子亲和能 χ | 4.0 eV | `4.0` eV |
| 带隙 Eg | 0.8 eV | `0.8` eV |
| 导带有效态密度 Nc | 1.0e24 m⁻³ | `1.0e18` cm⁻³ |
| 价带有效态密度 Nv | 8.0e23 m⁻³ | `8.0e17` cm⁻³ |
| 吸收系数 α | 4.0e5 m⁻¹ | `4.0e3` cm⁻¹ |
| 辐射复合系数 B | 0.0 | `0` |
| 俄歇系数 Cn / Cp | 0.0 / 0.0 | `0` / `0` |

单层器件，只有一个半导体层。

### 3.2 三个场景的差异

| | M1 | M2 | M3 |
|---|---|---|---|
| 层名 | `m1_double_donor_p_slab` | `m2_double_acceptor_n_slab` | `m3_amphoteric_slab` |
| 缺陷 family | **double donor** | **double acceptor** | **amphoteric** |
| 电荷态（正→负） | `+2, +1, 0` | `0, −1, −2` | `+1, 0, −1` |
| 浅施主 Nd | `0` | `5.0e15` cm⁻³ | `0` |
| 浅受主 Na | `5.0e15` cm⁻³ | `0` | `0` |
| 掺杂极性 | p 型 | n 型 | intrinsic |
| 简并度约定 | `scaps_binomial` | `scaps_binomial` | **`unity`** |
| 态简并度 g | `[1, 2, 1]` | `[1, 2, 1]` | **`[1, 1, 1]`** |
| config SHA-256 | `43516a15…407290` | `32c78286…27d223` | `7e772e2e…86915c` |

**M3 的简并度是故意与 M1/M2 不同的** —— 它检验的正是简并度约定本身有没有被
正确传播。已对照手册确认（SCAPSManual2016 §3.6.2.3）：SCAPS **恰好只提供**这
两种约定 —— 默认是二项式 `g_s = C(H, s)`（手册 Eq. 9，H = 电荷态总数），另有一个
checkbox 可把全部简并度**设为 1**；不存在自由逐态设定。因此：

- **M1 / M2**：保持默认（checkbox 不勾）→ `g = [1, 2, 1]`；
- **M3**：勾选「set to one」→ `g = [1, 1, 1]`；
- 把 checkbox 的实际状态记进 manifest 的 `degeneracy_option`。

原「若 SCAPS 不暴露此项则停下来报回」的条款已解除 —— 手册确认暴露，且恰好是
仓库预设的两个选项。SCAPS 面板上缺陷类型按钮的对应名称：`double donor` /
`double acceptor` / `amphoteric`（script 名 `layer.defect.doubledonor` /
`.doubleacceptor` / `.amphoteric`）。电荷态输入顺序 SCAPS 强制**最正在前**
（"level 1 the most positive charge"），与 §3.2 表的顺序一致。

### 3.3 缺陷能级阶梯（三个场景共用）

仓库用「首级 + 相关能」编码：`first_transition_eV_above_vb: 0.30`，
`correlation_energies_eV: [0.15]`，即 `E_t,s = E_t,s−1 + U_s`。展开后：

| 跃迁 | 电荷态对 M1 | 电荷态对 M2 | 电荷态对 M3 | **能级（above E_V）** |
|---|---|---|---|---|
| 第一跃迁 | (+2 / +1) | (0 / −1) | (+1 / 0) | **`0.30` eV** |
| 第二跃迁 | (+1 / 0) | (−1 / −2) | (0 / −1) | **`0.45` eV** |

两级都在带隙内（`0 < 0.30 < 0.45 < 0.8`）。相关能为正 → 第二跃迁**离价带更远**。

**这一项最容易错**，两个方向都要确认：

1. **参考点方向** —— 0.30 / 0.45 是从**价带顶往上**量。已对照手册（§3.6.3）：
   SCAPS 提供三个参考选项 `above EV` / `below EC` / `above Ei`，**请直接选
   `above EV`** 并填 `0.30` / `0.45`；若坚持用 `below EC` 则填 `0.50` / `0.35` eV。
   另外多价面板有「Use Correlation Energy」checkbox（§3.6.2.3）：勾选后第二级
   以**相对第一级的能量差**输入 —— 即可直接填仓库的 `U = 0.15 eV`；用哪种输入
   方式都行，但要在 manifest 里记录。
2. **哪一级配哪一对电荷态** —— SCAPS 的两个能级输入框哪个对应 (+2/+1)、哪个对应
   (+1/0)，请**如实记录在 manifest 里**，不要凭直觉假定。这是本组场景最可能出现
   静默错配的地方：填反了器件仍能收敛，剖面却是另一个物理系统的。

### 3.4 逐跃迁俘获参数

两个跃迁的截面**不同**，不能只填一组：

| 量 | 第一跃迁（0.30 eV） | 第二跃迁（0.45 eV） |
|---|---|---|
| 电子俘获截面 σn | `2.0e-15` cm²（SI `2.0e-19` m²） | `1.0e-15` cm²（SI `1.0e-19` m²） |
| 空穴俘获截面 σp | `7.0e-16` cm²（SI `7.0e-20` m²） | `5.0e-16` cm²（SI `5.0e-20` m²） |
| 电子热速度 vth,n | `1.0e7` cm/s | `1.0e7` cm/s |
| 空穴热速度 vth,p | `8.0e6` cm/s | `8.0e6` cm/s |

缺陷总密度 **Nt = `2.0e15` cm⁻³**（SI `2.0e21` m⁻³），三态**共享一个总量**：
`ΣP_s = 1`，`N_s = Nt·P_s`。不是三个各自 `2.0e15` 的独立缺陷。

已对照手册确认（§3.6.2.3）：多价面板上**逐级双击**即可分别编辑每级的
`capture cross section`（electrons / holes，cm²）—— 逐跃迁截面是暴露的，
原「若只接受一组则停下来报回」的条款解除。注意**热速度是层级参数**
（`layer.vthn` / `layer.vthp`，cm/s），不随级区分 —— 与本表一致（两行同值），
在层面板填一次即可。importer 会硬性拒绝两组截面完全相同的 manifest
（防「同一组填两遍」）。

### 3.5 接触

与 S0–S2 同：`built_in_potential_mode: semiconductor_work_function` + `phi_left: 0.0`
→ **内建电势为零，前后接触都是平带**。

- 前接触：**flat band**；后接触：**flat band**（SCAPS 接触面板的「flat bands」
  选项，script 名 `contact.flatband`）
- 表面复合速度 Sn / Sp：平衡态下不影响解，用默认值，但**必须在 manifest 记录**
- 前/后反射率：暗态无关，同样记录

⚠ 手册（§3.3）注明：**2014-01-01 之前**的 SCAPS 版本用只看浅掺杂的简化公式算
平带金属功函数 —— 缺陷带电时（M1/M2 正是）它算出的 Φm 不真产生平带；
**2014 之后**的版本解包含深缺陷电荷的完整电中性方程。请用 ≥2014 的版本并把
精确版本串记入 manifest。本组场景两端接同一均匀层，两端 Φm 算法相同 →
内建电势恒为零，这点不受版本影响；受影响的是接触附近的能带弯曲。

### 3.6 **不要输入**的量（派生量）

与 S0–S2 同样的五个。它们由 §3.1 / §3.4 的基元反算，SCAPS 会自己算：

| 量 | 仓库值 | 对应什么 |
|---|---|---|
| `ni` | 1.7054568918e17 m⁻³ | `sqrt(Nc·Nv)·exp(−Eg/2kT)`，与 S0–S2 同 |
| `tau_n` | 2.5e-8 s | `1/(σn·vth,n·Nt)`，**第一跃迁**的截面 |
| `tau_p` | 8.9285714286e-8 s | `1/(σp·vth,p·Nt)`，**第一跃迁**的截面 |
| `n1` | 1.2950995328e17 m⁻³ | `Nc·exp(−(Eg−Et)/kT)` **在 Et = 0.39 eV** |
| `p1` | 2.2458375871e17 m⁻³ | `Nv·exp(−Et/kT)` **在 Et = 0.39 eV** |

**注意 `n1`/`p1` 的 0.39 eV 既不是第一跃迁（0.30）也不是第二跃迁（0.45）** ——
它是 S0–S2 的单能级位置，逐字节抄过来的遗留字段。实测三个候选值：

```text
Et=0.30  n1=3.984463e+15  p1=7.299815e+18
Et=0.39  n1=1.295100e+17  p1=2.245838e+17   ← M config 携带的值
Et=0.45  n1=1.319043e+18  p1=2.205072e+16
```

**这不要求你在 SCAPS 里额外加一个 0.39 eV 的单能级缺陷。** 实测：把 M1 的
`tau_n`/`tau_p` 改成 `1e30`（等于关掉这条基于 tau 的体 SRH）后，QF/DC lane 解出的
`phi` / 两个准费米势 / 两个净速率**逐比特相同**（SHA-256 `1e08dbead2bdc325…`）。
多价层上这条通道是惰性的 —— 层级 `tau`/`n1`/`p1` 不参与求解。

所以：**这五个量一个都不要输入**，SCAPS 里也不要为它们建任何额外缺陷。

---

## 4. 工作点

与 S0–S2 同：**暗态、零偏压、平衡态**，跑到收敛。

**网格**：手册 §5.1.1 明确指出静态网格对多价缺陷可能不够，建议在 numerical
panel 打开 **recalculate mesh**（手册 Fig. 5.2/5.3 正是用 amphoteric 缺陷占据
演示的）。开或不开由你，但 manifest 的 `numerics.recalculate_mesh` **必须记录
实际状态** —— importer 缺这个键直接拒收。

> 平衡态下净复合率按细致平衡应当 ≈ 0，`recombination_rate_cm3_s` 一列预期近似
> 全零，信息量很低。多价场景真正承载信息的是 **`charge_state_occupation_fraction_per_state`**
> 与 **`defect_charge_number_cm3`** 两列 —— 三态如何在一个总量上分配，正是
> D7-E2 唯一要问的问题。

---

## 5. 导出

### 5.1 CSV 列名

**注意与 S0–S2 不同**：`defect_occupancy` 被删除，末尾追加
`charge_state_occupation_fraction_per_state`，且列序也变了。仍是 9 列。

```text
position_um,electron_density_cm3,hole_density_cm3,electrostatic_potential_V,conduction_band_eV,valence_band_eV,defect_charge_number_cm3,recombination_rate_cm3_s,charge_state_occupation_fraction_per_state
```

模板：`docs/scaps-collection-templates/M{1,2,3}_profile_template.csv`
（仅表头，逐字符即契约；importer 连**列序**都校验）。

### 5.2 每列取自 SCAPS 的哪个量

| 列 | 单位 | 说明 |
|---|---|---|
| `position_um` | µm | 原点在**左接触** |
| `electron_density_cm3` | cm⁻³ | n(x)，> 0 |
| `hole_density_cm3` | cm⁻³ | p(x)，> 0 |
| `electrostatic_potential_V` | V | 零点在 manifest 显式声明 |
| `conduction_band_eV` | eV | E_C(x)，> E_V(x) |
| `valence_band_eV` | eV | E_V(x) |
| `defect_charge_number_cm3` | cm⁻³ | 见 §5.3 |
| `recombination_rate_cm3_s` | cm⁻³s⁻¹ | 正值 = 净复合 |
| `charge_state_occupation_fraction_per_state` | 1 | 见 §5.4，编码已定案：`P₀\|P₁\|P₂` |

位置容差沿用 S0–S2：首行 `0.000000`、末行 `0.300000`。网格端点不落在接触上时
**不要插值补齐** —— 报回来由我方调整协议。

### 5.3 `defect_charge_number_cm3`

多价缺陷的净电荷是三态按占据分数加权求和：

```text
ρ_def(x) = Nt · Σ_s  q_s · P_s(x)          Nt = 2.0e15 cm⁻³
```

代入各场景的电荷态：

| 场景 | q = (q₀, q₁, q₂) | ρ_def(x) / Nt | 预期符号 |
|---|---|---|---|
| M1 | (+2, +1, 0) | `2·P₀ + 1·P₁ + 0·P₂` | 非负 |
| M2 | (0, −1, −2) | `0·P₀ − 1·P₁ − 2·P₂` | 非正 |
| M3 | (+1, 0, −1) | `1·P₀ + 0·P₁ − 1·P₂` | **可正可负**，随位置换号 |

M3 允许换号是它的物理内容，不是错误 —— 所以多价**不能**照搬 S0–S2 那套
「非正 / 非负」的硬校验。这也是多价 importer 需要单独写、而不是复用现有那个的原因之一。

若 SCAPS 直接导出缺陷净电荷密度，**优先用它导出的值**，不要自己按上式乘。
importer 会逐行校验它与 `Nt·Σ q_s·P_s` 的一致性（容差 `1e-3·Nt`，manifest 可调、
上限 1% —— 见协议文档）；不一致会被**拒收并原样报错**：

```text
M1 row 2: exported net defect charge ... is inconsistent with the state
fractions (derived ...); report this back instead of editing
```

那是一个发现，如实交回，不要改数字凑一致。

### 5.4 `charge_state_occupation_fraction_per_state`：编码已定案（原提案 A）

单元格内放三个占据分数，`|` 分隔，**最正态在前**：

```text
0.812|0.171|0.017        ← P(最正), P(中间), P(最负)
```

- 顺序 = §3.2 电荷态行从左到右（M1: +2,+1,0；M2: 0,−1,−2；M3: +1,0,−1），
  也 = SCAPS 多价面板的级序（level 1 最正）；
- 每个分数 ∈ [0, 1]；三值之和 = 1，容差由 manifest 的
  `occupation_fraction_sum_tolerance` 声明（skeleton 预填 `1e-4`，上限 `1e-3`）；
- 分隔符必须是 `|`（CSV 安全；manifest 里 `occupation_fraction_separator`
  锁定为它）。

**为什么不拆三列（原提案 B）**：suite JSON 的 `raw_profile_columns` 被
`reproducibility/numerical_refinement_registry.yaml` 按字节哈希钉死，且整个
`external_reference_contract` 块被嵌进 D7 认证 lane 的 protocol hash —— 改列声明
会作废已认证的 refinement 输出。单列编码让 suite 一个字节不动。

**从 SCAPS 哪里取三个分数**：EB-panel 的 occupation 图（手册 §6.4.1）可按
「charge of the states」着色，直接显示**各电荷态的分数** —— 但该模式**不显示
中性态**（q=0）。中性态分数 = `1 − 其余两态之和`，或从默认的逐级
「occupation with electrons」曲线换算。无论走哪条路：把 SCAPS 的**原始未裁剪
导出**整份保留一同交回（每个场景各自的），并用 Curve Info 确认每条曲线的
确切含义再对号入座。

importer 对本列的实测拒绝分支（逐字）：

```text
两个分数        -> M1 row 2: state fraction cell must hold exactly three
                   '|'-separated fractions (most positive state first)
和不为 1        -> M1 row 2: state fractions must sum to 1 within 0.0001
越界            -> M1 row 2: state fractions must lie in [0, 1]
```

---

## 6. 交付前自查

逐条打勾，每条都对应 importer 里一个真实的拒绝分支：

- [ ] 三个 `.def` 都在，且是 SCAPS 保存的原文件
- [ ] 三个 CSV 表头逐字符等于 §5.1 那一行，**顺序也对**
- [ ] 每个 CSV 至少 3 行数据
- [ ] `position_um` 严格递增；首行 `0.000000`、末行 `0.300000`
- [ ] `electron_density_cm3` 与 `hole_density_cm3` 严格为正
- [ ] 每行 `conduction_band_eV > valence_band_eV`
- [ ] 无 NaN / Inf / 空单元格
- [ ] M1 的 `defect_charge_number_cm3` 无负值且非全零；M2 无正值且非全零；
      **M3 不设此约束**（换号是物理内容）
- [ ] 每行三个占据分数：`|` 分隔、最正态在前、各 ∈ [0,1]、和 = 1
      （容差 = manifest 的 `occupation_fraction_sum_tolerance`，skeleton 预填 1e-4）
- [ ] 每行净电荷与 `Nt·Σ q_s·P_s` 一致（容差 1e-3·Nt）
- [ ] 两个跃迁的截面**分别**填了 §3.4 的两组，不是同一组填两遍
- [ ] 能级参考方向已确认（`above EV`），且「哪一级配哪一对电荷态」已写进
      manifest 的 `level_to_charge_pair_mapping`
- [ ] 简并度：M1/M2 = binomial（默认）、M3 = unity（勾「set to one」），
      checkbox 状态已记入 `degeneracy_option`
- [ ] `numerics.recalculate_mesh` 已记录实际状态
- [ ] SCAPS 的**原始未裁剪导出**已一并保留（见 §5.4）

以上每一条都有 importer 硬校验兜底。关键拒绝分支实测消息（逐字）：

```text
列序换位        -> ValueError: M2 CSV columns must exactly match the M1-M3 contract
只有 2 行数据   -> ValueError: M3 profile requires at least three direct rows
末行 != 厚度    -> ValueError: M1 profile thickness mismatch
M1 出现负电荷   -> ValueError: M1 double_donor net defect charge must be
                   nonnegative and nonzero
M2 出现正电荷   -> ValueError: M2 double_acceptor net defect charge must be
                   nonpositive and nonzero
M3 用 binomial  -> ValueError: M3 degeneracy convention must be 'unity'
两组截面相同    -> ValueError: M1 transition capture cross sections must differ
                   between transitions; one set entered twice is the classic
                   silent operator error
config 哈希漂移 -> ValueError: parameter scenario M2 config hash mismatch
```

（M3 净电荷随位置换号的样例被**接受** —— 也已实测。）

---

## 7. 这份数据打开什么

importer 已就位，交回即可落盘并解锁：`D7-E2` 的外部半边 —— 多价缺陷的 SCAPS
charge-state / recombination profile 比对。

**不能**解锁的，与 S0–S2 交付单所列相同：`D9.6`（需要与标定数据不相交的第三方
deck）、`D8-E3`（三重阻塞，缺数据只是其中一重）。

---

## 8. 派发前我方待办 —— 已全部完成（2026-09-02）

1. ✅ 编码定案：单列 `|` 分隔（原提案 A，理由见 §5.4）。suite JSON 与测试
   **零改动**（byte-hash 钉死的冻结物保持不动）。
2. ✅ schema `solarlab.scaps_multivalent_defect_parameter_manifest` 已定义
   （`docs/scaps-defect-m1-m3-reference-protocol.md`）。
3. ✅ `scripts/import_scaps_multivalent_defect_reference.py` 已实现，含按
   family 的符号规则与「和为 1」「净电荷一致性」校验。
   测试：`tests/unit/validation/test_import_scaps_multivalent_defect_reference.py`。
4. ✅ 协议文档已补，与 S0–S2 对齐。
5. ✅ 本单 §0 / §5.4 / §6 已回填实测拒绝消息；CSV 模板与 manifest skeleton 已就位。

交回数据后的导入命令：

```bash
python scripts/import_scaps_multivalent_defect_reference.py \
  --project-root . \
  --suite reproducibility/scaps_multivalent_defect_suite.json \
  --parameter-manifest /path/to/parameter_manifest.json \
  --m1-csv /path/to/M1.csv --m1-source-deck /path/to/M1.def \
  --m2-csv /path/to/M2.csv --m2-source-deck /path/to/M2.def \
  --m3-csv /path/to/M3.csv --m3-source-deck /path/to/M3.def \
  --solver-version <精确版本串> \
  --extracted-at <ISO 日期> \
  --operator "<操作者姓名>" \
  --confirm-independent-scaps-export \
  --confirm-direct-unmodified-rows \
  --out reproducibility/scaps_defect_m1_m3_reference.json
```
