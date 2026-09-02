# SCAPS-1D M1–M3 多价缺陷 参数录入与导出交付单

面向执行 SCAPS 仿真的操作者。配套契约：`docs/multivalent-metastable-defect-contract.md`。
冻结 suite：`reproducibility/scaps_multivalent_defect_suite.json`。

与 S0–S2 交付单（`docs/scaps-s0-s2-collection-sheet.md`）结构平行；两份可对照阅读。

---

## ⚠ 0. 先看这一条：本单尚不可直接派发

S0–S2 那一套有 importer（`scripts/import_scaps_defect_reference.py`），交回的数据
当天就能落盘。**M1–M3 没有。** 实测确认：

- `scripts/` 下只有 `import_scaps_defect_reference.py` 与 `import_scaps_cbo_reference.py`，
  **不存在多价 importer**；
- `charge_state_occupation_fraction_per_state` 这个列名在全仓库只出现三处 ——
  suite JSON 里声明它、一个测试断言它被声明、以及本文档。**没有任何代码消费它**；
- 没有多价版的 parameter manifest schema
  （S0–S2 有 `solarlab.scaps_explicit_defect_parameter_manifest`，多价无对应物）；
- 没有按 family 的符号/标签校验规则
  （S0–S2 有「S0 恒零 / S1 非正 / S2 非负」，多价无对应物）。

因此本单中**两处标记为「待定案」的内容不是契约**，是提案。派发给合作方之前必须先
写出 importer 把它们定案，否则采回的数据无法验收 —— 而多价数据的采集成本比 S0–S2 高。

其余部分（物理参数、能级阶梯、简并度、config 哈希、导出工作点）**全部已在仓库中冻结**，
可以直接依赖。

---

## 1. 交付物清单

三个场景 M1 / M2 / M3，每个交三样，共十件：

| # | 文件 | 说明 |
|---|---|---|
| 1–3 | `M1.def` `M2.def` `M3.def` | SCAPS 输入定义文件，原样保存 |
| 4–6 | `M1.csv` `M2.csv` `M3.csv` | 逐位置剖面，列名见 §5.1 |
| 7 | `parameter_manifest.json` | **schema 待定案**，见 §0 |
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
正确传播。若 SCAPS 不允许自由设定态简并度（只提供 binomial / unity 两个选项），
M3 请选 **unity**，M1/M2 请选 **binomial**，并在 manifest 里注明选项名称。
若 SCAPS 根本不暴露此项，**停下来报回** —— 那会使 M3 失去意义，需要重新设计场景。

### 3.3 缺陷能级阶梯（三个场景共用）

仓库用「首级 + 相关能」编码：`first_transition_eV_above_vb: 0.30`，
`correlation_energies_eV: [0.15]`，即 `E_t,s = E_t,s−1 + U_s`。展开后：

| 跃迁 | 电荷态对 M1 | 电荷态对 M2 | 电荷态对 M3 | **能级（above E_V）** |
|---|---|---|---|---|
| 第一跃迁 | (+2 / +1) | (0 / −1) | (+1 / 0) | **`0.30` eV** |
| 第二跃迁 | (+1 / 0) | (−1 / −2) | (0 / −1) | **`0.45` eV** |

两级都在带隙内（`0 < 0.30 < 0.45 < 0.8`）。相关能为正 → 第二跃迁**离价带更远**。

**这一项最容易错**，两个方向都要确认：

1. **参考点方向** —— 0.30 / 0.45 是从**价带顶往上**量。SCAPS 面板若默认
   「below E_C」，请填 `0.50` 和 `0.35` eV。
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

若 SCAPS 只接受一组截面而不区分跃迁，**停下来报回**，不要用平均值凑。

### 3.5 接触

与 S0–S2 同：`built_in_potential_mode: semiconductor_work_function` + `phi_left: 0.0`
→ **内建电势为零，前后接触都是平带**。

- 前接触：**flat band**；后接触：**flat band**
- 表面复合速度 Sn / Sp：平衡态下不影响解，用默认值，但**必须在 manifest 记录**
- 前/后反射率：暗态无关，同样记录

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

`docs/scaps-collection-templates/` 下**故意没有** M1–M3 的 CSV 模板：末列编码尚未
定案（§5.4），提供模板等于诱导按错误格式提前导出。定案后补。

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
| `charge_state_occupation_fraction_per_state` | 1 | 见 §5.4，**编码待定案** |

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
两者不一致就如实交回并注明 —— 那是一个发现。

### 5.4 ⚠ `charge_state_occupation_fraction_per_state`：编码待定案

列名说「per state」，但一个 CSV 单元格只能放一个值，而这里每个位置有 **3 个**
占据分数 `(P₀, P₁, P₂)`（对应 §3.2 电荷态行的从左到右顺序，即最正 → 最负）。
仓库**没有规定**它们怎么塞进这一列。

**在 importer 写出来之前，不要按任何一种猜测去导出。**

以下是提案，供定案时参考，**目前不是契约**：

- **提案 A**（改动最小）：单元格内用 `|` 分隔，如 `0.812|0.171|0.017`；
  行内三值之和须为 1。
- **提案 B**（最不易错，但要改 suite 的列声明）：拆成三列
  `charge_state_occupation_fraction_s0/_s1/_s2`。

无论哪种，都要额外定义：状态顺序、和为 1 的容差、以及三个 family 各自的物理约束。

**当前建议**：先做 §3、§4、§5.1–5.3 的所有工作并把 SCAPS 的原始导出**整份保留**
（不要裁剪成 9 列），等编码定案后再从原始导出重排成最终 CSV。这样不会白跑。

---

## 6. 交付前自查

可以现在就查的（与 S0–S2 同源）：

- [ ] 三个 `.def` 都在，且是 SCAPS 保存的原文件
- [ ] 三个 CSV 表头逐字符等于 §5.1 那一行，**顺序也对**
- [ ] 每个 CSV 至少 3 行数据
- [ ] `position_um` 严格递增；首行 `0.000000`、末行 `0.300000`
- [ ] `electron_density_cm3` 与 `hole_density_cm3` 严格为正
- [ ] 每行 `conduction_band_eV > valence_band_eV`
- [ ] 无 NaN / Inf / 空单元格
- [ ] M1 的 `defect_charge_number_cm3` 无负值；M2 无正值；**M3 不设此约束**
- [ ] 每行三个占据分数之和 = 1（容差待定案）
- [ ] 两个跃迁的截面**分别**填了 §3.4 的两组，不是同一组填两遍
- [ ] 能级参考方向已确认，且「哪一级配哪一对电荷态」已写进 manifest
- [ ] 简并度：M1/M2 = binomial，M3 = unity，选项名称已记录
- [ ] SCAPS 的**原始未裁剪导出**已一并保留（见 §5.4）

尚不能查的（依赖 §0 的定案）：manifest schema 校验、per-state 列编码校验、
按 family 的符号规则。

---

## 7. 这份数据打开什么

交回并完成 importer 后可解锁：`D7-E2` 的外部半边 —— 多价缺陷的 SCAPS
charge-state / recombination profile 比对。

**不能**解锁的，与 S0–S2 交付单所列相同：`D9.6`（需要与标定数据不相交的第三方
deck）、`D8-E3`（三重阻塞，缺数据只是其中一重）。

---

## 8. 派发前我方待办

按依赖顺序：

1. 定案 `charge_state_occupation_fraction_per_state` 的编码（§5.4），必要时同步
   修订 `reproducibility/scaps_multivalent_defect_suite.json` 的
   `raw_profile_columns` 与那条断言它的测试。
2. 定义 `solarlab.scaps_multivalent_defect_parameter_manifest` schema，
   含逐跃迁能级/截面槽位与简并度约定槽位。
3. 写 `scripts/import_scaps_multivalent_defect_reference.py`，含按 family 的
   符号规则（M1 非负 / M2 非正 / M3 无约束）与「和为 1」校验。
4. 补 `docs/scaps-defect-m1-m3-reference-protocol.md`，与 S0–S2 的协议文档对齐。
5. 回填本单的 §0、§5.4 与 §6，把「待定案」替换为实测的拒绝分支消息。

以上 5 项全部在仓库内部，不依赖外部输入。
