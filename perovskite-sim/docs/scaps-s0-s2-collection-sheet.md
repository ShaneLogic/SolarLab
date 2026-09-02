# SCAPS-1D S0–S2 参数录入与导出交付单

面向执行 SCAPS 仿真的操作者。配套契约：`docs/scaps-defect-s0-s2-reference-protocol.md`。
冻结 suite：`reproducibility/scaps_defect_s0_s2_suite.json`。
导入器：`scripts/import_scaps_defect_reference.py`。

模板文件在 `docs/scaps-collection-templates/`：
`S0_profile_template.csv`、`S1_profile_template.csv`、`S2_profile_template.csv`、
`parameter_manifest_skeleton.json`。

---

## 0. 一句话

参数**已经定死**，不需要你选，也不许你改。你要做的是：把下面的参数逐项抄进 SCAPS，
在暗态零偏压下跑到收敛，把**逐位置剖面**原样导出，连同 `.def` 和 manifest 交回。

真正被采集的是 SCAPS 的**输出**，不是输入。

---

## 1. 交付物清单

三个场景 S0 / S1 / S2，每个交三样，共十件：

| # | 文件 | 说明 |
|---|---|---|
| 1–3 | `S0.def` `S1.def` `S2.def` | SCAPS 的输入定义文件，原样保存 |
| 4–6 | `S0.csv` `S1.csv` `S2.csv` | 逐位置剖面，列名见 §5.1 |
| 7 | `parameter_manifest.json` | 由 `parameter_manifest_skeleton.json` 填空得到 |
| 8 | SCAPS 版本号 | 精确版本串，例 `3.3.10` |
| 9 | 操作者真实姓名 | 用于署名声明，不能匿名 |
| 10 | 导出日期 | ISO 格式，例 `2026-09-02` |

---

## 2. 红线

违反任何一条，导入器会**直接拒绝**，不会给你一个「差不多」的结果。

- **不许插值、拟合、重采样、平滑**。CSV 里的行必须是 SCAPS 直接吐出来的那些行，
  一行不多一行不少。
- **不许改数值**。允许改的只有表头一行（改成 §5.1 的列名）。
- **不许换单位**之外的加工。单位换算见 §5.2，只做那一种。
- **不许改 §3 的任何参数**。改了 `canonical_config_sha256` 就对不上。
- **不许用非 SCAPS-1D 的求解器**。契约写死 `required_solver: "SCAPS-1D"`。

---

## 3. 参数录入表

### 3.1 三个场景的共用底表

`configs/scaps_defect_s0_neutral.yaml` 等三个 YAML 是权威来源；下表是它们换算到
SCAPS 常用的 cm 制之后的形式。左列是仓库里的 SI 值，右列是你要输进 SCAPS 的值。

| 量 | 仓库值（SI） | **输进 SCAPS 的值** |
|---|---|---|
| 温度 T | 300.0 K | `300` K |
| 层厚 | 3.0e-7 m | `0.300` µm（= 300 nm） |
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
| 电子热速度 vth,n | 1.0e5 m/s | `1.0e7` cm/s |
| 空穴热速度 vth,p | 8.0e4 m/s | `8.0e6` cm/s |

单层器件，只有一个半导体层。

### 3.2 三个场景的差异

| | S0 | S1 | S2 |
|---|---|---|---|
| 层名 | `s0_neutral_slab` | `s1_acceptor_n_slab` | `s2_donor_p_slab` |
| 浅施主 Nd | `0` | `5.0e15` cm⁻³ | `0` |
| 浅受主 Na | `0` | `0` | `5.0e15` cm⁻³ |
| 缺陷类型 | **neutral** | **single acceptor (A/−)** | **single donor (D/+)** |
| 掺杂极性 | intrinsic | n 型 | p 型 |
| config SHA-256 | `77b89e16…5025c7` | `f49900b2…bcb681` | `7fff245d…3d8066` |

其余全部相同。

### 3.3 缺陷块（三个场景共用，只有「类型」不同）

| 量 | 仓库值（SI） | **输进 SCAPS 的值** |
|---|---|---|
| 能级分布 | `single_level` | 单能级（不是 Gaussian / uniform / band tail） |
| 能级位置 Et | 0.39 eV above VB | `0.39` eV，参考点选 **above E_V** |
| 总密度 Nt | 2.0e21 m⁻³ | `2.0e15` cm⁻³（`integrated_total`，即总量非峰值） |
| 电子俘获截面 σn | 2.0e-19 m² | `2.0e-15` cm² |
| 空穴俘获截面 σp | 7.0e-20 m² | `7.0e-16` cm² |
| 简并度 | 1.0 | `1`（SCAPS 若不暴露此项，跳过并在 manifest 的 `numerics` 里注明） |

**能级参考点选错方向是最容易犯的错**：0.39 eV 是从**价带顶**往上量。若 SCAPS
面板默认「below E_C」，请填 `0.8 − 0.39 = 0.41` eV，并在 manifest 里注明你用了哪一种。

### 3.4 接触

仓库 config 是 `built_in_potential_mode: semiconductor_work_function` 且 `phi_left: 0.0`
—— 即**内建电势为零，两侧接触都是平带**。

- 前接触：**flat band**
- 后接触：**flat band**
- 表面复合速度 Sn / Sp：平衡态下不影响解，用 SCAPS 默认值即可，但**必须在
  manifest 的 `numerics` 里如实记录你用的值**。
- 前/后反射率：暗态无关，同样记录。

### 3.5 **不要输入**的量（派生量）

下面五个量出现在仓库 YAML 里，但它们**不是独立输入** —— 是从 §3.1/§3.3 的基元
反算出来的。SCAPS 会自己算。你手动输入会引入不一致。

| 量 | 仓库值 | 由什么导出 |
|---|---|---|
| `ni` | 1.7054568918e17 m⁻³ | `sqrt(Nc·Nv)·exp(−Eg/2kT)` |
| `n1` | 1.2950995328e17 m⁻³ | `Nc·exp(−(Eg−Et)/kT)` |
| `p1` | 2.2458375871e17 m⁻³ | `Nv·exp(−Et/kT)` |
| `tau_n` | 2.5e-8 s | `1/(σn·vth,n·Nt)` |
| `tau_p` | 8.9285714286e-8 s | `1/(σp·vth,p·Nt)` |

（已用 `kT = 0.025852 eV` 逐项复核，相对误差 < 1e-6。）

若 SCAPS 报出的 `ni` 与上表差异显著，**不要去调参数迁就它** —— 那是一个需要如实
记录的发现，请连同 SCAPS 报出的值一起交回。

---

## 4. 工作点

契约写死为 `dark_equilibrium_zero_bias`：

- **暗态**（无光照）
- **零偏压**
- **平衡态**（不是 J-V 扫描的某一点，不是 C-V，不是 C-f）

跑到收敛，收敛判据填进 manifest 的 `numerics`。

> 注意：平衡态下净复合率按细致平衡应当 ≈ 0，所以 `recombination_rate_cm3_s`
> 一列预期近似全零。这一列在本工作点信息量很低，保留它是为了交叉校验，不是主要
> 比对对象。真正承载信息的是电势剖面、能带边、以及缺陷占据与缺陷电荷两列。

---

## 5. 导出

### 5.1 CSV 列名

**逐字符一致，顺序也一致**（实测：交换任意两列位置即被拒）：

```text
position_um,electron_density_cm3,hole_density_cm3,electrostatic_potential_V,conduction_band_eV,valence_band_eV,defect_occupancy,defect_charge_number_cm3,recombination_rate_cm3_s
```

直接用 `docs/scaps-collection-templates/S{0,1,2}_profile_template.csv`（已含表头），
把 SCAPS 的数据行追加在后面。

### 5.2 每列取自 SCAPS 的哪个量

| 列 | 单位 | 来源 | 换算 |
|---|---|---|---|
| `position_um` | µm | 位置坐标 | 原点在**左接触**（`position_origin: left_contact`） |
| `electron_density_cm3` | cm⁻³ | n(x) | 必须 > 0 |
| `hole_density_cm3` | cm⁻³ | p(x) | 必须 > 0 |
| `electrostatic_potential_V` | V | φ(x) | 零点在 manifest 里显式声明 |
| `conduction_band_eV` | eV | E_C(x) | 必须 > `valence_band_eV` |
| `valence_band_eV` | eV | E_V(x) | |
| `defect_occupancy` | 1 | 缺陷能级的**电子**占据分数 f | 必须落在 `[0,1]` |
| `defect_charge_number_cm3` | cm⁻³ | 见 §5.3 | 符号约定见下 |
| `recombination_rate_cm3_s` | cm⁻³s⁻¹ | 净复合率 | 正值 = 净复合 |

**位置容差极紧**：`position_tolerance_um` 上限 `1e-6` µm = **1 皮米**。首行必须
是 `0.000000`，末行必须是 `0.300000`。SCAPS 若导出的网格端点不落在接触上，
**不要插值补齐** —— 报告这个情况，由我方调整协议。

### 5.3 `defect_charge_number_cm3` 怎么算

符号约定固定为 `negative_acceptor_positive_donor`（受主为负，施主为正）。
每个场景的 `neutral_reference` 不同，映射也不同：

| 场景 | 缺陷 | 中性参考 | ρ_def(x) |
|---|---|---|---|
| S0 | neutral | `all_occupancies` | **恒等于 0**，与 f 无关 |
| S1 | acceptor | `empty`（空时中性，被占据时带负电） | `−Nt · f(x)` |
| S2 | donor | `filled`（占据时中性，空时带正电） | `+Nt · (1 − f(x))` |

其中 `Nt = 2.0e15 cm⁻³`。

若 SCAPS 直接导出缺陷电荷密度，**优先用它导出的值**，不要自己乘。若两者不一致，
如实交回并注明 —— 那也是一个发现。

导入器会硬性检查（实测消息如实引用）：

- S0 任一行非零 → `ValueError: S0 neutral defect charge must be exactly zero`
- S1 任一行为正、或全为零 → `ValueError: S1 acceptor defect charge must be nonpositive and nonzero`
- S2 任一行为负、或全为零 → `ValueError: S2 donor defect charge must be nonnegative and nonzero`

---

## 6. manifest 填空

拷贝 `docs/scaps-collection-templates/parameter_manifest_skeleton.json`，
把所有 `__FILL__` 替换掉。骨架已实测可通过导入器。

必须自己填的：

| 位置 | 填什么 |
|---|---|
| `solver.version` | 精确 SCAPS 版本串，必须与命令行 `--solver-version` 完全一致 |
| `numerics.*` | 网格点数、网格模式、收敛判据、最大迭代次数、接触表面复合速度、前后反射率 |
| `sign_conventions.electrostatic_potential` | 显式说明 SCAPS 电势零点在哪 |

**不要动**的（改了必被拒）：`schema`、`schema_version`、`unit_conventions` 整块、
`sign_conventions` 的另外三项、`comparison_protocol` 整块、每个场景的
`canonical_config_sha256` / `charge_transition` / `doping_polarity`。

`scaps_parameters` 已按 §3 预填。**逐项与你实际输进 SCAPS 的值核对**；若某项
SCAPS 界面里叫别的名字或不接受该值，改成实际值并在 `numerics` 里说明原因。
这一块的作用是「记录你实际输了什么」，不是「记录应该输什么」。

顶层键必须**恰好**是这八个，多一个少一个都拒：

```text
comparison_protocol, numerics, scenarios, schema, schema_version,
sign_conventions, solver, unit_conventions
```

---

## 7. 交付前自查

逐条打勾，能省一轮返工。以下每条都对应导入器里一个真实的拒绝分支。

- [ ] 三个 `.def` 都在，且是 SCAPS 保存的原文件
- [ ] 三个 CSV 表头逐字符等于 §5.1 那一行，**顺序也对**
- [ ] 每个 CSV 至少 3 行数据（不含表头）
- [ ] `position_um` 严格递增，无重复、无倒序
- [ ] 首行 `0.000000`，末行 `0.300000`（容差 1e-6 µm）
- [ ] 所有 `electron_density_cm3` 和 `hole_density_cm3` 严格为正
- [ ] 所有 `defect_occupancy` 落在 `[0, 1]`
- [ ] 每行 `conduction_band_eV > valence_band_eV`
- [ ] S0 的 `defect_charge_number_cm3` **每一行都是 0**
- [ ] S1 的该列**无正值**，且**至少有一个负值**
- [ ] S2 的该列**无负值**，且**至少有一个正值**
- [ ] 无 NaN / Inf / 空单元格
- [ ] manifest 里搜不到残留的 `__FILL__`
- [ ] manifest 的 `solver.version` 与你报的 SCAPS 版本号一字不差

---

## 8. 导入命令（我方执行，附此供你核对语义）

```bash
python scripts/import_scaps_defect_reference.py \
  --project-root . \
  --suite reproducibility/scaps_defect_s0_s2_suite.json \
  --parameter-manifest <parameter_manifest.json> \
  --s0-csv <S0.csv> --s0-source-deck <S0.def> \
  --s1-csv <S1.csv> --s1-source-deck <S1.def> \
  --s2-csv <S2.csv> --s2-source-deck <S2.def> \
  --out <reference.json> \
  --solver-version "<SCAPS 版本>" \
  --extracted-at "<YYYY-MM-DD>" \
  --operator "<真实姓名>" \
  --confirm-independent-scaps-export \
  --confirm-direct-unmodified-rows
```

最后两个 flag 是**署名声明**，不是形式：

- `--confirm-independent-scaps-export` —— 断言每个 CSV 都是 SCAPS 独立运行导出的
- `--confirm-direct-unmodified-rows` —— 断言这些行未经插值、拟合或重采样

`--operator` 必须是真名。这两句话只能由真的开过 SCAPS 的人签。

---

## 9. 这份数据打开什么

交回后可解锁：`DEF-4` 的外部半边，以及 D9 依赖 S0–S2 的部分。

**不能**解锁的：

- `D9.6`（calibration / validation 分离）—— 那需要一个与标定数据**不相交**的
  第二独立 deck，不是这三个。现状是 `scaps_reference.json` 的 `Nd_ETL` 扫描既是
  `contact_phi_B_eV` 的拟合目标又是比较目标，同一批数据自证。
- `D7-E2`（多价缺陷 M1–M3）—— 另一套 deck，列名差一列
  （末列是 `charge_state_occupation_fraction_per_state`）。
- `D8-E3`（隧穿逐通道）—— 三重阻塞，缺数据只是其中一重；另两重（能级/势约定、
  势垒身份原语）在仓库内部，拿到数据也解不开。

如需 M1–M3 的同类交付单，另行索取。
