# Phase 1 numerical refinement certificates

本文档定义 `reproducibility/numerical_refinement_registry.yaml` 的执行和证据契约。该 registry 预注册路线图要求的五条 `grid x tolerance` lane，并为 c-Si 增加一条 versioned resolved companion lane；阈值、配置内容哈希、adapter 和矩阵在求解前固定。修改阈值必须使用新的 lane ID，不能根据已有结果原地放宽。

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
| `csi-qf-frequency-domain` | N100/200/300 | FD step 1/0.5/0.25 | C-V、Mott-Schottky intercept/effective doping |
| `csi-qf-frequency-domain-resolved-v2` | N200/300/400 | FD step 1/0.5/0.25 | resolved C-V、Mott-Schottky intercept/effective doping |
| `twod-uniform-limit` | x/y multiplier 1/2/4 | componentwise atol factor 1/0.1/0.01 | 2D-to-1D J-V envelope、Voc、Jsc |
| `interface-recombination-charge-off` | N30/60/90 | residual factor 1/0.1/0.01 | interface flux、归一化 J-V、Voc |

归一化 J-V/trace 使用 absolute `L_inf <= 0.5%`，Voc 使用 absolute `<= 1 mV`，Jsc 使用 relative `<= 0.2%`。离子库存漂移 gate 为 `<= 1e-10`。c-Si 的 all-face spread 和 backward error gate 分别为 `<= 5e-4` 和 `<= 1e-10`。其余 lane-specific quality gate 见 registry；它们仍是 internal candidate gate，不能解释成外部物理误差条带。

2D 的 `grid` 是同步乘数：每层竖直区间数为 `5 x multiplier`，横向区间数为 `4 x multiplier`，匹配的 1D 区间总数为三层竖直区间总数。它不是只细化 y 而固定 x。1D 和 2D 两侧都比较电子加空穴的传导电流，不混入仅 1D 路径具有的离子/位移电流；除 refinement gate 外，每个 cell 还要求绝对 2D-to-1D 归一化差不超过 0.5%，横向均匀性同时检查 `n` 和 `p`。

## Protocol provenance

每个 production adapter 在 cell metadata 中写入：

- `protocol`：canonical protocol document；
- `protocol_schema`：experiment、numerical execution 或 bundle schema；
- `protocol_hash`：canonical document 的 SHA-256。

mobile-ion J-V 使用真实 `ExperimentProtocol`；c-Si 将每个 DC bias 的 QF impedance protocol 明确组成 bundle；frozen/interface steady ladder 使用不虚构 scan rate 的 numerical protocol；2D bundle 分列实际执行的 1D forward/reverse protocol、被比较的 forward branch 和 2D ascending finite-time protocol。2D 的 1D `v_rate` 显式设为 `V_step / settle_time`，使两条路径的每点 dwell 相同。

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
python scripts/run_numerical_refinement.py csi-qf-frequency-domain --dry-run
python scripts/run_numerical_refinement.py csi-qf-frequency-domain-resolved-v2 --dry-run
python scripts/run_numerical_refinement.py twod-uniform-limit --dry-run
python scripts/run_numerical_refinement.py interface-recombination-charge-off --dry-run
```

建议固定 BLAS 线程后逐 lane 执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py scaps-mirror-frozen-ion-ss
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py ionmonger-mobile-ion-transient
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py csi-qf-frequency-domain
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py csi-qf-frequency-domain-resolved-v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py twod-uniform-limit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/run_numerical_refinement.py interface-recombination-charge-off
```

先执行一个 cell 并留下可恢复状态：

```bash
python scripts/run_numerical_refinement.py twod-uniform-limit --max-cells 1 --allow-noncertified-exit-zero
python scripts/run_numerical_refinement.py twod-uniform-limit
python scripts/run_numerical_refinement.py twod-uniform-limit --retry-failed
```

source、executor 或 environment 变化会产生新的 `run_id`，不会误接到旧 run。相同 run 重复执行只复用已验证的 cell。

## 真实矩阵成本和证据边界

完整 Phase 1.1 minimum-roadmap 矩阵是 5 条 lane、每条 9 个 cell，共 45 个 cell；resolved c-Si companion 使当前 registry 合计 54 个 cell。其内部工作量至少包括：

- frozen SCAPS：279 个 residual-certified steady voltage points；
- mobile IonMonger：9 次 forward/reverse transient J-V，每次 40 个采样点，且可能触发 recovery/bisection；
- c-Si minimum-roadmap lane：54 个 bias/FD-step operating-point calls，共 162 个 frequency points；resolved-v2 companion 另有同等工作量；
- 2D：117 个 2D finite-time voltage settles，外加 9 个 matched 1D forward/reverse sweeps 和 2D seed；
- interface charge-off：225 个 residual-certified steady voltage points，并计算 interface-state RHS/flux residual。

因此 CI 只运行 schema、resume 和注入式 adapter smoke；完整物理矩阵应作为受控的长时任务逐 lane 运行。当前 infrastructure、dry-run 或 adapter smoke 不能替代这 54 个 cell 的真实结果。

`certified` 只表示该冻结代码/config/protocol/environment 下的内部数值收敛。它不等于 SCAPS/IonMonger 外部 solver parity，不等于实验验证，也不证明材料参数或模型闭包唯一正确。
