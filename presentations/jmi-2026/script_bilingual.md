# JMI 2026 — Bilingual Presentation Script

**Title:** High-throughput DFT and small-sample ML for chalcogenide PV absorber discovery, with device-scale validation
**Authors:** Xuanyan Chen et al. — HKUST(GZ)
**Slot:** ~25 min · 32 content slides

Format: each slide gets one **EN** delivery paragraph followed by the parallel **中文** version. Speak the English line, then the Chinese line, or pick whichever suits your audience.

---

## Cover slide

**EN —** Good morning. Today I will present our work on accelerating photovoltaic absorber discovery: combining high-throughput DFT, small-sample machine learning, and our own device-scale simulator, SolarLab. The whole pipeline runs from a database of more than 100,000 candidate structures down to a predicted device J–V curve.

**中文 —** 早上好。今天我汇报的工作是加速光伏吸收层材料的发现：将高通量 DFT、小样本机器学习以及我们自己开发的器件级仿真平台 SolarLab 三者结合起来。整条流程从超过 10 万个候选结构的数据库出发，最终输出一条可预测的器件级 J–V 曲线。

---

## Section divider — 01 PROBLEM (问题)

**EN —** I will start by motivating the problem: the efficiency–stability tradeoff that the field has not yet broken, and what existing absorbers can and cannot do.

**中文 —** 首先说明问题背景：领域内尚未突破的"效率—稳定性"权衡，以及现有吸收层材料能做什么、不能做什么。

---

## Slide 2 — Efficiency–stability design space (效率—稳定性设计空间)

**EN —** Three regimes coexist today. Silicon delivers 22 % efficiency with module lifetimes beyond 10⁴ hours. CIGS reaches 23.4 %, but is constrained by indium and gallium reserves. Perovskites surpass 26 %, but their T-eighty often falls under 500 hours at one sun and 85 °C. The unoccupied region — efficiency above 25 % combined with T-eighty above 10⁵ hours — is what motivates a third absorber class.

**中文 —** 当前共存三类体系：硅在 22 % 效率下可达万小时级寿命；CIGS 创纪录 23.4 %，但铟、镓资源受限；钙钛矿冲到 26 % 以上，但 1 sun、85 °C 下的 T₈₀ 经常不到 500 小时。我们要瞄准的"高效率 + 长寿命"区域目前完全空缺，因此需要寻找第三类吸收层材料。

---

## Slide 3 — CIGS: Cu(In,Ga)Se₂ chalcopyrite (CIGS 黄铜矿吸收层)

**EN —** CIGS is the structural template we will borrow from. It is a tetragonal I-4̄2d chalcopyrite — essentially a doubled zincblende cell — with a directly tunable band gap from 1.04 eV to 1.68 eV via the Ga / (In + Ga) ratio. It supports champion cells of 23.4 % and field lifetimes beyond 25 years. The In-free sister, kesterite Cu₂ZnSn(S,Se)₄, has been studied extensively at HKUST(GZ).

**中文 —** CIGS 给我们提供了结构模板。它属于四方相 I-4̄2d 黄铜矿——本质是双倍化的闪锌矿晶胞，禁带宽度可通过 Ga 比例在 1.04 至 1.68 eV 之间近线性调节。冠军电池效率 23.4 %，户外寿命超 25 年。香港科大（广州）也在持续研究其无铟同系物——锌锡硫硒矿（kesterite）Cu₂ZnSn(S,Se)₄。

---

## Slide 4 — Perovskite ABX₃ family (钙钛矿 ABX₃ 家族)

**EN —** Perovskites are our second reference structure. The cubic Pm-3̄m parent cell has corner-sharing BX₆ octahedra with the A cation in the cuboctahedral void. Stability follows the Goldschmidt tolerance factor: cubic phase for t between 0.9 and 1.0. Champion single-junction cells reach 26.7 %, and Hou Yi's group at NUS has demonstrated inverted devices with ligand anchoring and all-perovskite tandems. We will see later that we use a MAPbI₃ stack as the reference for our 2D solver fidelity test.

**中文 —** 钙钛矿是我们的第二个参考结构。立方 Pm-3̄m 母相由共角的 BX₆ 八面体构成，A 位阳离子坐落在八面体配位空隙中。稳定性遵循 Goldschmidt 容忍因子，t 在 0.9~1.0 之间为立方相。单结冠军 26.7 %；新加坡国立侯毅团队在反式器件、全钙钛矿叠层方面有重要进展。后面会用 MAPbI₃ 体系作为 2D 仿真器的保真度参考样品。

---

## Slide 5 — Limitations of established absorbers (现有吸收层的局限)

**EN —** Looking at the record-cell timeline: silicon took five decades to reach 27.6 %, perovskite reached 26.7 % within ten years. But the stability gap remains more than two orders of magnitude — driven by ion migration, A-site loss, and photo-induced degradation that encapsulation cannot fully address. To achieve commercial deployment we need both record efficiency and outdoor lifetime, which motivates new absorber chemistries.

**中文 —** 从冠军电池时间线看：硅花了五十年达到 27.6 %，钙钛矿十年内追到 26.7 %。但稳定性差距仍有两个数量级以上——离子迁移、A 位损失、光致退化等是吸收层本征问题，封装能减缓但无法根除。商业化要求"高效率 + 长寿命"同时满足，因此需要新的吸收层化学体系。

---

## Slide 6 — Talk outline (报告提纲)

**EN —** This is the roadmap. Six stages: PROBLEM, then STRATEGY — the chalcogenide template; ML and FEATURES — the small-sample HSE06 surrogate; CANDIDATES — our shortlist; BRIDGE — the device-simulation framework; and finally PAYOFF — the predicted device J–V plus outlook.

**中文 —** 这是整体提纲。六个阶段：问题 → 策略（硫族结构模板）→ ML 与特征（基于 HSE06 的小样本代理模型）→ 候选材料（筛选漏斗）→ 桥梁（器件级仿真）→ 收益（预测的器件 J–V 和展望）。

---

## Section divider — 02 STRATEGY (策略)

**EN —** Now the strategy: how do we narrow more than 10⁵ candidates to a tractable shortlist?

**中文 —** 接下来是策略：如何把超过 10 万条候选缩减到可处理的小集合？

---

## Slide 7 — Chalcogenide structural template (硫族结构模板)

**EN —** We restrict the search space using a chalcogenide structural prior. The anion sublattice is S, Se or Te — selenides dominate our shortlist. The active cation is a heavy p-block species — Sb, Sn, Bi — whose lone pair gives a dispersive valence-band maximum. The spectator cation — Cs, Ca, Sr, K — donates charge without contracting that dispersion. AB₂C₄ and ABC₃ are the dominant prototypes; Pnma is the dominant space group. This prior cuts the space from 10⁵ candidates to a physically plausible subset.

**中文 —** 我们用"硫族结构先验"来限定搜索空间。阴离子取 S、Se、Te，名单上以硒化物为主；活性阳离子取 Sb、Sn、Bi 等重 p 族元素，其孤对电子带来色散较强的价带顶；旁观阳离子取 Cs、Ca、Sr、K，仅捐电子不破坏价带色散。原型以 AB₂C₄ 和 ABC₃ 为主，空间群以 Pnma 居多。这个先验把 10⁵ 量级的搜索空间压到一个物理合理、可合成的子集。

---

## Slide 8 — End-to-end pipeline (端到端流程)

**EN —** This is the full pipeline. We start from Materials Project and OQMD, run the ML surrogate to rank candidates, perform DFT validation on the high-ranking subset, feed the validated parameters into our SolarLab device simulator, and close the loop with experimental collaboration. Today's talk walks through the middle of this diagram.

**中文 —** 这是端到端流程。起点是 Materials Project 和 OQMD 数据库，用 ML 代理模型进行排序，对高分子集做 DFT 验证，再把参数输入 SolarLab 器件仿真，最后与实验组合作闭环。今天的内容主要围绕中间几个环节展开。

---

## Section divider — 03 ML + FEATURES (机器学习与特征)

**EN —** Now the heart of the work: how do we train a useful model on 473 labelled materials?

**中文 —** 接下来是工作的核心：如何在 473 个有标签样本上训练出可用的模型。

---

## Slide 9 — Motivation for a learned surrogate (代理模型的必要性)

**EN —** Why a surrogate? Each candidate requires a four-stage VASP run: PBE relaxation, SCF, DFPT for the dielectric tensor, and HSE06 for the band structure. DFPT and HSE06 dominate runtime — tens to hundreds of CPU-hours per structure. Enumerating the entire 10⁵ pool by full DFT is computationally infeasible, so we train a surrogate on the 473 materials we already have and use it to rank the unlabelled pool.

**中文 —** 为什么要做代理模型？每个候选需要四级 VASP 流程：PBE 弛豫 → SCF → DFPT 介电张量 → HSE06 能带；其中 DFPT 与 HSE06 占主要耗时，单结构动辄数十到上百 CPU 小时。10⁵ 量级全部跑 DFT 计算上不可行，所以用现有 473 个有标签样本训练代理，对未标注池进行排序。

---

## Slide 10 — Hand-crafted feature set 167-D (167 维手工特征)

**EN —** The features. About 27 dimensions are compositional Magpie statistics — mean, std, min, max, mode of element properties weighted by stoichiometry. Another 140 dimensions come from CrystalNN site fingerprints, RDF, density, packing — all from matminer. Every featuriser tolerates NaN, which the HistGBR splits handle natively. We cache the resulting numpy matrices on disk so re-fits run in seconds. Frozen MEGNet 16-D embeddings score R² = 0.366 — below our hand-crafted baseline.

**中文 —** 特征构造。组成方面约 27 维 Magpie 统计——按化学计量权重对元素性质做均值、方差、最值、众数等统计；结构方面 140 维来自 matminer 的 CrystalNN 位点指纹、RDF、密度、堆积。所有特征都允许 NaN，HistGBR 在分裂时原生支持。numpy 矩阵缓存到磁盘，重训练只需数秒。冻结 MEGNet 16 维嵌入得 R² = 0.366，低于手工特征基线。

---

## Slide 11 — Composite PV figure of merit (复合光伏品质因子)

**EN —** We compress four physical quantities into a single scalar y in [0, 1]. The four factors: f_gap is a triangular Shockley–Queisser window peaked at 1.34 eV; 1 / (1 + m_avg) rewards low effective mass and therefore high mobility; tanh(ε/10) saturates beyond ε ≈ 10 to reflect diminishing returns from dielectric screening; 1 / (1 + E_b / 0.1) favours Wannier–Mott regimes below 100 meV. Bounded, differentiable, physically interpretable — replaces four independent regression heads with one.

**中文 —** 我们把四个物理量压缩为单一标量 y ∈ [0, 1]。四个因子：f_gap 是一个以 1.34 eV 为峰、贴近 Shockley–Queisser 窗口的三角函数；1/(1+m_avg) 奖励低有效质量、高迁移率；tanh(ε/10) 在介电常数大于 10 后饱和，反映介电屏蔽的边际效应；1/(1+E_b/0.1) 偏好束缚能小于 100 meV 的 Wannier–Mott 体系。整体有界、可微、物理可解释，把原本四个回归头替换成一个。

---

## Slide 12 — Histogram-based gradient boosting (基于直方图的梯度提升)

**EN —** The mean predictor is HistGBR. Stage-wise additive: F_m = F_{m−1} + ν · h_m, where each h_m fits the negative gradient of the loss at the previous stage. Histogram binning into 256 buckets reduces split scans from O(N) to O(B), giving roughly a three-times speedup. NaN handling is native — missing values learn an independent split direction at every node. We use 600 boosting iterations, depth 4, learning rate 0.05, minimum 10 samples per leaf. The target is wrapped in log1p(y / 0.08) to compress the heavy-tailed PV-score distribution.

**中文 —** 均值预测器选用 HistGBR。逐级累加：F_m = F_{m−1} + ν · h_m，每个 h_m 拟合上一级损失的负梯度。直方图分桶（256 桶）把分裂扫描复杂度从 O(N) 降为 O(B)，约 3× 加速。NaN 由模型原生处理——每个节点为缺失值学习独立的分裂方向。超参数：600 轮提升、深度 4、学习率 0.05、叶子最小样本 10。目标变量用 log1p(y/0.08) 进行变换，压缩重尾的 PV 评分分布。

---

## Slide 13 — GP + RF uncertainty stack (高斯过程 + 随机森林不确定性堆叠)

**EN —** For the uncertainty channel we stack a Gaussian process and a random forest. The GP uses a Matérn-5/2 kernel with anisotropic length scale optimised by L-BFGS-B on the marginal log-likelihood. White-noise variance absorbs DFT scatter. The random forest with 300 trees compensates for GP underconfidence in sparsely sampled regions. Stack mean and variance are non-negative ridge mixtures of the two, calibrated on out-of-fold predictions.

**中文 —** 不确定性通道采用高斯过程 + 随机森林堆叠。GP 使用 Matérn-5/2 核，各向异性长度尺度通过 L-BFGS-B 在边缘对数似然上优化；白噪声方差吸收 DFT 数据噪声。300 棵树的随机森林弥补 GP 在稀疏采样区域的过度自信。堆叠均值和方差通过非负岭回归在折外预测上校准混合权重。

---

## Slide 14 — UCB acquisition (UCB 采集函数)

**EN —** Active learning uses upper-confidence-bound acquisition: α(x) = μ(x) + κ · σ(x), with κ = 2. Crucially we decouple μ and σ between two models — μ from HistGBR (the better mean predictor), σ from the GPR + RF stack (the calibrated uncertainty source). This decoupling prevents GP underconfidence from suppressing the high-score tail. Top-N entries by α form the next DFT batch; returned labels feed back into both models.

**中文 —** 主动学习采集函数采用 UCB：α(x) = μ(x) + κ · σ(x)，κ = 2。关键设计是把均值与方差解耦：μ 来自 HistGBR（在已标注集上更准的均值预测器），σ 来自 GPR + RF 堆叠（校准良好的不确定性源）。这样可避免 GP 的低不确定性抑制高分尾部。按 α 排序的前 N 个进入下一轮 DFT 计算，返回的标签同时反哺两个模型。

---

## Slide 15 — Validation under random and grouped CV (随机与分组交叉验证)

**EN —** Validation. Five-fold by five-seed random CV on the 473-material set: HistGBR achieves R² = 0.466 ± 0.035, MAE = 0.012, top-20 recall 0.50. That is +50 % over our prior GPR / GB / ExtraTrees baseline. Ablations: GBR 0.399, ExtraTrees 0.234, MEGNet fine-tuned 0.307, MEGNet frozen 0.366. The honest story shows in grouped CV: prototype-grouped R² collapses to roughly zero — extrapolation to unseen prototypes is poor — and chalcogen-family-grouped R² is around 0.15, with selenium the hardest hold-out.

**中文 —** 验证结果。5 折 × 5 随机种子 CV：HistGBR 的 R² = 0.466 ± 0.035，MAE = 0.012，top-20 召回率 0.50；相对此前 GPR / GB / ExtraTrees 基线提升 50 %。消融：GBR 0.399，ExtraTrees 0.234，MEGNet 微调 0.307，冻结 0.366。真实情况体现在分组 CV：按结构原型分组 R² ≈ 0，对未见原型外推能力差；按硫族家族分组 R² ≈ 0.15，硒最难。

---

## Slide 16 — Per-group residual audit (分组残差诊断)

**EN —** This residual audit drives the next active-learning batch. Group-priority score equals RMSE × √n + 0.02 × (missed top-20). Selenium with n = 182 has RMSE 0.042 and 9 of 17 true top-20 missed — highest priority. The high-score bin y > 0.1 has RMSE 0.145, identifying the under-modelled tail. ABC₃ with n = 56 shows the most pronounced data-coverage gap, with a 50 % top-20 miss rate. These groups are where the next DFT batch should focus.

**中文 —** 残差诊断决定下一轮主动学习方向。优先级分数 = RMSE × √n + 0.02 × (top-20 漏检数)。Se 组（n=182）RMSE 0.042，17 个真 top-20 漏检 9 个，最高优先级；高分区 y > 0.1（n=14）RMSE 0.145，是欠拟合的尾部；ABC₃（n=56）漏检率高达 50 %，是覆盖最差的子群。下一轮 DFT 应重点补这些组。

---

## Section divider — 04 CANDIDATES (候选)

**EN —** What did the pipeline pick out?

**中文 —** 整条流程筛出了哪些材料？

---

## Slide 17 — Top-fourteen shortlist (前 14 候选)

**EN —** A fourteen-candidate shortlist with PV score above 0.10. All chalcogenides; eleven of fourteen are selenides; the top six are all selenides. AB₂C₄ supplies seven of fourteen entries. Pnma is the dominant space group with seven entries; the rest fall in SG 12, 14, 72, 15. Aggregate descriptors: mean E_g = 1.32 eV, mean ε = 12.5, mean m_avg = 0.58 m₀. Twenty-four percent are direct-gap; the rest are kept for combined low effective mass, high dielectric constant and low binding energy.

**中文 —** 14 候选短名单，PV 分数都大于 0.10。全部为硫族化合物，14 个中 11 个是硒化物，前 6 全部是硒化物。AB₂C₄ 占 7 个；Pnma 空间群占 7 个，其余分布在 SG 12、14、72、15。聚合描述符：平均 E_g = 1.32 eV，ε = 12.5，m_avg = 0.58 m₀。24 % 为直接带隙，其余 76 % 因综合具备低有效质量、高介电、低束缚能而保留。

---

## Slide 18 — Screening funnel (筛选漏斗)

**EN —** This is the funnel. Stage 1: more than 10⁵ structures from MP and OQMD. Stage 2: 647 candidates with HSE band structure and DFPT dielectric tensor completed. Stage 3: 473 ML-ready, after removing metallic-band and missing-CONTCAR entries. Stage 4: 41 PV-promising with composite score above 0.05. Stage 5: 14 with score above 0.10 — feeding the next active-learning DFT round. The funnel narrows by roughly four orders of magnitude.

**中文 —** 这是筛选漏斗。第一级：MP + OQMD 中 10⁵ 量级结构；第二级：647 个完成 HSE 能带 + DFPT 介电的候选；第三级：473 个 ML 就绪样本（剔除金属带、CONTCAR 缺失等 174 个）；第四级：41 个综合分数大于 0.05 的 PV 候选；第五级：14 个分数大于 0.10 的入选短名单，进入下一轮主动学习。漏斗整体压缩约四个数量级。

---

## Slide 19 — Lead candidate Se₁₆Sn₈Zr₄ (头号候选 Se₁₆Sn₈Zr₄)

**EN —** The lead candidate is Se₁₆Sn₈Zr₄. AB₂C₄ prototype, Pnma space group. Band gap 1.12 eV — within 0.22 eV of the Shockley–Queisser optimum at 1.34 eV. Effective mass 0.43 m₀, low — suggesting good mobility. Dielectric constant 19.5 — strong screening, suppressing exciton formation. Exciton binding energy 7 meV, well below k_BT at room temperature, placing the absorber firmly in the Wannier–Mott regime. Composite PV score 0.487 — the highest in the screened set. Ternary phases of Sn–Zr–Se have synthesis routes in the literature, but no PV device demonstration yet.

**中文 —** 头号候选 Se₁₆Sn₈Zr₄。AB₂C₄ 原型，Pnma 空间群。E_g = 1.12 eV，距 SQ 最优值 1.34 eV 仅 0.22 eV；m_avg = 0.43 m₀，预示高迁移率；ε = 19.5，介电屏蔽强，抑制激子；E_b = 7 meV，远低于室温 k_BT，处于 Wannier–Mott 区。复合 PV 评分 0.487，全集最高。Sn–Zr–Se 三元相文献已报道合成路径，但尚无光伏器件演示。

---

## Slide 20 — Secondary candidate Cs₂Sb₆Se₂ (次选 Cs₂Sb₆Se₂)

**EN —** The secondary candidate, Cs₂Sb₆Se₂. ABC₃ prototype, C2/m space group. Band gap 1.28 eV near the SQ optimum. Effective mass 0.81 m₀ — heavier than the lead, but in a device-relevant range. Dielectric constant 12.6, binding energy 56 meV. Composite score 0.298 — complementing the lead from a distinct prototype. Antimony's 5s² lone pair gives a deep, dispersive valence-band maximum. Cs–Sb–Se ternaries have established solution-processing routes.

**中文 —** 次选 Cs₂Sb₆Se₂。ABC₃ 原型，C2/m 空间群。E_g = 1.28 eV，靠近 SQ 最优；m_avg = 0.81 m₀ 偏重但仍在器件可接受范围；ε = 12.6，E_b = 56 meV。复合分数 0.298，与头号候选在结构原型上互补。Sb 的 5s² 孤对赋予深而色散的价带顶。Cs–Sb–Se 三元相已有溶液法合成报道。

---

## Section divider — 05 BRIDGE — DEVICE SIM (桥梁——器件级仿真)

**EN —** A high-PV-score material is necessary but not sufficient for a high-PCE device. We need a bridge from bulk descriptors to a predicted J–V curve.

**中文 —** 高 PV 分数对器件高 PCE 是必要条件但非充分条件。我们需要一个从体材料描述符到器件级 J–V 曲线的"桥梁"。

---

## Slide 21 — Motivation for device-scale simulation (器件级仿真的必要性)

**EN —** DFT delivers the bulk descriptors — band gap, effective mass, dielectric constant, exciton binding energy. But device performance also depends on stack geometry, contact selectivity, optical absorption profile, and recombination kinetics. We need drift-diffusion coupled to Poisson across the full HTL / absorber / ETL stack, coherent TMM optics for thin-film interference, two-dimensional effects for grain boundaries, and mobile-ion transport to capture hysteresis in perovskite-class absorbers.

**中文 —** DFT 给出体材料描述符——带隙、有效质量、介电常数、激子束缚能。但器件性能还依赖叠层几何、接触选择性、光学吸收剖面、复合动力学。我们必须在 HTL/吸收层/ETL 整个叠层上求解 Poisson + 漂移扩散方程；用相干 TMM 处理薄膜光学干涉；用二维效应捕捉晶界；引入离子迁移再现钙钛矿的迟滞。

---

## Slide 22 — SolarLab framework (SolarLab 框架)

**EN —** SolarLab is our in-house framework. Backend is a Python solver with FastAPI server-sent events streaming intermediate results. Frontend is a TypeScript / Vite workstation with publication-quality figure rendering. Numerics use Radau implicit Runge–Kutta with BDF fallback and bisection-in-time on stiff steps. Optics is coherent TMM under the ASTM G-173 AM1.5G reference spectrum. Selective contacts use the Robin formulation in S_n and S_p, validated against the 1D solver to within 6 µV. The whole framework is open-source, scriptable and REST-driven for high-throughput integration.

**中文 —** SolarLab 是我们自研的器件级仿真平台。后端是 Python 求解器 + FastAPI SSE 流式输出；前端是 TypeScript / Vite 工作站，带出版级图形渲染；数值方案为 Radau 隐式 Runge–Kutta + BDF 备选 + 时间二分细化；光学采用 ASTM G-173 AM1.5G 下的相干 TMM；选择性接触使用 S_n、S_p 形式的 Robin 边界，与 1D 求解器对比误差小于 6 µV。整套框架开源、可脚本化、REST 驱动，便于接入高通量流水线。

---

## Slide 23 — Drift-diffusion governing equations (漂移扩散控制方程)

**EN —** The governing equations are three coupled PDEs per absorber layer. Electron and hole continuity link to current densities J_n and J_p, which decompose into drift plus diffusion via the Einstein relation. Poisson's equation closes the system, including a mobile-ion source term ρ_ion. Recombination R covers Shockley–Read–Hall, Auger, and radiative channels. Generation G is the optical generation profile from TMM integrated over the AM1.5G spectrum.

**中文 —** 每个吸收层有三组耦合 PDE：电子连续性、空穴连续性，以及把电流密度分解为漂移 + 扩散（由 Einstein 关系连接）。Poisson 方程闭合系统，包含离子源项 ρ_ion。复合项 R 涵盖 SRH、Auger、辐射三种通道；产生项 G 来自 TMM 在 AM1.5G 光谱上的光生剖面积分。

---

## Slide 24 — Transfer-matrix-method optics (TMM 光学)

**EN —** Optics use the coherent transfer-matrix method. Each layer j is a 2×2 characteristic matrix L_j acting on the tangential E and H field components. Phase thickness δ_j depends on the complex refractive index n + ik. The system matrix is the product L_1 L_2 … L_N, yielding wavelength-resolved reflectance, transmittance and absorption. The generation profile G(x, λ) is proportional to |E(x, λ)|² × α(λ), integrated over AM1.5G. This supersedes Beer–Lambert, capturing interference, back-reflector and interface effects.

**中文 —** 光学采用相干 TMM。每层 j 用 2×2 特征矩阵 L_j 作用于切向 E、H 场分量；相位厚度 δ_j 由复折射率 n + ik 决定；系统矩阵 M = L_1 L_2 … L_N 给出波长分辨的 R、T、A。光生剖面 G(x, λ) ∝ |E(x, λ)|² · α(λ)，再对 AM1.5G 光谱积分。相比 Beer–Lambert 模型，TMM 还能刻画干涉、背反射、界面效应。

---

## Slide 25 — Mobile-ion transport (离子迁移)

**EN —** Mobile-ion transport explains the J–V hysteresis observed in perovskite cells. The ion-vacancy density P satisfies a continuity equation with a steric factor (1 − P / P_lim) capping density at the available lattice-site limit, estimated as 1 to 5 percent of the cation-site density in MAPbI₃. Ion drift under bias redistributes mobile charge, screens the field, and modulates recombination. The coupling into Poisson is through ρ_ion = q (P − P_eq). Time scales of 0.01 to 100 seconds reproduce the measured scan-rate dependence.

**中文 —** 离子迁移解释了钙钛矿电池的 J–V 迟滞。空位浓度 P 满足带饱和因子 (1 − P/P_lim) 的连续性方程，P_lim 取 MAPbI₃ 阳离子位密度的 1–5 %。偏压驱动下离子重新分布，屏蔽内电场，调制复合速率，并通过源项 ρ_ion = q (P − P_eq) 反馈到 Poisson 方程。0.01–100 s 的特征时间尺度再现了扫描速率依赖性。

---

## Slide 26 — Robin selective-contact boundary conditions (Robin 选择性接触边界)

**EN —** Contacts use Robin boundary conditions instead of Dirichlet pinning. Each contact has separate surface-recombination velocities S_n and S_p; selectivity emerges from S_n,L ≪ S_p,L for a hole-extracting left contact, and the mirror condition on the right. The S → ∞ limit recovers Dirichlet pinning; S → 0 gives a perfectly blocking Neumann contact. This formulation captures the surface-recombination physics that SCAPS-1D approximates with Dirichlet pinning only.

**中文 —** 接触采用 Robin 边界条件而非 Dirichlet 钉扎。每个接触各自具有 S_n、S_p；选择性体现在 S_n,L ≪ S_p,L（构成提取空穴的左接触），右侧镜像。S → ∞ 极限退化为 Dirichlet 钉扎；S → 0 给出完全阻断的 Neumann 边界。这套公式自然刻画了表面复合物理，而 SCAPS-1D 通常仅用 Dirichlet 钉扎近似。

---

## Slide 27 — Capability comparison with SCAPS-1D (与 SCAPS-1D 能力比较)

**EN —** This is the capability comparison with SCAPS-1D, the reference solver in the field. SolarLab supports 1D + 2D, true drift-diffusion mobile-ion transport, multi-layer coherent TMM optics, time-resolved hysteresis, selective Robin contacts in S_n and S_p, series-matched tandem coupling, and is fully scriptable with REST + Python. SCAPS-1D is 1D-only, treats ions as a steady-state hack, uses Beer–Lambert optics, has no time-resolved hysteresis, and is GUI-driven. The Radau + BDF solver also outperforms SCAPS's Gummel iteration on stiff steps.

**中文 —** 这是与领域参考求解器 SCAPS-1D 的能力对照。SolarLab 支持 1D + 2D；离子迁移用真正的漂移扩散；光学是多层相干 TMM；带时间分辨迟滞；接触是 S_n、S_p 形式的选择性 Robin；支持串联匹配的叠层电池；并完全脚本化（REST + Python）。SCAPS-1D 仅 1D；离子是稳态近似；光学是 Beer–Lambert；无迟滞；只能 GUI。Radau + BDF 求解器在硬步上也优于 SCAPS 的 Gummel 迭代。

---

## Slide 28 — 1D vs 2D solver fidelity (1D vs 2D 保真度比较)

**EN —** This is the fidelity test. Reference stack is spiro HTL 200 nm / MAPbI₃ 400 nm / TiO₂ ETL 100 nm with selective Robin contacts. The 2D variant adds a single vertical grain boundary at the lateral midplane of the absorber. The 1D V_oc is 0.910 V; the 2D V_oc drops to 0.860 V, a 50 mV reduction directly attributable to grain-boundary recombination — a loss channel resolved only by the 2D solver. A 1D-only model would misestimate the open-circuit voltage by this margin. The 28-voltage 2D sweep takes about 22 minutes on 4 CPU cores.

**中文 —** 这是保真度对比。参考叠层：spiro HTL 200 nm / MAPbI₃ 400 nm / TiO₂ ETL 100 nm + 选择性 Robin 接触。2D 版本在吸收层横向中线加一条竖直晶界。1D V_oc = 0.910 V；2D V_oc = 0.860 V，比 1D 低 50 mV，差距完全来自 2D 才能解析的晶界复合通道。如果仅用 1D 模型，开路电压会高估 50 mV。28 个偏压点的 2D 扫描在 4 核 CPU 上约 22 分钟。

---

## Section divider — 06 PAYOFF (收益)

**EN —** Closing the loop: from a 10⁵-database scan to a predicted device J–V curve.

**中文 —** 闭环收尾：从 10⁵ 量级数据库扫描到一条预测的器件 J–V 曲线。

---

## Slide 29 — Predicted device J–V (预测的器件 J–V)

**EN —** This is the predicted J–V for a leading candidate, on the same MAPbI₃ reference stack as the 1D-vs-2D run. The current curve is a placeholder: MAPbI₃ baseline with V_oc shifted by +30 mV as a candidate-gap surrogate. Once we wire the real Se₁₆Sn₈Zr₄ n, k data and DFT parameters into the 2D solver, this placeholder is replaced. The point is that we now have a workflow that takes a database hit all the way to a predicted PCE — the figure of merit for synthesis decisions, and explicit, falsifiable performance targets for our experimental collaborators.

**中文 —** 这是头号候选在与 1D vs 2D 相同的 MAPbI₃ 参考叠层上的预测 J–V。当前曲线是占位符——以 MAPbI₃ 基准 V_oc 加 30 mV，作为候选材料带隙的替代值；一旦把真实 Se₁₆Sn₈Zr₄ 的 n、k 数据与 DFT 参数接入 2D 求解器，占位符会被替换。重点是：我们已经搭起了一条从数据库命中到器件级 PCE 的完整工作流——这正是合成决策需要的品质因子，也是我们能向实验合作方提出的、明确可证伪的性能目标。

---

## Slide 30 — Outlook (展望)

**EN —** The roadmap from here. Multi-task learning on the 36,000 PBE gaps in Materials Project should close the prototype-grouped CV gap. Spectroscopic Limited Maximum Efficiency (SLME) replacing the SQ upper bound — this requires the absorption coefficient α(E). Targeted DFT on Se-rich AB₂C₄ and ABC₃ candidates in space groups 62, 12, 72, 15. Closed-loop automation of DFT batch submission and surrogate re-fitting. And experimental collaboration: synthesise the top three candidates and feed measured J–V data back into the ML pipeline.

**中文 —** 后续计划。利用 Materials Project 上 ~36,000 个 PBE 带隙做多任务学习，闭合按结构原型分组的 CV 鸿沟；用 SLME（α(E) 加权的最大效率）替换 Shockley–Queisser 上界；针对硒为主、空间群 62/12/72/15 的 AB₂C₄、ABC₃ 候选做定向 DFT 计算；闭环自动化 DFT 提交与代理模型再训练；并与实验合作方合作，合成前三候选材料并将实测 J–V 数据反馈入 ML 流水线。

---

## Slide 31 — Summary (总结)

**EN —** To summarise. We trained a small-sample ML model on 473 DFT-labelled materials and improved the GPR / GB / ExtraTrees baseline by 50 % in R². We produced a fourteen-candidate chalcogenide shortlist; Se₁₆Sn₈Zr₄ sits near the Shockley–Queisser optimum. SolarLab provides the bridge from DFT-derived parameters to predicted device-level J–V. A compact, physically motivated descriptor and a small-sample model identify a tractable subspace under stringent grouped-CV diagnostics. And experimental collaboration completes the discovery → simulation → experiment loop.

**中文 —** 总结。在 473 个 DFT 有标签样本上训练的小样本 ML 模型，相对 GPR / GB / ExtraTrees 基线提升 50 %；筛出 14 候选硫族材料的短名单，其中 Se₁₆Sn₈Zr₄ 处于 SQ 最优附近；SolarLab 提供 DFT 参数到器件级 J–V 的桥梁；紧凑且物理可解释的描述符 + 小样本模型，在严格的分组 CV 诊断下仍能锁定可处理子空间；与实验合作方合作，闭合"发现 → 仿真 → 实验"循环。

---

## Slide 32 — References (参考文献)

**EN —** Key references. Yan and Hou groups for the chalcogenide and perovskite chemistry. Friedman 2001 and Ke 2017 for boosting. Rasmussen and Williams for GP. Snoek 2012 for UCB. Ward and matminer for the feature library. Heyd 2003 for HSE06. Yu and Zunger 2012 for SLME. Selberherr 1984 for device simulation. Pettersson 1999 for TMM. Burgelman 2000 for SCAPS. Calado 2016 and van Reenen 2015 for ion migration. Chen 2019 for MEGNet. Thank you — happy to take questions.

**中文 —** 主要参考文献。硫族与钙钛矿化学：Yan 与 Hou 团队；提升模型：Friedman 2001、Ke 2017；高斯过程：Rasmussen & Williams；UCB：Snoek 2012；特征库：Ward 与 matminer；HSE06：Heyd 2003；SLME：Yu & Zunger 2012；器件仿真：Selberherr 1984；TMM：Pettersson 1999；SCAPS：Burgelman 2000；离子迁移：Calado 2016、van Reenen 2015；MEGNet：Chen 2019。谢谢，欢迎提问。

---

## Speaker delivery notes

- **Pace target:** ~25 minutes for 32 content slides ≈ 45 s per slide. Sections 1, 5, 6 are conceptually heavy — slow down. Sections 2, 4 are lighter — keep moving.
- **Equation slides (11, 23, 24, 25, 26):** read the symbols out loud once, then say what each piece *does*; do not re-derive on stage.
- **Comparison slides (27, 28):** anchor on the punchline first (50 mV V_oc gap; 2D-only loss channel), then walk the bullets.
- **Outlook + summary (30, 31):** finish strong, end on the experimental-collaboration sentence.
- **节奏建议：** 25 分钟讲 32 张内容页，约每张 45 秒。第 1、5、6 节信息密度高，放慢；第 2、4 节较轻，加快。
- **公式页（11, 23, 24, 25, 26）：** 读一遍符号即可，重点说每一项"做什么"，现场不要推导。
- **对比页（27, 28）：** 先抛核心结论（50 mV V_oc 差距、2D 才能看到的损失通道），再讲细节。
- **展望与总结（30, 31）：** 收尾要稳，最后落在"实验合作"这句话上。
