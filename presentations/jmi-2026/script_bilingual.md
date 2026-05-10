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

**EN —** Why this design? An ML model needs one target, not four. If we ran four separate regressors — one for band gap, one for effective mass, one for dielectric, one for binding energy — each would pick its own optimum, and a candidate could win on three but lose on the fourth. Multiplying them together forces the model to find materials that are *jointly* good. Each factor is a known PV constraint: f_gap is a triangular window peaked at the Shockley–Queisser optimum of 1.34 eV; 1/(1+m_avg) rewards high mobility; tanh(ε/10) saturates because more screening above ε ≈ 10 buys little; 1/(1+E_b/0.1) keeps us in the Wannier–Mott regime where excitons dissociate easily. The score is bounded in [0, 1], differentiable, and a high score is hard to fake.

**中文 —** 为什么这样设计？ML 模型需要一个回归目标，而不是四个。如果对带隙、有效质量、介电常数、束缚能各跑一个回归，每个都会找自己的最优解；某个候选可能三个赢、一个输。把四项相乘后，模型必须同时优化所有四个，才能拿高分。每个因子都对应一个已知的光伏约束：f_gap 是以 SQ 最优 1.34 eV 为峰的三角窗；1/(1+m_avg) 奖励高迁移率；tanh(ε/10) 在 ε ≈ 10 之后饱和，因为继续增大屏蔽收益递减；1/(1+E_b/0.1) 把候选限制在激子易解离的 Wannier–Mott 区。整体得分在 [0, 1] 之间、可微、且很难"作弊"。

---

## Slide 12 — Histogram-based gradient boosting (基于直方图的梯度提升)

**EN —** Why HistGBR for the mean predictor? Three practical reasons. First, our 167-D feature set mixes compositional statistics with structural fingerprints — heterogeneous tabular data, exactly where boosted trees beat neural networks. Second, our sample size is only 473 — too small for deep models, just right for trees. Third, our features have NaN entries from missing structural data, and HistGBR splits handle NaN natively without imputation. The mechanism is intuitive: build a weak tree, look at where it makes mistakes, build the next tree to fix those mistakes, repeat 600 times. Histogram binning into 256 buckets makes each split scan O(B) instead of O(N), giving roughly 3× speedup. We log-transform the target with log1p(y/0.08) so the model resolves both the high-score peak and the bulk equally well — without it, the heavy tail dominates the loss.

**中文 —** 为什么用 HistGBR 做均值预测？三个实际原因。第一，我们 167 维特征里既有组成统计、又有结构指纹——这是典型的异构表格数据，梯度提升树通常胜过神经网络。第二，样本量只有 473，深度模型嫌小，树模型刚好。第三，特征里有 NaN（部分结构数据缺失），HistGBR 在分裂时原生支持 NaN，无需插补。机制本身很直观：先建一棵弱树，看它哪里错了，下一棵树专门修这些错，重复 600 次。直方图分 256 桶让每次分裂扫描从 O(N) 降到 O(B)，约 3 倍加速。我们对目标做 log1p(y/0.08) 变换，让模型对高分峰和均值区都有同等分辨率——否则重尾会主导损失函数。

---

## Slide 13 — GP + RF uncertainty stack (高斯过程 + 随机森林不确定性堆叠)

**EN —** Why a stack? Active learning needs a calibrated standard deviation σ(x). A pure Gaussian process gives smooth, theoretically grounded uncertainty — but it tends to be over-confident in regions where training data is sparse, exactly the regions we most want to explore. A pure random forest gives an empirical σ from per-tree variance, but no closed-form posterior. Combining the two restores calibration: where the GP collapses to its prior and reports near-zero σ, the RF still has tree-to-tree variance that picks up the slack. The mixing weights are fit by non-negative ridge regression on out-of-fold predictions, which guarantees the σ band actually covers the true value at the right rate. The Matérn-5/2 kernel is chosen because PV-score is twice-differentiable in the descriptors, but not infinitely smooth — Matérn-5/2 matches that regularity.

**中文 —** 为什么要做堆叠？主动学习需要一个校准良好的标准差 σ(x)。纯高斯过程给出的不确定性平滑且理论严谨——但在训练数据稀疏的区域往往过度自信，而那恰是我们最需要探索的区域。纯随机森林通过 300 棵树的预测方差给出经验 σ，但没有解析后验。把两者堆叠就能恢复校准：GP 退化到先验、σ 接近零的地方，RF 的树间方差仍能给出非零的不确定度。堆叠权重通过非负岭回归在折外预测上拟合，保证 σ 区间以正确比例覆盖真值。核函数选 Matérn-5/2，因为 PV 评分在描述符空间是二阶可导但不是无穷光滑——Matérn-5/2 正好匹配这种光滑度。

---

## Slide 14 — UCB acquisition (UCB 采集函数)

**EN —** The motivating question is: under a finite DFT budget — say we can run 50 more candidates — which 50 will teach the model the most? Two extremes are wrong. Picking only the highest-μ candidates exploits what we already know but never explores. Picking only the highest-σ candidates explores wildly but ignores what we have learned. UCB combines both: α(x) = μ(x) + 2σ(x). A candidate scores highly if it's *likely* a winner, OR if we are *uncertain* enough about it that it might be a hidden winner. The key design choice is decoupling μ and σ across two models: μ from HistGBR (the most accurate mean predictor), σ from the GPR + RF stack (the calibrated uncertainty source). If a single model produced both, the regularisation that calibrates σ would degrade μ, and we would lose accuracy where it matters most. Top-N by α become the next DFT batch; returned labels feed back into both models.

**中文 —** 出发点是：DFT 预算有限——比如还能算 50 个候选——哪 50 个能让模型学到最多？两个极端都不对。只挑均值最高的，是在剥削已有认知，从不探索；只挑方差最大的，是盲目探索，不看已有结论。UCB 把两者结合：α(x) = μ(x) + 2σ(x)。一个候选得高分要么*可能*是赢家，要么*不确定到*可能藏着赢家。关键设计是把均值和方差跨两个模型解耦：μ 来自 HistGBR（最准的均值预测器），σ 来自 GPR + RF 堆叠（校准的不确定性源）。如果用同一个模型同时给 μ 和 σ，校准 σ 的正则化会拖累 μ，导致最关键的均值精度下降。按 α 排序的前 N 个进入下一轮 DFT，返回的标签同时反哺两个模型。

---

## Slide 15 — Validation under random and grouped CV (随机与分组交叉验证)

**EN —** Validation. Five-fold by five-seed random CV on the 473-material set: HistGBR achieves R² = 0.466 ± 0.035, MAE = 0.012, top-20 recall 0.50. That is +50 % over our prior GPR / GB / ExtraTrees baseline. Ablations: GBR 0.399, ExtraTrees 0.234, MEGNet fine-tuned 0.307, MEGNet frozen 0.366. The honest story shows in grouped CV: prototype-grouped R² collapses to roughly zero — extrapolation to unseen prototypes is poor — and chalcogen-family-grouped R² is around 0.15, with selenium the hardest hold-out.

**中文 —** 验证结果。5 折 × 5 随机种子 CV：HistGBR 的 R² = 0.466 ± 0.035，MAE = 0.012，top-20 召回率 0.50；相对此前 GPR / GB / ExtraTrees 基线提升 50 %。消融：GBR 0.399，ExtraTrees 0.234，MEGNet 微调 0.307，冻结 0.366。真实情况体现在分组 CV：按结构原型分组 R² ≈ 0，对未见原型外推能力差；按硫族家族分组 R² ≈ 0.15，硒最难。

---

## Slide 16 — Per-group residual audit (分组残差诊断)

**EN —** This is where we keep ourselves honest. Random CV on the previous slide gave R² = 0.47, which sounds great. But "averaged across the dataset" hides the question: where is the model still failing? We split the test residuals by chemistry, by structural prototype, by score range — and look at each group separately. The priority score combines two things we care about: RMSE × √n (which groups have most error) plus 0.02 × top-20 misses (which groups lose us actual winners in the ranking). Selenium is the highest priority — RMSE 0.042 across 182 samples and 9 of the 17 true top-20 missed. The high-score bin y > 0.1 has RMSE 0.145, three times the global average — confirming we under-fit exactly the tail we want to predict. ABC₃ has only 56 samples and a 50 % top-20 miss rate. These three groups directly tell the next DFT batch what to compute.

**中文 —** 这一页是诚实之处。上一页随机 CV 给的 R² = 0.47 听起来不错，但"平均到全数据集"会掩盖一个关键问题：模型究竟在哪里还没学好？我们把测试残差按化学家族、结构原型、得分区间分组，一组一组单独看。优先级分数综合了两个我们关心的指标：RMSE × √n（哪些组误差最大）+ 0.02 × top-20 漏检数（哪些组在排序时漏掉了真正的赢家）。硒族优先级最高——182 个样本 RMSE 0.042，17 个真 top-20 漏掉 9 个；高分区 y > 0.1 的 RMSE 0.145，是全局均值的三倍，说明我们对最关心的尾部欠拟合最严重；ABC₃ 只有 56 个样本，top-20 漏检率高达 50 %。这三个组直接告诉下一轮 DFT 应该补哪些样本。

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

**EN —** Why build our own framework rather than using SCAPS or other established tools? Because the existing tools are 1D-only, treat ions as a steady-state hack, use Beer–Lambert optics, and are GUI-driven — none of which we can wire into a high-throughput active-learning pipeline. SolarLab is built around four design choices. First, a Python backend with FastAPI server-sent events so each J–V sweep streams its progress to whatever client invokes it — useful for both interactive UI and automated batches. Second, a TypeScript / Vite frontend for publication-quality figures during interactive exploration. Third, a Radau implicit Runge–Kutta solver with BDF fallback and bisection-in-time on stiff steps — necessary because the drift-diffusion + ion + Poisson system is genuinely stiff near flat-band. Fourth, the framework is open-source and REST-driven so the same engine that produced today's hero plot can be called programmatically from the active-learning loop.

**中文 —** 为什么不用 SCAPS 或其他成熟工具，而要自建？因为现有工具是 1D-only、用稳态近似处理离子、光学是 Beer–Lambert、GUI 驱动——这些都没办法嵌进高通量主动学习流水线。SolarLab 的四个设计选择：第一，Python 后端 + FastAPI SSE，让每次 J–V 扫描的中间结果实时推送给调用方——既支持交互式 UI 也支持自动批处理。第二，TypeScript / Vite 前端，方便交互式探索并出版级出图。第三，Radau 隐式 Runge–Kutta + BDF 备选 + 时间二分细化——因为漂移扩散 + 离子 + Poisson 系统在平带附近非常硬，必须用隐式格式。第四，整个框架开源、REST 驱动，使得今天讲的那张图所用的引擎，能够直接被主动学习循环以编程方式调用。

---

## Slide 23 — Drift-diffusion governing equations (漂移扩散控制方程)

**EN —** The output we care about is the J–V curve — V_oc, J_sc, FF and PCE. To predict those, we need to know where carriers are, how fast they move, and how many recombine before reaching the contacts. That's exactly what these three equations track. Continuity says: at any point, the carrier density changes because of currents flowing in or out, plus generation, minus recombination. The current density itself has two terms: drift, which is carriers pushed by the electric field, and diffusion, which is carriers spreading from high to low concentration — the Einstein relation D = (k_B T / q) μ ties the two together. Poisson's equation closes the loop: charge density determines the field, the field drives drift. Generation G comes from TMM optics, which we'll see next; recombination R sets V_oc through the Shockley–Read–Hall, Auger and radiative channels.

**中文 —** 我们最终关心的输出是 J–V 曲线——V_oc、J_sc、FF 和 PCE。要预测这些，需要知道载流子在哪里、跑多快、有多少在到达接触前复合。这三组方程就是回答这三个问题的。连续性方程说：任意一点的载流子密度变化，等于流入流出电流之差，加上产生减去复合。电流密度本身分两部分：漂移（电场推动）和扩散（高浓度往低浓度扩散），由 Einstein 关系 D = (k_B T / q) μ 连接。Poisson 方程把环路闭合：电荷密度决定电场，电场驱动漂移。产生项 G 来自下一页要讲的 TMM 光学；复合项 R 通过 SRH、Auger、辐射三个通道直接决定 V_oc。

---

## Slide 24 — Transfer-matrix-method optics (TMM 光学)

**EN —** Why not just Beer–Lambert? In a thin-film stack — say a 400-nm absorber sandwiched between an HTL and an ETL — light reflects back from each interface and interferes with itself. That creates a standing wave inside the absorber, with peaks and troughs at specific wavelengths. Beer–Lambert assumes simple exponential decay, missing those interference fringes; in 200–400 nm absorbers it under-counts photogeneration by 10–20 %. TMM solves the full problem: each layer becomes a 2×2 matrix that propagates the electric and magnetic field components forward; the system matrix is the product over all layers; from that we read off wavelength-resolved reflectance, transmittance and absorption directly. The carrier-generation profile G(x, λ) is proportional to |E(x, λ)|² × α(λ), integrated over AM1.5G — and this is what feeds the G term in the previous slide's drift-diffusion equations.

**中文 —** 为什么不用 Beer–Lambert？在薄膜叠层里——比如 400 nm 吸收层夹在 HTL 和 ETL 之间——光在每个界面反射、与自己干涉，在吸收层内部形成驻波，特定波长处出现波峰和波谷。Beer–Lambert 把这一切简化为指数衰减，干涉条纹完全丢失；在 200–400 nm 吸收层中会低估光生 10–20 %。TMM 求解完整问题：每层用 2×2 矩阵把电磁场分量前推，系统矩阵 = 各层矩阵之积；从中可直接得到波长分辨的反射、透射、吸收谱。光生剖面 G(x, λ) ∝ |E(x, λ)|² · α(λ)，再在 AM1.5G 光谱上积分——正是上一页漂移扩散方程里 G 项的输入。

---

## Slide 25 — Mobile-ion transport (离子迁移)

**EN —** Why does this slide exist? Because of one experimental fact: when you measure a perovskite J–V curve, the forward sweep and the reverse sweep do not match — and the gap between them depends on how fast you scan. That hysteresis is the smoking gun for mobile ions. A simulator without ion transport gives you exactly one J–V curve regardless of scan rate, and cannot fit the data. So we add a continuity equation for ion vacancies. The key term is the steric factor (1 − P / P_lim): mobile ions cannot pile up beyond the available lattice sites — without this cap they would run away to one contact. Under bias, ions drift, accumulate at one electrode, screen the internal field, and change the recombination rate. The feedback into Poisson is the source term ρ_ion. Characteristic time scales of 0.01 to 100 seconds — set by the ion mobility — reproduce the measured scan-rate dependence.

**中文 —** 这一页存在的根本原因是一个实验事实：测钙钛矿 J–V 曲线时，正向扫描和反向扫描不重合，而且不重合的程度跟扫描速率有关。这种迟滞就是离子迁移的"铁证"。一个不含离子迁移的仿真器，无论扫多快都给同一条 J–V 曲线，根本拟合不了实验。所以我们加入离子空位的连续性方程。关键是饱和因子 (1 − P/P_lim)：可动离子的密度不能超过可用晶格位——没有这个上限，它们会全部跑到一边的接触面去。偏压下离子迁移、在某一极聚集、屏蔽内电场、改变复合速率；通过源项 ρ_ion 反馈到 Poisson 方程。由离子迁移率决定的 0.01–100 s 特征时间尺度，正好再现了实测的扫描速率依赖。

---

## Slide 26 — Robin selective-contact boundary conditions (Robin 选择性接触边界)

**EN —** Why care about contact boundary conditions? Because they set how easily carriers leave the absorber, and that directly affects V_oc. Real measured surface-recombination velocities span six orders of magnitude — from 0.1 to 10⁵ metres per second. SCAPS and similar 1D solvers usually use Dirichlet pinning, which forces carrier density at the contact to its equilibrium value. That is the S → ∞ limit only — perfectly extracting contact, infinite velocity. Real contacts are nowhere near that limit. We use Robin boundary conditions: the current at the contact is proportional to S × (carrier density − equilibrium density). Two velocities per contact — S_n for electrons, S_p for holes — and selectivity is built in by making S_n small and S_p large at the HTL, mirrored at the ETL. The S → ∞ limit recovers Dirichlet pinning, and S → 0 gives a perfectly blocking Neumann contact, so Robin smoothly interpolates between every physically relevant case. This is one of the capabilities listed on the next slide as absent from SCAPS.

**中文 —** 为什么接触边界条件重要？因为它决定载流子离开吸收层的难易程度，直接影响 V_oc。实测表面复合速度跨越六个数量级——从 0.1 到 10⁵ m/s。SCAPS 等 1D 求解器通常采用 Dirichlet 钉扎：把接触处载流子密度强制等于平衡值，这只对应 S → ∞ 极限——完美提取、无穷速度。真实接触远远达不到这个极限。我们用 Robin 边界：接触电流正比于 S × (载流子密度 − 平衡密度)。每个接触有两个速度——电子的 S_n、空穴的 S_p——通过让 HTL 的 S_n 远小于 S_p（ETL 镜像）天然实现选择性。S → ∞ 退化为 Dirichlet 钉扎，S → 0 给出完全阻断的 Neumann 边界，Robin 在两个极限之间平滑插值，覆盖所有物理相关情形。这正是下一页列出的、SCAPS 缺失能力之一。

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
