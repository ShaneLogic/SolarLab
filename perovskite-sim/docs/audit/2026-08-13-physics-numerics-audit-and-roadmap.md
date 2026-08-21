# SolarLab 物理-数值审计与研究路线图（2026-08-13）

## 1. 执行摘要

本报告基于三层证据进行综合审计：其一，代码级锚点与局部源文件核读；其二，审计摘要与复核摘要（AUDIT/VERIFY digest）；其三，文献映射与创新评审面板。证据分级采用三档：**CONFIRMED** 表示代码事实单向成立，或复核票数一致；**待验证（PLAUSIBLE）** 表示高影响结论存在分歧、票数偏薄、或“代码事实成立但后果强度未证成”；**已否证（REFUTED）** 表示复核已明确推翻原始强说法。后文每一条核心发现均显式标注证据等级。

**[CONFIRMED] 总判定 1：SolarLab 当前最准确的定位，是“物理覆盖面很广、但只在部分研究类别上达到研究级可用”的前向模拟器，而不是一个在所有边界条件、所有实验协议、所有器件类别上都热力学闭环、数值可认证、可直接做参数反演的通用平台。** 它的 1D Scharfetter–Gummel 漂移-扩散主干、Poisson 预分解、光生载流子预算守恒修复、以及多条实验模块已经达到很高工程成熟度；但在接触热力学闭环、活性路径上的陷阱电荷电静力、自洽低频离子阻抗、2D 离子/界面物理、以及反演与高通量基础设施方面，仍存在会直接限制论文级主张的结构性缺口。

**[CONFIRMED] 总判定 2：最强可发表赛道不是继续追逐“SCAPS 幅值完全一致”，而是承认现有校准脚手架后，把优势迁移到 SolarLab 独有的资产组合上。** 这组资产包括：相干 TMM 光学、严格的光子预算守恒、丰富的界面平面态/界面重组实验分支、移动离子瞬态、以及一个已经存在但尚未离子化的小信号频域线性代数内核。与 SCAPS、IonMonger、Driftfusion、ChargeFabrica、∂PV 等相比，SolarLab 真正有机会产出新方法论文的地方，在于“把这些资产接起来并给出数值有效性证书”，而不是在单点上补齐某个早已存在于文献中的公式。

**[PLAUSIBLE] 总判定 3：未来一轮研究路线应停止以“绝对幅值拟合 SCAPS”为主要牵引，而应按‘可认证离子谱学 → 瞬态 DAE + 代数界面态 → 可辨识性/反演’的顺序推进。** 这是因为，当前最有把握、最短路径、且最能回答真实器件问题的创新方向，是把现有的瞬态离子物理与频域认证框架合并成一个可认证的离子阻抗平台；而更长线、也更有潜在方法学影响力的方向，则是把目前仅在稳态路径上出现的界面平面态物理提升为瞬态代数态，再进一步通过精确 Jacobian/IFT/并行评估进入可辨识性反演。

## 2. 物理模型全景

**[CONFIRMED] SolarLab 的前向物理覆盖面本身非常宽。** 1D 主干采用方法线（Method of Lines）框架，空间离散是 Scharfetter–Gummel 有限体积/有限元混合表达，时间推进使用 `solve_ivp(Radau)`。Poisson 方程在每次 RHS 调用内通过三对角 LU 因式分解求解，而不是在时间状态中显式保留电势变量。移动离子、双离子、位点排斥（lattice-gas / steric）修正、TMM 相干光学、光子回收与自洽再吸收、场依赖迁移率、温度标度、空间陷阱分布、连续带隙渐变、选择性/Robin 接触、以及多种异质结界面重组闭合都已存在于代码树中。这一“资产密度”明显高于多数单一目标代码。

**[CONFIRMED] 其中真正已经做扎实的，是守恒结构。** 电子/空穴方程在体重组项上始终以同一 $R$ 成对扣除，因而电荷守恒是结构性的；离子方程在零通量边界下采用望远镜式离散，因而总离子数守恒不是“趋近”，而是离散结构保证；2026-07 修复后的 Beer–Lambert 与 TMM 两条光学生成路径，均已改写为“每个 dual cell 接收其自身精确吸收的光子数，再除以 RHS 回乘的同一个权重”，因此光生预算同样是结构性守恒，而非网格细化后才近似成立。

**[CONFIRMED] 但 constitutive closure（本构闭环）不完整。** 目前体输运与接触平衡仍然是纯 Maxwell–Boltzmann 假设：没有块体 Fermi–Dirac 统计、没有带隙窄化、没有不完全电离；外在寄生元件方面也没有串联电阻、并联泄漏、自热或热-电耦合。对于薄膜趋势研究，这些缺失并不一定使所有结论失效；但对于高掺杂 c-Si 发射极、器件填充因子、真实功率转换效率映射、以及任何要求“由材料本征参数直接预测接触/发射极行为”的研究，这些就是一阶缺项，而不是二阶修饰。

**[CONFIRMED] 陷阱能态模型的物理丰富度低于它在输入层面看起来所暗示的程度。** 代码支持单一离散 SRH 陷阱层级，并且可以在空间上给出近界面浓度轮廓；但 `distribution: gaussian`、`E_char_eV`、`N_peak_cm3` 这类输入目前只是“被验证但不参与后续求解”的信息字段，真正的块体与界面重组仍只读取一个等效单能级。这意味着当前的“高斯能量分布界面缺陷”在物理上并未落地，只是 YAML 模式上看起来像存在。

**[CONFIRMED] 2D 分支不是 1D 物理的横向完全延拓，而是一个侧重“横向均匀性/简化微结构”验证的受限分支。** 它确实有 2D Poisson、2D SG 通量、Robin 接触、2D 场依赖迁移率、2D 辐射再吸收，以及晶界寿命图样；但它没有移动离子自由度，界面重组也只走 bulk SRH/rad/Auger 汇总，完全没有 1D 的界面平面态/界面缺陷通道。因此，“2D 已经具备全部 perovskite 关键物理，只差更多几何形状”这一说法不成立。

## 3. 完备性与闭环性审计

**[CONFIRMED] 体方程的守恒闭合优于边界与界面的热力学闭合。** 换言之，SolarLab 最强的是“在给定一套数组之后，把 PDE 离散得非常规整”；最弱的是“这些数组与边界数据是否来自同一个热力学世界”。这一差异解释了为什么代码内部会出现大量“趋势是可信的、绝对幅值和某些极端边界不是”的自我注记。

**[待验证（PLAUSIBLE）] 接触热力学不相容问题是真问题，但原始最强说法需要收窄。** 审计初稿中最强的说法是：Poisson 接触势 `V_bi_bc` 与四个载流子接触库 `n_L,p_L,n_R,p_R` 是相互独立的自由参数，因此在绝大多数 preset 上都不存在真正的离散热平衡；复核阶段认为“这一说法的代码核心成立，但其适用范围和严重度被夸大”。保守而可靠的重述应是：在 legacy/manual 接触路径上，Poisson 接触势幅值与接触载流子库并不在构建阶段被统一约束，运行时也没有系统级 fail-close 检查；虽然后续测试中确实存在一个专门的热平衡一致性断言，但它不是通用的配置装载器/求解器前置证书。这意味着当前代码更适合被视作“带兼容性护栏的工程实现”，而不是“自动生成热力学自洽边界态的求解器”。

**[CONFIRMED] 活性路径上的陷阱电荷电静力仍然缺失。** 这是本报告最关键的物理闭环缺口之一。`rho = q(p-n+...)` 的主空间电荷装配没有块体 SRH 占据电荷项；界面陷阱片电荷虽然有代码脚手架（`iface_state_charge` 与相应的 Poisson 加项），但当前活动路径上并不会赋值，因而等价于死代码。换句话说，当前的界面缺陷 sweep 可以改变重组速率，却不能通过占据电荷去反向塑造势垒形状与局域场分布。这正是“为什么某些 Nd_ETL/CBO 幅值总是调不对，而方向却能做对”的最合理已知解释之一。

**[CONFIRMED] 界面平面态/共享占据/双边通道等界面重组家族，已经形成了很有价值的‘否证链’。** 这在研究方法上是优点而不是缺点。代码与笔记明确记录：单纯在 bulk-node 密度上改写代数占据形式，并不能修复 SCAPS 残差；共享占据在若干路径上已被直接测成 no-op；真正能动起来的是带 TE 投影的界面平面态闭合，以及其稳态代数态扩展。但与此同时，这些更“有效”的界面路径又依赖 `iface_state_calibration_factor` 这类拟合标尺，因此现阶段应将其定义为“用于方向与相对排序的半经验界面物理”，而不是“完全从材料参数出发的可预测界面模型”。

**[待验证（PLAUSIBLE）] `het_recomb_despike = 0.53` 与界面校准因子不应被直接称为‘错误’，但必须被明确标识为校准脚手架。** 代码事实是单向成立的：这些参数确实存在、确实只在特定路径上起作用、确实是为对齐某套参考趋势而调入的。更强的说法——例如“它们在所有异质结节点上都破坏详细平衡，并且因此使全部结论无效”——复核并未一致支持。因此更好的政策是：不把它们妖魔化，也绝不继续用“matched by physics”掩盖它们的存在，而是把它们标记为“已知校准支架，适用于当前 lane，不能外推为一般第一性原理界面闭合”。

**[CONFIRMED] 2D 分支在闭环性上远弱于 1D 主干。** 缺移动离子、缺界面重组、缺几何-材料-ID 涂覆层，这意味着它现在适合作为“横向均匀性与少量寿命异质性假设”的研究工具，而不是“完整 2D perovskite 物理平台”。凡是需要晶界离子聚集、界面电静力、异质结横向复合、或真实介观形貌的研究，2D 现版本都不构成完整前向模型。

## 4. 数值算法稳定性审计

**[CONFIRMED] SolarLab 的离散内核本身是强项。** Bernoulli 函数有小量 Taylor 分支和大参数保护；Poisson 因式分解缓存成熟；2026-07 的光生预算修复把 Beer–Lambert 与 TMM 两条路径都提升到了“网格无关的守恒式实现”；前向 J–V 路径也已补上近 flat-band 的步长上限、分裂重试、错误分支拒绝、BDF 尾部补救等多层防护。这些都说明问题不在“数值作者不懂 stiff drift-diffusion”，而在“求解器控制律与状态缩放策略还不够系统”。

**[CONFIRMED] 当前最大的数值政策缺口，是统一对待一个跨度极大的 SI 密度状态，并只给一个标量 `atol=1e-6`。** 这一事实本身没有争议。争议在于它是否已经明确污染了当前所有报告量；复核结论是：强后果尚未被严格证成，但作为 stiff ODE 控制策略，它显然不够防御性。更准确地说，SolarLab 现在缺的不是“一个更小的 rtol”，而是按物种与参考尺度构造的 componentwise absolute tolerance，以及一套真正被执行过的 tolerance-refinement 实验。

**[CONFIRMED] 暂态载流子路径缺少 positivity mechanism。** 体 SRH 分母没有下界，Auger 在负密度方向上会变号，且当前并无对 interior `n,p` 的投影/对数态强制机制。复核也指出，某些界面路径确实会在局部临时夹断负密度，但这不能抵消“主暂态载流子自由度整体上无正性保障”这一事实。对趋势级模拟，这未必立刻产生灾难；对稳健性、可复现性与极端边界扫描，则是必须明确承认的风险源。

**[CONFIRMED] RHS 非光滑性是真问题，而且不止一个拐角。** TE cap 的硬 `min()`、Poole–Frenkel 的 $\sqrt{|E|}$ 尖点、界面某些限幅/钳制、以及位点拥挤的对数项，都把非光滑算子直接放到了 Radau/Newton 的 RHS 上。这里并不意味着“结果一定错”，而是意味着求解器的经典光滑误差模型与牛顿局部收敛假设，不再能被不加修正地解释为“严格理论保证”。这也是为什么 SolarLab 代码中会反复出现“这条路径在某些极端 regime 下要靠经验性 guard/fallback 才能过”的原因。

**[CONFIRMED] 稳态 Newton 的 `converged=True` 语义目前过宽。** 代码允许在 stall-acceptance 条件下返回 `converged=True`，而 residual 又是用全器件峰值密度缩放。这一设计并非毫无道理——它显然是为了解决暗耗尽区的局部相消噪声——但从科研报告语义上说，`converged`、`stall-accepted but current-bounded`、`transient-assisted point` 最好分开，而不是共享同一个布尔终点。否则外层分析会过度解读一个其实已经过人手工程判据筛选的状态。

**[CONFIRMED] “求解器很贵”这件事已经不是猜测，而是被代码结构解释清楚的事实。** 目前暂态 Radau 与稳态 Newton 都在大量依赖 dense finite-difference Jacobian；而 Poisson 又是在每次 RHS 中全局消去，因此真实 Jacobian 对状态是全耦合的。复核明确收窄了一条常见误解：`jac_sparsity` 不是现有 1D ODE 形态下的直接解药；真正有希望的方向，要么是解析 Jacobian/Schur 结构利用，要么是直接转向保留代数态的 DAE/formulation。

## 5. 已验证 vs 仅声明

**[CONFIRMED] 已被真正验证、且可放心写进方法部分的内容，主要集中在守恒与局部 lane 认证，而不是绝对外部一致性。** 第一，光生预算守恒已被修到结构级：Beer–Lambert 与 TMM 两条路径都不再允许“吸收到的光子比进入器件的还多”。第二，2D Stage A 的“横向均匀极限必须回到 1D”已经有严格回归门；这使 2D 至少在其被定义的适用域内不是随意拼装。第三，`reproducibility/config_benchmark_matrix.yaml` 所代表的“把 claim level、evidence tier、limitations 显式写出来”的做法是真正可信的科研工程实践，它比许多学术代码的验证文化都更成熟。

**[CONFIRMED] 与之相对，‘SCAPS 基线由物理自然命中’这一叙述不能再使用。** 一方面，旗舰 parity 配置确实携带了多枚拟合/脚手架参数；另一方面，2026-07 的光生预算修复之后，原先看起来更接近 SCAPS 的 J_sc 其实部分来自“超过吸收光子预算的数值盈余”。换句话说，当前 SCAPS 对齐最诚实的表述应是“带已知校准支架的趋势对齐与部分基线对齐”，而不是“由同一组物理无拟合命中”。

**[CONFIRMED] 两条‘paper reproduction’ lane 本质上是 calibrated reproduction，不是独立外部验证。** 这并不是丑闻；它只是决定了这些 lane 在论文里该放在哪里。它们适合被写成“回归锚点”或“cross-code calibration checks”，不适合被写成“外部实验/文献验证的核心证据”。如果要做真正的外部验证，应优先采用文献中已给出全参数、不可随意拟合的 analytic benchmark 或 protocol-level cross-code test。

**[REFUTED] “网格收敛只在一个 preset 上做过”这一最强说法已被否证。** 更准确的版本是：收敛证据覆盖面仍然偏窄，但不止一个 preset，且 c-Si 线路已有更严谨的局部认证。这个修正非常重要，因为它说明 SolarLab 的问题不是“完全没有收敛文化”，而是“收敛文化没有系统覆盖到最该覆盖的类”。

**[CONFIRMED] 还有一类必须明说的“仅声明未实测”：协议层与动态实验层。** 例如阻抗、TPV、hysteresis 中的预偏置、驻留时间、循环历史，目前多为各模块内部隐式默认，而不是一个显式可组合的 protocol 对象。因此即使前向求解器本身可信，实验叙事也可能因为“默认前史”而失去可复现性。对真实器件研究，这是方法学层面的缺口，而不是实现细节。

## 6. 科研适用性判定表

| 研究类别 | 当前判定 | 证据等级 | 说明 |
|---|---|---:|---|
| Class 1：perovskite J–V hysteresis / 扫速研究 | READY-WITH-CAVEATS（趋势级） | **CONFIRMED** | 瞬态 J–V 会保留离子记忆，但缺统一 protocol 层，缺系统扫速验证与 soak/pre-bias 组合。 |
| Class 2：Impedance / IMPS / IMVS | NOT-READY（离子器件） | **CONFIRMED** | 离子可见路径是慢的 transient lock-in；认证频域路径拒绝离子；默认频段又偏离 LF ionic branch。 |
| Class 3：界面重组 / passivation | READY-WITH-CAVEATS（可排位，不可报幅值） | **CONFIRMED** | 方向、排序与机制筛选可做；活性路径无陷阱电荷电静力，幅值仍依赖校准脚手架。 |
| Class 4：band alignment / CBO-VBO | READY-WITH-CAVEATS（最强 lane） | **CONFIRMED** | 这是当前最可信的研究类别，但 deep-cliff / spike / tunneling 等边界仍有已知引擎极限。 |
| Class 5：optical design / TMM / thickness | READY-WITH-CAVEATS | **CONFIRMED** | 相干光学和光子预算很强；但 transport-layer parasitic vs collectable current 分辨、EQE 量化、graded optics 仍不足。 |
| Class 6：tandem design / current matching | READY-WITH-CAVEATS（光学强，结物理弱） | **待验证（PLAUSIBLE）** | 顶/底电池生成分配与串联匹配可做；结区、寄生、PR/rr 对实际子电池 lane 的活性仍需更强验证。 |
| Class 7：CIGS graded bandgap | NOT-READY（完整 graded-absorber 研究） | **CONFIRMED** | 电学渐变已实现，但光学 `alpha(λ,y)`/`n,k(y)` 未分级；不能支持真正 notch 优化主张。 |
| Class 8：c-Si / degenerate-doping | NOT-READY（发射极/Voc/J0 物理） | **CONFIRMED** | 缺块体 FD、带隙窄化、不完全电离。受限 lane 可算，但不适合把结果解释成发射极物理。 |
| Class 9：2D grain boundary / microstructure | NOT-READY | **CONFIRMED** | GB 有效面积由网格决定且与 grain size 脱钩；2D 无移动离子，无界面重组。 |
| Class 10：degradation / stability | NOT-READY（作为物理模型） | **CONFIRMED** | 现有退化是 phenomenological proxy，不是化学/热/湿/场耦合降解物理。 |
| Class 11：parameter extraction / inverse problems | NOT-READY | **CONFIRMED** | 无 inference、无 posterior、无 adjoint/AD、无 UQ、无并行。 |
| Class 12：high-throughput screening / optimization | 有限可用，不足以称研究级高通量平台 | **待验证（PLAUSIBLE）** | 现有随机搜索/串行 sweep 可做 advisory scouting，但不是 DOE+并行+surrogate 的优化平台。 |
| 额外：EQE / spectral response | NOT-READY（定量） | **CONFIRMED** | 现实现象学上可用，但量化解释仍受 operating-point 与 generation representation 限制。 |
| 额外：温度依赖研究 | USABLE FOR TRENDS ONLY | **CONFIRMED** | 温度标度已有，但 carrier statistics / interface temperature consistency 仍受限。 |

## 7. 必修项 (P0)

**[CONFIRMED] P0-1：把“阻抗是一个功能”改成“阻抗是一个带认证前置条件的实验”。** 最小交付物不是新物理，而是三件事：一套残差认证的离子 DC operating point；自动或至少告警式的频率窗口选择；以及明确区分“离子可见但慢”和“快速但拒绝离子”的两条路径。只要这一步不做，任何 sub-Hz ionic impedance 论文都缺可信起点。

**[CONFIRMED] P0-2：对活性路径上的界面陷阱电荷采取二选一政策：要么真正接通并校正 gauge，要么把它公开降级为 parked feature。** 当前最危险的不是“没有这个物理”，而是“代码里像有，论文叙事里像在逼近，实际上活性路径上没有”。这一点必须在代码、UI、文档、审计表述中一致。

**[CONFIRMED] P0-3：2D 物理主张必须收缩到它当前真正覆盖的范围。** 如果短期内不打算给 2D 加移动离子和界面重组，那么文档、工作站文案、以及任何对外报告都应把它限定为“横向均匀性/寿命图样研究器”，而不是“完整 perovskite 2D 微结构模拟器”。这不是营销问题，而是研究伦理问题。

**[CONFIRMED] P0-4：CIGS/graded-bandgap 叙事必须从“graded device physics”收缩到“graded transport only”，除非光学分级尽快实现。** 现有 `cigs_graded_notch` 适合做“电学渐变影响 Voc/重组”的实验，不适合做“notch 优化 Jsc-PCE”的主结果。若近期不做 graded `alpha/nk`，则应在所有面向研究者的输出里明确限制。

**[CONFIRMED] P0-5：建立 componentwise tolerance policy 与最小容忍度研究，而不是继续围绕全局 `rtol/atol` 做经验微调。** 这是目前最值得做、也最便宜的数值基础设施修补之一。它未必立刻改变所有最终数值，但会显著提升“当前结果什么时候可信、什么时候只是 solver policy 的产物”这一点的可解释性。

**[待验证（PLAUSIBLE）] P0-6：给接触热力学相容性增加 runtime 级 guard 或至少 evidence label。** 复核已经否定了一些过强说法，但并未证明当前 legacy/manual 路径就是严格自洽的。最佳策略不是先在论文里下重判，而是在构建和求解入口就告诉用户：当前 deck 属于“有测试覆盖/无测试覆盖/仅兼容性保持”的哪一类。

**[CONFIRMED] P0-7：把所有“matched by physics”“reproduced”改写为分层证据语言。** 对已经校准过的 SCAPS/Driftfusion/Courtier 参考，应使用“calibrated reproduction”“trend-aligned lane”“pinned regression anchor”等术语；对真正有独立证据的 lane，再使用“validated”或“certified”。这一步几乎零开发成本，却能显著降低论文被抓住措辞漏洞的风险。

## 8. 创新路线图

**[CONFIRMED] 近中期综合排序第 1 位：Certified ion-aware small-signal impedance。** 这是三组 judge panel 唯一都给出高排名的方向之一，也是短期 feasibility 与 scientific impact 最平衡的选项。它不是去“重新发现 impedance zoo”，而是把 SolarLab 现有的两半——离子可见的 transient AC 与有数值证书的小信号线性代数内核——接成一个可以对每个频点报告条件数、后向误差和导纳连续性证书的离子谱学引擎。若写得好，这是标准的 methods paper，而非功能说明书。

**[CONFIRMED] 综合排序第 2 位：Transient DAE SolarLab（移动离子 + 代数界面态）。** 这条线风险高，但如果成功，它将是 SolarLab 最具“平台跃迁”意义的工作。创新点不在于 mass-matrix Radau 本身——那是现成 prior art——而在于它把当前只出现在稳态路径上的界面平面态/选择性接触/小信号认证，带入同一个瞬态状态拓扑中，使 hysteresis、TPV、impedance 终于能在同一套 richer interface physics 上运行。

**[CONFIRMED] 综合排序第 3 位：Electrostatically self-consistent interface-state Poisson closure。** 这条线在三个 judge lens 中有两个进入 top-3，尤其受“PV 器件物理学家视角”看重。它不是再换一个重组公式，而是补上“界面态既改重组，也改电势障”的缺失回路。一旦接通，它最有希望解释当前 Nd_ETL/CBO 幅值为什么长期只能靠 rate-scaling 去补。

**[待验证（PLAUSIBLE）] 综合排序第 4 位：identifiability-focused inverse-problem layer。** 这不是“为了时髦而上 AD/MCMC”，而是把 SolarLab 自己的伤口变成研究问题：`iface_state_calibration_factor`、`het_recomb_despike`、各类界面 trap knob，到底哪些是可由 J–V/scan-rate/TPV/impedance 识别的，哪些其实只是结构性模型缺失的补丁？只要这个问题被严肃做成 identifiability paper，即便答案是否定的，也有发表价值。

**[待验证（PLAUSIBLE）] 第二梯队方向包括 generalized-SG beyond MB、differentiable tandem optimization、以及 falsification-driven validation infrastructure。** 这些都值得做，但它们要么更偏“补课”，要么对当前 repo 的最紧迫研究瓶颈没有前四项直接。

## 9. 推荐的下一步

**[CONFIRMED] 未来 4–6 周的最优动作序列应当是“先清理可信度，再造新能力”。** 第一组动作是低风险高收益的可信度修补：为阻抗补残差认证 operating point 与频带告警；把 2D、CIGS grading、SCAPS parity 的文案与 evidence tier 对齐；把 componentwise tolerance 研究列为一个独立的数值基线任务；公开声明哪些 feature 是 parked/dead code，避免使用者误会。完成这一步后，SolarLab 会从“知道自己哪里不稳”提升到“能把不稳之处明确呈现给用户”。

**[CONFIRMED] 未来 2–4 个月，最应该同时推进两条互补线：一条偏实验能力，一条偏界面物理。** 实验能力线是离子可见的小信号阻抗适配器；界面物理线是活性路径上的陷阱电荷电静力闭合。这两条线彼此增强：前者为动态器件研究提供高价值输出，后者为异质结幅值问题提供更接近物理的解释变量。

**[PLAUSIBLE] 未来 6–12 个月，若团队愿意承担架构重写成本，再进入 DAE/adjoint/inference 长线。** 这一步不应被当成“自然下一步”，而应当被当成“在前两阶段确实产出高价值证据后，值得启动的平台换代计划”。一旦开始，就应明确把目标写成“统一 transient ions、algebraic interface states 与 certified small-signal outputs”，而不是“实现一个新的 Radau”。

**[CONFIRMED] 若研究管理上需要一个一句话排序，我建议：先做 certified ionic spectroscopy，再做 interface-charge closure，最后决定是否投入 transient DAE backbone。** 这条路径能最大程度利用现有代码资产，又不会在最早期就把项目推入高风险 solver rewrite。

## 10. 附录

### 附录 A：证据分级规则

**[CONFIRMED]** 用于以下情况：代码事实单向成立；或 VERIFY digest 中确认票数一致且无高质量反驳。例如“2D 状态只含 `(n,p)`”“反演/并行基础设施不存在”“块体输运没有 FD/带隙窄化/不完全电离”“degradation law 是 phenomenological proxy”。

**[待验证（PLAUSIBLE）]** 用于以下情况：高影响结论存在复核分歧；或“代码事实成立，但其严重后果尚未被强证成”。例如“接触边界因此完全不存在任何离散热平衡”“全局标量 `atol` 已经污染了当前所有 Voc/Jsc”“`het_recomb_despike` 必然使全部异质结趋势失真”。

**[已否证（REFUTED）]** 用于复核已明确推翻的最强说法。例如“只有一个 preset 做过网格收敛证明”已经被更正为至少两个 lane 存在明确收敛证据。

### 附录 B：本报告最关键的代码锚点（节选）

| 代码锚点 | 读出的事实 | 证据等级 |
|---|---|---:|
| `perovskite_sim/physics/recombination.py:10-25` | SRH 分母无下界，Auger 在负密度方向会变号；体载流子暂态正性无结构保障。 | **CONFIRMED** |
| `perovskite_sim/solver/mol.py:1845-1847` | 主空间电荷 `rho` 只含自由载流子、离子背景偏离与全电离掺杂；无 trap-charge 主项。 | **CONFIRMED** |
| `perovskite_sim/solver/mol.py:212` + `:2185-2195` | `iface_state_charge` 被声明并被读取，但活动路径上无赋值。 | **CONFIRMED** |
| `perovskite_sim/twod/solver_2d.py:657-684` | 2D 状态只含 `(n,p)`，重组只走 bulk SRH/rad/Auger。 | **CONFIRMED** |
| `backend/main.py:925-926` | 阻抗默认频段为 10 Hz–100 kHz；这是默认值，不是物理上不可更改的硬限。 | **CONFIRMED** |
| `perovskite_sim/experiments/quasi_fermi_steady_state.py:412-417` | 认证频域/准费米路径拒绝 mobile ions、nonzero ionic background 等。 | **CONFIRMED** |
| `perovskite_sim/autoloop/search.py:62-87` | 设计搜索默认是 seeded random search，串行 `for` 循环评估。 | **CONFIRMED** |
| `perovskite_sim/scaps_compat/loader.py:486-521` | `distribution: gaussian` 仅被验证，不参与下游 SRH 物理。 | **CONFIRMED** |
| `perovskite_sim/experiments/steady_state.py:426-469` | 稳态 residual 以全局峰值密度缩放，且 `_done()` 一律返回 `converged=True`。 | **CONFIRMED** |
| `perovskite_sim/experiments/degradation.py:142-147, 150-173` | 退化态由两个手调 RMS proxy 累积，并只通过 absorber `tau_n/tau_p` 反馈。 | **CONFIRMED** |

### 附录 C：本报告明确降级或纠正的强说法

**[已否证（REFUTED）]** “网格收敛只在一个 preset 上证明过。” 更准确表述是：覆盖仍偏窄，但并非只有一个；至少 IonMonger 与 c-Si lane 各有明确证据。

**[待验证（PLAUSIBLE）]** “当前所有 legacy/manual 接触 deck 都没有任何离散热平衡。” 更准确表述是：存在 runtime 级缺 guard 的兼容性风险，且最强一致性主张未被系统证明；但并非所有配置都可直接归类为‘毫无离散热平衡’。

**[待验证（PLAUSIBLE）]** “`het_recomb_despike` 一定使所有异质结趋势失真。” 更准确表述是：它是 fitted、且实现方式不保持严格详细平衡，因此应被公开视为 calibration scaffold，而不是被自动提升为普适物理闭合。
