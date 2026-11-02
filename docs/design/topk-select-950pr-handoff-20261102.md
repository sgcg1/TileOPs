# TopKSelectFwdOp · Ascend 950PR 最新交接

交接日期：2026-11-02。面向完全没有本会话上下文的新会话。

**接手结论：环境已打通，H128分组扫描已有Decode净收益，真实阶段诊断已定位首轮histogram/compact约占已测区间81%。后续路线已复评，但首轮优化的新候选尚未实现。下一步从首轮专门化/粗键复用的独立候选开始，不重新部署环境，也不重复跑已完成的探针。最终v0.1.4候选尚未接受。**

本交接覆盖2026-10-31交接中已经过时的状态和权限叙述。尤其不能继续沿用“候选未提交”“不许推送”“只能手动搬B2Y部分文件”“Conda环境”“旧SSH服务器”作为当前前提。历史文档和失败实验保留，不能删除或篡改。

## 1. 我们在做什么任务

将TileOPs的TopKSelectFwdOp适配到Ascend 950PR，并优化其精确选择性能。最终交付使用**TileLang主仓Ascend后端**，不是独立tilelang-ascend适配仓，也不改换Triton路线。

性能基线为官方CANN TopKV2。已有GE静态模型只是调用官方算子的测量入口，不是另写一个自定义TopK冒充基线。

### 接口与必须保持的契约

输入为FP32 scores[B,S,L,G]、int32 starts/ends[B,S]，K为属性；每个(b,s,g)在[start,end)中选择，返回新分配、连续的int32 [B,S,G,K]绝对KV索引。

- 输出槽位无序，**选中集合必须确定**；同分保留较小绝对索引。
- NaN排在正无穷之前，统一NaN payload；±0视作同分。
- 短窗口返回全部有效位置，其余填L；空/反向窗口全部填L。
- 不修改输入，保留非连续输入与分组支持。
- FP16粗键仅作筛选，最终决策仍基于原始FP32完整键；舍入边界、上下溢和特殊值不能近似放行。
- 2048是候选缓存容量，不是K或输入规模限制；溢出必须完整计数并精确重扫。

性能倍数定义为**同口径CANN耗时/TileLang耗时**。小于1表示仍比CANN慢。事件、纯kernel、内部counter区间和多kernel流水包络分别报告，不能混算；没有95% MFU之类的现行验收公式。

### 当前协作条件与授权

| 内容 | 当前约定 |
| --- | --- |
| 实现代码 | https://github.com/sgcg1/TileOPs ，个人fork的main |
| 辅助材料/结果 | https://gitcode.com/JotaroSSR/B2Y ，main |
| 服务器操作 | 新服务器不能由agent连接，只由用户在容器内执行 |
| 同步 | 容器允许直接克隆/更新两个仓库；不再手动拷贝B2Y局部或另建tools目录 |
| 发布权限 | 用户已明确本项目完全开源，授权推送本任务代码、结果、分析及指南；无需反复请求确认 |
| 执行指令 | 用户两台电脑之间无法直接复制粘贴；任何新指令先写B2Y并推送，聊天只给已发布入口和说明 |
| 结果回传 | 直接上传B2Y根目录AAA_Results；不要求用户创建校验文件或额外子目录 |
| 内部归档 | agent仍须保存原文、来源、版本、字节数/SHA及分析，维护experiments/topk索引，成功失败都保留 |

发布授权不包含凭据入库，也不包含修改其他用户或共享系统环境。容器root身份不代表可以动共享主机。Git署名如需调整只用私人仓库的--local，不修改他人的全局配置。

用户明确要求：先查错误前提/逻辑跳跃/信息缺失；保持独立判断，核实结论；发现表述错误直接指出，并提醒遗漏成本和过拟合风险。授权范围内持续推进，避免只给计划或反复确认。

## 2. 已经完成了什么

### 2.1 环境与仓库已可用

| 项目 | 最近回传确认值 |
| --- | --- |
| 容器 | yjl，使用容器自带Python，不使用Conda |
| 私人根目录 | /home/y00937463；/home:/home已挂载 |
| 容器镜像 | quay.io/celenewyear/tilelang-ascend:cann9.3.0-weekly.20260930-ubuntu22.04-py3.12 |
| 设备 | Ascend950PR_9579；最近实验物理卡0，进程内映射npu:0 |
| Python | /usr/local/python3.12.14/bin/python3 |
| torch / torch_npu | 2.13.0+cpu / 2.13.0；NPU任务确已执行，不能仅凭+cpu后缀判定未用NPU |
| TileLang | 主仓0.1.15+cpu.git0176ba14，加载自site-packages；用户已编译并运行其仓内测试 |
| CANN | 9.3.0 Weekly20260930，安装信息V100R001C12B136，实际目录/usr/local/Ascend/cann-9.3.0 |
| 编译器 | bisheng，clang15.0.5，构建时间2026-09-29T23:53:43+08:00；InstallDir为cann-current/bin |
| 服务器仓库 | /home/y00937463/TileOPs、/home/y00937463/B2Y |

不要再安装/激活Conda、重新下载镜像或升级torch。旧部署文档中的“待升级2.13”等措辞是历史步骤，当前已经完成。新终端按真实CANN环境恢复变量，不能照抄旧服务器的SDK路径。

本地是Windows PowerShell：D:/04_Project/TileOPs和D:/04_Project/B2Y。本地没有可运行本任务的NPU；CPU/AST/源码复算在本地完成，设备验证由用户回传。B2Y在本工具环境中写入通常需要sandbox提升申请，这是工具边界，不是重新向用户索取已经给出的任务授权。

### 2.2 实现与版本身份

- **实测代码提交**：7247ad4f355c62a8679dcdc8fdf182c0b219db5e。
- **kernel文件字节SHA-256**：987da9f9d314f8744bef05a628e60a14b88e63db320374f36a3b6411ce5eb2e3。
- **Decode生产C++ SHA-256**：751a82889d13811a199edf8358af0d3fc84e6a54d17f985b6a27c877a1c10626。
- **Decode输入SHA-256**：c59b9485e9750eaf04d4668934c0383510a15b2cc9365ef9c6c4b9eaf2bf7593。
- 写入本交接前，本地TileOPs main为aae89ced，B2Y main为c516de6；这些是文档HEAD快照，本交接提交后会变化，不能把HEAD变化当作内核已优化。
- 最近服务器报告仍运行7247ad4f；不要假定服务器自动跟随本地main。以每次报告实际SHA/加载路径为准。

核心文件：src/tileops/_kernels/attention/topk_select_ascend.py；host为src/tileops/ascend/topk_select.py；设备契约为tests/ops/test_topk_select_ascend.py；CPU扫描测试为tests/kernels/test_topk_select_ascend_scan.py。

算法为精确radix选择、2048条键/索引双缓冲，缓存溢出回退完整过滤。默认grouped scan、H128。host的workers=min(rows,实际Vector核数)，本机历史确认56；N65536/K1024不用FP16粗筛，其余当前走粗筛。Decode R1因此仍只有一个worker。

### 2.3 H128 grouped scan已有净收益

候选通过14项NPU契约，最近完整JUnit记录232.353秒。两轮同进程固定输入、交替顺序、各50个事件：

| 轮次 | 旧H128事件μs | 新grouped H128事件μs | 旧/新速度比 | 时间降低 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 204.404995 | 161.917500 | 1.262402倍 | 20.7859% |
| 2 | 204.215504 | 162.101999 | 1.259796倍 | 20.6221% |

旧版与新串行消融生成源码相同，新grouped的两轮生成源码也与独立Decode测量一致。可以确认此Decode上的扫描净收益；Prefill缺少同轮旧扫描对照，不宣称其也有同样提升或已证明无回退。

H256 grouped scan出现253个未填槽位（值仍为16384哨兵），已限制为不放行；H64 grouped也没有设备验证。不能重新把H256设默认，不能把“禁用”写成“修复”。

### 2.4 已有性能差距

以下是scan-r1完整回传中的同轮**纯内核**中位数，不是本次stage1事件：

| 规格 | TileLang | CANN TopKV2 | 性能×CANN | TileLang耗时/CANN耗时 |
| --- | ---: | ---: | ---: | ---: |
| Decode N16384/K2048 | 157.0505μs | 9.2275μs | 0.058755倍 | 17.0198倍 |
| 128行诊断 N65536/K2048 | 1.8686455ms | 无同规格基线 | 不计算 | 不计算 |
| Prefill R32768/N65536/K1024 | 323.8217785ms | 50.6476015ms | 0.156406倍 | 6.3936倍 |
| Prefill R32768/N65536/K2048 | 411.563602ms | 52.2957615ms | 0.127066倍 | 7.8699倍 |

每组保留50个事件/50个kernel数值样本；CANN与候选输入和规格匹配。原始profiler CSV未全量回传，不能称设备时间排序/全部警告已独立审核。不能拿2026-10-30旧服务器8.721μs替代这轮9.2275μs。

正式CANN条件：G1、全窗口、FP32，largest=true、sorted=false、indices_dtype=INT32、sort_policy=0、dim=-1；每shape/K独立静态.om，GE使用must_keep_origin_dtype，预分配、校验后warmup；profiling与事件独立。旧ACLNN INT64再Cast方案不是正式基线。

### 2.5 计时探针与真实阶段诊断已完成

probe1（20261102-probe1）在compile:r1-timed的Python源码检查失败；回传C++确认预期屏障/计数顺序没有保留，旧检查器此次不是误报。只证明生成源码边界不对，不能归咎于Bisheng最终指令重排。

probe2（20261102-probe2，发布工具提交6853356）改用主仓register_ascend_postproc_callback，在调度后严格插桩。10组配置各50个事件，4组timed各50个counter，共500/200样本；21份文件与10组转换复核一致。64−16轮差分1004.118ticks/μs，与950约定1000偏差0.412%；counter稳定，完整事件存在明显漂移，不能宣称插桩零开销。空区间92ticks也不是可机械扣除的常量。

真实stage1（工具提交f8bc23f）已在Decode R1/N16384/K2048/W1/H128/coarse/grouped运行完成，原上传文件：

D:/04_Project/B2Y/AAA_Results/topk-stage1-20261102-163258.txt

原上传提交7a389907815ebe6354a4636084df7f69892d18a9；独立归档experiments/topk/20261102-stage1-decode-163258/，发布提交0d4aa72。

| 阶段/聚合 | 中位耗时μs |
| --- | ---: |
| 第0轮histogram | 55.983 |
| 第0轮compact | 73.2385 |
| 两段同次相加后取中位数 | 129.2055 |
| 五轮scan同次合计 | 7.345 |
| 已停用的第4轮 | 1.983 |
| 最终tie/收尾 | 9.589 |
| timed全部27段同次总和 | 159.467 |
| timed counter外包络 | 161.7985 |

首轮两段占同次已测阶段总和的中位比例81.039%，占counter包络约79.876%。分母不是无插桩总时间；不能把不同段的中位数相加冒充一次运行。

4个版本各50个事件中位数：production 162.7365μs、traced 164.1345μs、fenced 166.4200μs、timed 168.6810μs；完整插桩比production增加3.65284%。50×27=1350个阶段样本和50×56绝对时间戳全部复算；13/14个VF算术体与生产一致，只有状态更新VF另加快照。工具检查与报告显示随机/全相等集合正确，但这不是新增一轮完整14项测试。

候选状态（从0编号）：

| round | active | accepted | 已提交 | 剩余quota | next candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 16384 | 1767 | 1767 | 281 | 835 |
| 1 | 835 | 0 | 1767 | 281 | 835 |
| 2 | 835 | 263 | 2030 | 18 | 25 |
| 3 | 25 | 18 | 2048 | 0 | 0 |
| 4 | 0 | 0 | 2048 | 0 | 0 |

本例没有缓存溢出；第1轮accepted=0仍在收窄FP32前缀，不能删掉。第3轮后quota归零，末轮/tie空转是次级目标。**已定位哪里慢，但未证明根因是原子、转换还是访存。**

### 2.6 文档与保存已落实

B2Y已有10条收件/实验档案，含同一次运行的摘要/源码补充，不等于10次独立复现。索引为experiments/topk/README.md和index.json；每条metadata记录出处、缺失项与原字节身份，失败也保存。

阶段总结已发布c516de6，优化路线复评已发布e3d677f；个人fork路线文档已随aae89ced同步。新的首轮专门化、粗键缓存、分片计数、首轮分核都还没有实现/设备结果。

## 3. 当前卡在哪

**没有已知的新环境故障或待用户审批项。当前停在“诊断与路线复评完成、新优化候选尚未开始实现”这一位置。**

1. 主要工作是把首轮129.2μs转化为可检验的优化。只知道阶段占比，不能预设原子竞争一定是根因；C++表达式重复也不证明Bisheng没有消除。
2. 全部阶段证据目前只覆盖一个固定Decode输入；Prefill K1024为直接FP32，K2048为粗筛，不能从Decode直接决定缓存/溢出策略。
3. 插桩有3.65%总事件扰动，短阶段偏差比例未知；最终汇编/硬件依赖、独立重复、完整负载/频率时序及本轮CANN均缺失。可用于此case排序，不能全面认证阶段计时；stage_profiling_accepted及最终版本接受仍为false。
4. 还未达到CANN水平，也未完成最终v0.1.4验收。不能仅凭14项通过或status=completed打成功标签。
5. agent不能执行新服务器NPU任务；准备代码、工具和指南后，由用户运行再上传。这是固定协作方式，不要重问密码或尝试旧SSH。

最后一次回传stage1为completed，前后健康OK；交接时未连接服务器，**当前卡0是否空闲、是否有其他任务未知**。不得据旧日志宣称无运行进程，不擅自终止共享任务。

## 4. 下一步计划是什么

### 4.1 第一项实际工作：独立的首轮专门化候选

用户后续说“继续工作”时，应从此处实施，而不是再做一轮摘要或重复旧探针：

1. 冻结7247ad4f对应源码及其SHA作为baseline；先检查本地/远端新增结果与Git状态。
2. 优先把COARSE首轮与后续精确FP32轮的动态判断/粗键重复计算做最小独立修改，先显式复用一次粗键，必要时再分离首轮路径。不要同时叠加缓存、分片和跨核。
3. 保持COARSE=false旧路径、完整FP32键、窗口/哨兵、K>2048、溢出计数和确定性语义。首轮拆出后尤其保留round→byte_shift和缓存乒乓奇偶关系，不能简单从0重编号；避免超过Python静态嵌套上限。
4. CPU语义与生成代码、小规格设备正确性之后做同输入无插桩AB/BA。没有收益就记录失败并停止放大，不因C++更短就认为更快。
5. 有希望的候选再跑14项、独立进程重复、128行与完整Prefill；多种seed/分布/窗口核对不过拟合。各版本源码不同是正常的，应分别冻结身份，不要求不同算法SHA相同。
6. 实现提交个人fork，工具/新命令先发布B2Y，再让用户执行；使用新版本和新runs目录，结果直接上传AAA_Results。

### 4.2 后续候选顺序与成本

| 候选 | 动机 | 必须计入的成本/风险 |
| --- | --- | --- |
| uint8粗键缓存 | histogram计算粗键后存UB，compact复用；本例仅边界835条再读原FP32 | Decode每worker增加16KiB，N65536时64KiB；多一次UB写读，大边界桶可能无收益；不能把粗键当最终精度 |
| 分片histogram | 先4片逻辑bank，用独立VF归并int32到原256桶 | bank数组4KiB；零化/归并/布局与bank冲突；只是分散争用，原子次数没变 |
| 复用分片计数做compact | 分片/桶前缀分配互不重叠区间，分片内保留游标 | histogram/compact元素归属一致，生产/消费有VF边界；边界总数必须完整，>2048继续精确回退 |
| Decode只分首轮 | 2/4/8个worker：局部hist→全局阈值/偏移→并行compact→单核精确收窄 | 同一流多kernel的launch、GM通信、工作区及元数据重建；跨kernel不保留UB；不为分支每次做D2H同步 |
| quota归零后条件跳过 | 末轮和tie仍有可见成本 | 不能无条件删tie；旧v0.1.1 guard未确认整体收益；不预告能省掉全部插桩段时间 |
| Prefill专属调查 | 统计候选分布、>2048频率、重扫占比与行间差异 | 当前单worker工具不直接适用，需重新设计记录槽位/归属；先小规模固定行再完整测量 |

同核方案先单项验证，仅组合有效项。Decode分核可尽早用小P验证launch/归并成本，避免投入完整流水后才发现开销抵消。

不要优先做朴素“每核local TopK再大归并”：本例P8时每块2048=K，会保留整行16384条；P2/P4候选上界4096/8192。优先比较全局粗阈值后压缩、再单核处理835条的路线。多kernel新路线必须改测量器，不能沿用只解析一条main kernel的性能倍数逻辑。

原第4.2节的其余优先级已经修正：grouped scan已落地，五轮scan仅7.345μs；本例无溢出；R32768/W56在每行等成本模型下完美均衡仅省约0.1463%总时间，不为单个尾部先引入复杂行拆分。大Prefill已按行占核，拆每行会减少同时处理的行数，不保证吞吐收益。

### 4.3 验收与最终交付

候选可以快速以小规格淘汰，但进入默认前必须完整契约、独立重复和主规格回归。通用默认造成任一主规格回退则拒绝；有明确守卫的Decode专用路由可单独评估，并验证未路由路径保持原实现/正确性和性能。

保留50次事件与匹配口径的kernel/流水设备记录；不按耗时挑最快值。先校验再warmup，输入输出/事件预分配，生成/编译/哈希/H2D和正确性检查不纳入定时。最终补全同轮CANN、接受状态、源码/输入/工具身份与历史记录，再谈正式版本/标签。

## 5. 有哪些踩过的坑，绝对不要再踩

1. **沿用过时限制或凭据。** 新服务器只由用户操作；早期保密/禁止发布已被用户最新开源授权更新。不要重问SSH密码，不把密码/令牌写进Git，不重新要求发布审批。
2. **越界环境修改。** 只写私人目录；不改公共SDK/驱动、系统时钟/频率、不重置NPU，不按用户/进程名批量kill。/home挂载是共享的。
3. **重新部署已经成功的环境。** 直接使用yjl容器Python；不再折腾Conda/新镜像或无依据升级依赖。torch的+cpu后缀与torch_npu实际执行不冲突。
4. **同一代码不同批次直接比较。** 旧服务器曾出现同生成码巨大漂移。先做当前同进程对照和只读健康/负载检查，不用历史最小值外推收益，也不修改共享频率“固定性能”。Aicore1650MHz不是GetSystemCycle换算；950探针按1000ticks/μs。
5. **混合计时范围。** 事件不等于纯kernel，counter不等于整个调用；不能机械扣空区间或强制区间和等于事件。Probe2约14μs事件不等于单次launch开销。多worker耗时不能相加当墙钟。
6. **被completed误导。** completed仅表示采集完成，stage_profiling_accepted/最终接受要单独判断。主要VF源码边界正确也不证明最终汇编依赖全部正确；没有新NPU结果就如实标待测。
7. **重试已失败H256。** H256 grouped有253个哨兵未填，H64 grouped未验证；保持H128，不靠更多线程猜性能。
8. **SIMD/缓存一定快的偏见。** 历史SIMD C2048直方图四组均回退，C16384有unsafe ThreadSync/race警告；完整行UB缓存也曾回退。通用popcount/shfl_sync降低失败、所试warp_reduce_sum更慢。新方案先验证主仓0176能力，不照搬CUDA warp或独立tilelang-ascend API，不照抄CANN UB经验常量。
9. **同VF隐含跨线程依赖。** 本次计数/扫描通过独立VF分隔生产消费；诊断快照由更新状态的thread0串行写7字段，不另让其他线程无屏障读取刚更新状态。不得放宽同步检查掩盖race。
10. **候选截断造成错误。** int32完整计数，65536同桶不能用uint16总计数；缓存满仍记录总数并精确重扫，不能只在前2048条里继续选。逻辑前缀过滤仍遍历全窗，不等于仅访问边界元素；需物化/寻址数据才能真缩域。
11. **把所有原子一概禁掉，或反过来用抢位截断tie。** 可以对全部已确定入选者分配无序槽位；不可以用先到先得决定同分最终保留谁。当前契约要求集合确定，不要求输出全局排序。
12. **误删看似无用的轮/tie。** round1 accepted=0仍在细化前缀；只有条件成立才跳过tie，全同分/特殊值仍要小索引选择。候选835只是此输入，不是粗筛容量保证。
13. **深层嵌套与编号错误。** 历史外包early-if触发too many statically nested blocks；拆首轮同时容易破坏byte_shift、粗前缀和乒乓读写奇偶，必须独立检查。
14. **对旧诊断放宽SHA/布局断言。** stage1只支持固定源码、R1/H128/coarse和14个VF、5轮27段。A0改布局后旧工具应拒绝；新建版本/映射，不手改hash绕过。probe1也不能删断言恢复测量。
15. **修改已测工具身份或复用缓存目录。** topk-950pr工具与独立probe/stage目录有manifest。顺手改README也会改变身份；改动需更新manifest、版本化并使用新runs/cache，不把结果混入旧EVIDENCE。回调只在独立Python进程注册，不动公共安装。
16. **profiler成功退出就是完整数据。** 曾47/50缺样；现有做法多发余量后按设备时间固定取最后N条，不挑最快；不混合多个旧profile目录。新流水要核对全部真实任务与间隙。
17. **错误CANN基线。** 不用INT64+Cast旧ACLNN替代；不遗漏must_keep_origin_dtype导致降精度；不把启动56 blocks直接说成Decode有效使用56核。比较限定已验证的G1全窗等规格。
18. **覆盖/美化原始证据。** 不改根目录“实验结果”“信息.txt”夹带更新，不覆盖AAA_Results旧上传。每次收到结果独立归档，含失败/不完整；修正另写补充分析。C++原始尾空行和空白也要保留。
19. **Windows换行导致哈希混淆。** 最新上传Git LF文本220182字节（SHA 6cf74dc9cfc9c65fb873baadd6b770cd631e4feea2106c0def13ad0af3f0fb45）；Windows CRLF检出225341字节（SHA ec1ba0aaac184c2aefc0a47d6c1a979ff762a3070f5be593d2ffb2d49b248316）。已证实仅5159个CR变化；档案同时保留raw-return.txt/local-checkout.txt。不要误判为两次实验或重写数据来对齐hash。
20. **把内部归档要求重新强加用户。** 用户只上传AAA_Results里的结果，不生成校验文件/额外层级。内部仍可且应记录出处和SHA。不要重建incoming/topk的复杂上传流程。

21. **提交成功就说远端已同步。** GitHub 443曾多次间歇超时，B2Y常可正常推送；本地commit与remote push分开确认，有限重试后如实报告未同步项。不因网络失败更换用户remote、修改凭据或force push。

## 6. 写给一个完全没有上下文的新会话

### 6.1 首先读取这些材料

| 顺序 | 入口 | 作用 |
| --- | --- | --- |
| 1 | 本交接；本地D:/04_Project/B2Y/AGENTS.md | 最新任务状态、授权与保存/指令规则 |
| 2 | [阶段进展总结](https://gitcode.com/JotaroSSR/B2Y/blob/main/deliveries/topk-progress-summary-20261102.md) | 快速把握已完成/未完成 |
| 3 | [最新优化路线](https://gitcode.com/JotaroSSR/B2Y/blob/main/deliveries/topk-optimization-roadmap-20261102.md) | 第4.2节的逐项修正、方法与成本；个人fork同名设计文档亦可读 |
| 4 | [stage1分析与档案](https://gitcode.com/JotaroSSR/B2Y/tree/main/experiments/topk/20261102-stage1-decode-163258) | analysis.md→review.json→report.json→production/timed C++；按需读，避免一次打印22万字节 |
| 5 | [H128扫描结果](https://gitcode.com/JotaroSSR/B2Y/blob/main/deliveries/scan-r1-results-20261102.md) | 已验证净收益和同轮CANN差距；其“下一步”是当时历史状态 |
| 6 | [全部档案索引](https://gitcode.com/JotaroSSR/B2Y/blob/main/experiments/topk/README.md) | probe1失败、H256失败、probe2、完整stage1的来源链 |
| 7 | src/tileops/_kernels/attention/topk_select_ascend.py、src/tileops/ascend/topk_select.py、tests/ops/test_topk_select_ascend.py | 当前实现、配置与契约真相 |

旧docs/design/topk-select-950pr-handoff-20261031.md只作为历史；其“未提交”“不许推送”、旧远端/Conda，以及第4.2节旧排序不能当现行指令。缓存ops-math快照位于本地artifacts/topkv2-analysis-20261030/official，source-index记录提交e287c428bf5bcdb8279b0e4764f4b7914ec66003；5个相关Git blob已核验，但不能当作当前CANN实际路由证据。

### 6.2 本地开始工作时核查什么

- 本地cwd通常D:/04_Project/TileOPs，PowerShell。origin为个人fork，main跟踪origin/main；不要改回原上游或切到旧npu/topk-select-950pr分支。
- 先查两仓Git状态和远端新上传，必要时fetch、检查差异后ff同步，保留用户变更。交接前TileOPs已跟踪工作区干净；B2Y原有未跟踪.idea/、main.py，**不要提交或清理它们**。本交接完成后HEAD会是新文档提交，kernel身份应仍为上述SHA。
- B2Y最新原始结果文件是AAA_Results/topk-stage1-20261102-163258.txt，已完成分析；除非有新上传，不把它再次当待处理实验，也不让用户补同一次源码。环境/工具链变化、新计时故障或用户明确要求时可以重新验证；“不重复旧探针”不等于禁止必要的新验证。
- 本地artifacts/含探针镜像、导出解析和Ruff等辅助产物，默认Git忽略。正式证据在B2Y和个人fork文档，不能只引用临时附件/旧工具session ID。未回传的服务器文件不补造。
- 能运行的本地验证为标准Python CPU/AST测试和静态检查。probe目录曾20项、stage目录7项CPU回归通过；stage测试需能找到生产源码（TOPK_SOURCE），不要把skip当实际执行。无实现变化时不必为读交接重跑全部测试，更不能声称本地已做NPU验证。

### 6.3 需要用户跑设备实验时

先完成候选代码、CPU/静态检查、工具和可审阅指南，提交个人fork并发布B2Y，再告诉用户入口。新命令必须写入B2Y，包含容器位置、已有环境、新输出目录、失败回传方式和测量范围。不要让用户从聊天跨电脑复制命令。

现有入口：

- B2Y/topk-950pr/START_HERE.md、env.sh、run.sh为完整旧闭环；env.sh用TOPK_ROOT、TOPK_REPO、TOPK_TOOLS、TOPK_DEVICE、TOPK_PYTHON及EVIDENCE。CANN实际环境需已加载或通过CANN_ENV_SCRIPT指定，不假设固定脚本存在。
- B2Y/diagnostics/topk-timer-probe/为已完成最小probe；export_complete.py只读导出已完成probe2，不用再跑。
- B2Y/diagnostics/topk-stage-timing/为固定源码stage1，run.py在私人新目录生成AST诊断副本；instrument.py严格映射/移除VF标记后插桩。新增首轮布局要对应新工具版本。
- 独立诊断进程自行设置private cache/tmp/logs，不改HOME或父终端EVIDENCE；完整闭环更新源码/工具后换新的EVIDENCE，已经开始的实验不要中途pull。
- [AAA_Results上传指南](https://gitcode.com/JotaroSSR/B2Y/blob/main/AAA_Results/README.md)是当前简化方式；服务器不需要新建tools目录、导出ZIP或校验清单。旧完整部署/升级步骤仅在确有需要时参考，当前不要从头重做。

用户上传后，按Git来源读取，先验证完整性/版本/输入/源码与采样，再独立归档和分析、更新索引及首页。更新结论要留出处和缺失项；代码修改依性能证据逐项推进。

### 6.4 接手后的第一句话与第一项动作

可以先向用户说明：“已接手：H128扫描已验证，阶段瓶颈在首轮histogram/compact，新候选尚未实现。我会从首轮专门化的独立对照开始，保持FP32与索引契约，并先把执行材料发布B2Y。”

随后实际检查本地状态、读取现有实现与最新路线，落实第4.1节候选；不要再次问已明确的容器、目录、账号权限或开源范围。若用户只要求阅读交接则先完成阅读，收到继续工作的指示后按此推进。本会话结束本身不启动任何服务器任务。
