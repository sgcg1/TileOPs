# TopK阶段计时前的最小探针

目的：验证当前主仓TileLang0.1.15/CANN9.3 Weekly上，设备计数器、完成屏障和观察开销是否适合后续阶段诊断。probe2完整证据已复核，具备进入单worker Decode诊断的基础；真实TopK阶段耗时及插桩适用性仍待现场验证。

## 最新进展：完整探针复核及Decode阶段工具

完整导出对应运行timer-probe2-20261102-154618，与上一轮摘要是同一次运行。21份嵌入文件的字节数/SHA、500个事件和200个counter样本全部核对；10组后处理可从original逐字节重现，API源码均匹配postproc。非空入口为load VF→begin→全部工作VF→end→store VF，定义内部为屏障→时钟→记录；不同观察模式的工作VF定义一致。最终Bisheng指令/硬件依赖尚未独立证明。

counter范围分别为r0 91～93、r1 613～621、r16 8801～8823、r64 35119～35161ticks。16−1与64−16的每新增一轮counter增量分别546.133和548.604ticks，约差0.45%。完整事件有明显长尾和轮次漂移：r1-plain最大108.447μs；r1-timed前/后25轮事件中位数13.931/18.863μs，但counter均615ticks。因此短任务的事件差分不能直接当作内部工作区间代理；具体漂移原因仍未定位。

先前约−0.30%～+0.16%的独立中位数变化不证明零开销。逐轮timed−plain事件差中位数为r1 +0.0175、r16 −0.1065、r64 +0.0850μs，MAD约0.2430/0.4035/0.3005μs；不能据其符号声称插桩加速。完整原文、原始样本、20份C++、分布统计与分析新增归档B2Y `experiments/topk/20261102-timer-probe2-complete/`，不改写前一轮摘要档案。

当前入口为B2Y `diagnostics/topk-stage-timing/README.md`：仅测Decode R1/N16384/K2048/W1/H128/coarse/grouped，比较production/traced/fenced/timed。工具从固定SHA的生产源码生成私人AST诊断副本，以14个VF内部唯一标记识别阶段后删除标记，再在C++入口围住阶段组；未知融合/映射/控制流拒绝执行。记录27个区间、5轮7字段状态与全量重扫标志。额外状态写入成本明确归入state_update_with_trace，不将其视为生产纯成本。

该工具不修改生产实现或安装环境，所有命令已放B2Y；首次回传即包含原始样本及源码，免去先摘要再补源码的往返。新增7项CPU测试覆盖AST去除诊断后的工作逻辑一致性、标记映射、未知布局/额外循环拒绝和逐轮统计；本地没有NPU编译/运行，stage_profiling_accepted仍为false。先审阅正确性和观察扰动，再据真实阶段数据提出优化，不从最小加法探针推断TopK瓶颈。

## 最新现场反馈：probe2完成，原始证据待复核

用户回传20261102-probe2摘要，status=completed、输出参考检查通过。8项工具SHA与发布提交6853356一致。报告称10个配置各50个事件样本，4个timed配置各50个counter样本；后处理入口检查全部通过。当前仅收到摘要，未独立检查逐样本或实际C++。

1/16/64轮timed counter中位数分别0.615/8.807/35.140μs，完整事件中位数14.165500/17.214000/43.438999μs；空区间counter为0.092μs。64−16差分得到1004.118255ticks/μs，相对探针1000约定偏差+0.411826%。timed相对plain事件中位数变化分别+0.155550%、−0.295396%、−0.067867%；缺少离散度，不能据此声称零开销或加速。

16−1差分却为8.192μs counter对3.048500μs事件，差分比2687.223ticks/μs。两者测量范围不同，不能直接据此否定counter，也不能只挑一致的大工作量差分宣称全范围校准。先读取原始分布和生成代码，再解释短任务偏差；不机械扣除空区间观察成本。

完整数据、复算与限制已归档B2Y `experiments/topk/20261102-timer-probe2-summary/`。当前操作入口改为 `diagnostics/topk-timer-probe/EXPORT_COMPLETE.md`：更新B2Y后，只读导出现有probe-report.json和源码为一个文本，不编译、不运行NPU、不修改原实验。此前“尚无probe2结果”等叙述保留为当时状态。真实TopK阶段计时和最终版本接受均未完成。

## 首次现场反馈：源码顺序检查失败

20261102-probe1现场回传status=failed、stage=compile:r1-timed、error=Counter/fence source order changed，脚本哈希与已发布版本一致。已确认Kernel的prelude参数和call_extern接口存在，并已取得r1-timed生成源码供检查；未进入统一50次采样，尚无有效计时校准结果。

首次摘要阶段尚未收到r1-timed.cpp或服务器完整probe-report.json。检查器用全文件字符位置匹配同名标记，假设先定义再调用，没有解析函数作用域；当时不能确认编译器重排或检查器误报。保留断言和原内核，下一步只读导出原失败目录中的源码与标记位置，不重新编译、不运行NPU。

操作步骤已放入B2Y `diagnostics/topk-timer-probe/INSPECT_SOURCE.md`，回传一份带行号的source-inspection文本。收到的各轮结果和此次失败原始摘要按独立ID归档于B2Y `experiments/topk/`，附来源、原文哈希与状态；未回传的信息标明缺失，不覆盖旧数据。

## 后续源码证据与probe2

用户随后回传48行r1-timed.cpp。还原编号前缀后，字节SHA与导出记录一致：0c3c1d405730f844ed6ea28a96f9cdbe769f2b98d37a0e92bd4bbb5d97220b91。重建的原摘要也与服务器给出的摘要SHA完全相符。此回传归档为B2Y实验ID `20261102-timer-probe1-source-inspection`，前述“缺失/未确认”保留为当时的证据状态。

实际entry第40～46行依次为input VF、start clock、work VF、fence、end clock、fence、output VF。辅助定义确实位于前部，旧检查器在此例没有误报：生成C++已不满足预期屏障/计数顺序。不能据此认定Bisheng最终指令重排或硬件完成语义，也不能恢复旧探针直接测量。

probe2改用主仓 `register_ascend_postproc_callback`，在TileLang调度之后处理带唯一模式标记的单个Vector入口。TIR只生成普通VF工作以及用于保留ticks参数的无效ABI锚点；后处理移除锚点，在加载/输出VF之间包围整个工作循环。begin/end辅助函数内部按屏障→时钟→GM记录排列。空区间放在输出之后，避免猜测已融合VF的内部边界。

后处理严格核对入口作用域、VF数量与x/y边界，只支持该小探针的已知生成形态；未知布局、未观察到回调、重复捕获均停止。保存原源码、回调后源码与API返回源码及SHA，区分不同捕获点。新目录独立编译，不修改SDK/TileLang安装或生产TopK。

本地已覆盖源码转换、边界被移动的拒绝、未知布局/重复标记的拒绝和空工作区间等CPU回归。源码修复不等于计时方法已通过；尚无probe2设备运行结果，仍需50次采样、尺度、输出及观察开销检查，`stage_profiling_accepted`保持false。所有命令已写入B2Y探针README。

已有H128完整结果见 `topk-select-950pr-scan-20261102.md`。本次不改TopK内核、host、契约测试或现有B2Y topk-950pr工具；探针独立放在B2Y `diagnostics/topk-timer-probe/`，输出与cache单独放在私人runs目录。旧实验和源码身份不受新增目录影响。

## probe1最初方案（已拒绝，保留历史）

采用主仓 `T.Kernel(prelude=...)` 和 `T.call_extern` 注入少量计数器/屏障辅助函数。官方文档确认950PR支持AscendC::GetSystemCycle，但当前安装版本的接口和实际降低仍须编译探针确认；接口缺失时保留失败阶段，不盲目更新依赖。

一个Vector worker、H128、4096个FP32值驻留UB。初始值和最终输出拷贝在计数器区间之外。比较同样VF工作量的plain、仅屏障fenced、屏障加计数器timed版本；工作量为1/16/64轮有数据依赖的加1操作，另有0轮timed空区间。输出消费中间状态，逐元素与精确CPU参考比较，减少工作被消除而误测的风险。仍应查看生成源码及必要时的最终指令，源代码存在不证明实际指令区间正确。

每种预热5次，再同进程交替正反序采50次；事件对象、输入、输出及记录缓冲预分配。全部配置做同样的时间戳缓冲回读，回读位于事件区间之外。NPU事件覆盖整个调用，含初始化/写回/调度；counter区间只覆盖中间VF和观察成本。使用64轮减16轮的计数器差/事件差估计单位，避免要求绝对总时间相等。

CANN对950的文档值为1000 ticks/μs，与Aicore curFreq字段不同。10%偏差窗口是诊断启发式，不是认证标准；不符合时不自动换倍率。记录屏障和计数器造成的事件百分比变化与空区间原值，不机械扣除开销，也不把负变化裁成0。

## 成功标准及限制

先确认编译完成、输出正确、时间戳确实被写入且递增、工作量变化与counter区间相符，再看尺度检查和观察扰动。探针的 `stage_profiling_accepted` 始终为false；completed只指采集完成。工具通过不证明TopK每个VF、原子统计或同步区间都能同样计时。

只有通过审阅，才扩展为TopK每轮histogram、scan、compact、状态更新、tie阶段记录，并同时保存候选数量、溢出重扫信息。多worker的阶段时间不能直接相加当作墙钟耗时。额外屏障可能改变原流水，需要与无插桩版总耗时比较；仿真周期与设备耗时分开报告。

原probe2发布前的本地验证范围为Python/Ruff、16项CPU解释、只读导出与后处理回归测试、文档Bash语法和文件哈希。本地未运行NPU。随后增加完整证据导出器及4项CPU回归，当前全部20项通过；现场最新进展和当前操作入口见文首，历史INSPECT_SOURCE.md仅用于probe1。

参考：[GetSystemCycle的950换算与同步示例](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSystemCycle_ISASI.md)、[主仓Kernel prelude](https://github.com/tile-ai/tilelang/blob/main/tilelang/ascend/language/kernel.py)、[主仓诊断流程](https://github.com/tile-ai/tilelang/blob/main/.agents/skills/tilelang-ascend/references/profiling.md)。
