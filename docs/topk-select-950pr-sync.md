# TopKSelect：个人 fork 与 B2Y 分工

2026-10-31 起，代码以个人 fork `https://github.com/sgcg1/TileOPs` 的 `main` 为准。

- TileOPs 仓库：实现、项目测试、设计与正式性能记录。
- B2Y 仓库：`topk-950pr/` 辅助脚本及 Docker 操作指南。
- 服务器：在已有 `yjl` 容器中克隆/更新个人 fork；由操作者从 B2Y 选择辅助文件手动传入。服务器不克隆 B2Y，不通过整项目 ZIP 覆盖源码。

完整操作指南位于 B2Y 的 `topk-950pr/README.md`。

新服务器交付包为 B2Y 的 `deliveries/topk-950pr-tools-20261031-r3.zip`，只含辅助材料；解压后从 `START_HERE.md` 开始。包内 `SOURCE.json` 记录参考源码提交和哈希。实际环境为已有 yjl、镜像 Weekly 20260930、私人目录 `/home/y00937463`，已挂载 `/home:/home`；使用容器内 python3，不使用 Conda、不重建容器。

默认容器内布局为：

```text
/home/y00937463/
  TileOPs/   # git clone 个人 fork，分支 main
  tools/     # 从 B2Y topk-950pr/ 手动传入的文件
  runs/      # 每轮独立原始结果，留在服务器
  tmp/       # 私人临时目录
```

路径、设备号和环境可覆盖，以实际获准配置为准。宿主机执行 `docker exec -it yjl /bin/bash`，随后在容器内执行：

```bash
git clone --branch main https://github.com/sgcg1/TileOPs.git /home/y00937463/TileOPs
git -C /home/y00937463/TileOPs fetch origin --tags
# 使用容器内 Python；如镜像未加载CANN，先source实际toolkit环境脚本。
source /home/y00937463/tools/env.sh
bash "$TOPK_TOOLS/run.sh" preflight
bash "$TOPK_TOOLS/run.sh" test
```

已有 clone 时检查工作区干净后用 `git pull --ff-only origin main`，不重复 clone。测试时只暴露指定的一块物理设备，进程内部逻辑设备为 `npu:0`。不自动 pip install 项目，以免 pyproject 的 TileLang pin 覆盖主仓 Ascend 构建。

## 历史整合说明

原本地仓库是独立初始化的快照，分支 `npu/topk-select-950pr` 的保存点为 `e96b1b36`。原来的五个提交、此次候选保存提交及 `topk-npu-v0.1.0`～`v0.1.3` 标签保留，合并将它们接入个人 fork 的历史；没有改写原提交或强推个人仓。

整合以个人 fork 原 main `1301da692d7b1c945f73e94198c5c7052aeeacbc` 为基础，仅迁入 TopK 适配、相关参考验证、测试、本地新增文档和 Ascend backend entry point。上游通用 GEMM、归约、注册模块等保持 fork 内容，不用旧快照覆盖更新。旧快照的完整状态仍可从原分支、标签及合并的历史父提交检索。

辅助测量脚本不放在 main 当前目录，原有脚本仍能从历史提交取回。此前整项目 ZIP 迁移方式已废弃，当前代码同步唯一入口为个人 fork。

## 验证状态

候选源码 `49aaea36` 已在旧服务器物理卡 0 的独立缓存下通过 14 项 NPU 测试，详见 `docs/design/topk-select-950pr-card0-resume-20261031.md`。没有调整候选算法，四组性能验收仍未完成，未发布 v0.1.4 标签。新服务器须重新运行全部契约和同轮基线，旧服务器的通过、漂移和频率读数都不直接外推。

当前 Windows 环境没有 torch、pytest 或 NPU。新交付材料在本地完成 Python/Ruff/Bash 静态检查与离线数据验证，新容器执行结果仍需由操作者验证。包内加入持续负载遥测、时钟交叉检查、独立缓存和按 case 分开的官方模型测量，不修改共享频率、驱动或系统配置。原始数据留在保密环境，可按包内 `RESULTS_TEMPLATE.md` 仅回传允许披露的摘要。
