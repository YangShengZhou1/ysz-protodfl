# 实验分期目录

每期工作一个文件夹，命名 `YYYY-MM-DD_主题`，内含：

- `checks/`：该期离线检查与分析输出（`report.md`、`summary.json`、`per_candidate.csv` 等）
- `checkpoints/`：该期需要从服务器取回的快照（可选，体积大）
- `README.md`：本期做了什么、对应哪条实验记录、怎么复现

训练日志与通信统计仍在 `outputs/logs/cifar10/`，按 tag 名对应，不随期次移动；需要跨期对比时按 tag 前缀找。

历史路径 `outputs/checks/<name>` 已全部迁入对应期次，文档中的引用已同步更新。

## 现有期次

| 期次 | 主题 | 记录位置 |
|---|---|---|
| `2026-11-02_startup-and-hop` | 启动检查、跳数语义 | `docs/02` 2026-11-02 条目 |
| `2026-11-03_startup-refactor` | 启动口径重构复核 | `docs/02` 2026-11-03 条目 |
| `2026-11-04_paired-and-kl` | 配对聚合检查与 KL 收益分布 | `docs/02` RUN-012～RUN-019、`docs/09` 第 4 与 9.5 节 |
| `2026-11-05_same-target-candidates` | 同目标候选差异（RUN-020）、聚合后重放（RUN-021） | `docs/2026-11-05/16_同目标候选差异_2026-11-05.md`、`docs/2026-11-05/17_短程重放_2026-11-05.md`、`docs/09` 第 10–11 节 |
