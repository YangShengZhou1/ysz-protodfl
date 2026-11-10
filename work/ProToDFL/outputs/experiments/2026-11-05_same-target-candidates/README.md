# 2026-11-05 同一目标客户端的候选差异与短程重放

本期主题：回答「同一客户端在多个相似伙伴中选谁」的三问——差异多大、经过 1 个本地 epoch 是否保留、瓶颈在候选还是选择。

包含：

| 目录 | 内容 | 脚本 |
|---|---|---|
| `checks/same_target/e5-round1-base-s0/` | 主口径：5 epoch、187 对、11 个可比较目标（RUN-020） | `scripts/same_target_candidate_analysis.py` |
| `checks/same_target/e5-round1-base-s0-strict/` | 严格口径：要求聚合前后多数类占比 < 0.9（152 对） | 同上，加 `--max-majority-fraction 0.9` |
| `checks/same_target/e1-round1-base-s0/` | 1 epoch 对照（33 对，仅 2 个目标可比较） | 同上 |
| `checks/replay-r10-base/` | **主实验 B**：r10 快照上 62 对、11 目标的聚合→重放 1 epoch（RUN-021） | `scripts/same_target_replay.py` |
| `checks/same_target/smoke-replay*/` | 短程重放脚本的冒烟输出，不是结论 | 同上 |
| `checkpoints/tagsrv20-e5-r10-base-s0/round_10/` | 从服务器取回的 5 epoch、第 10 轮快照（只聚合 base 的配对起点） | `scripts/remote_pull.py` |
| `pull.log` / `pull.err` | 快照下载日志（854 MB、41 个文件） | 同上 |
| `figures/fig1_spread.png` 等 4 张 | 实验一（RUN-020）交付插图：候选极差、regret、图距离、KL–Δacc 散点 | `scripts/make_run020_figures.py` |

对应记录：RUN-020 见 `docs/2026-11-05/16_同目标候选差异_2026-11-05.md`，RUN-021 见 `docs/2026-11-05/17_短程重放_2026-11-05.md`；两条都在 `docs/02` 有完整条目，`docs/09` 第 10、11 节有结论摘要；口径与决策见 `docs/03` 同日条目。

实验一交付物：

- `docs/2026-11-05/18_实验一_同客户端候选差异_2026-11-05.docx`：7 节、4 图、4 表，数字由 `scripts/run020_stats.py` 从本节 `per_candidate.csv` 重算，与三份 `report.md` 逐项核对一致。
- `docs/2026-11-05/实验一_同客户端候选差异_2026-11-05.zip`：文档 + 5 个脚本 + 三份口径输出 + 4 张图 + 输入 `kl_benefit.csv`，共 24 个条目。

主实验结果（RUN-021，62 对 / 11 目标，核查集为各目标 1000 样本）：

| 项 | 聚合后 | 重放 1 个本地 epoch 后 |
|---|---:|---:|
| 平均 Δacc | −1.64 pp | −2.33 pp |
| 目标内极差均值 | 6.68 pp | 12.31 pp |
| KL top-1 与最优的差 | 3.41 pp | 6.07 pp |
| KL top-1 恰好最优 | 1 / 11 | 1 / 11 |

复现（主口径）：

```bash
python scripts/same_target_candidate_analysis.py \
  --csv outputs/experiments/2026-11-04_paired-and-kl/checks/kl_benefit_server/kl_benefit.csv \
  --output-dir outputs/experiments/2026-11-05_same-target-candidates/checks/same_target/e5-round1-base-s0 \
  --label '5 epoch / round_1 / base / seed0 / 主口径'
```

短程重放（B）：

```bash
python scripts/same_target_replay.py \
  --checkpoint outputs/experiments/2026-11-05_same-target-candidates/checkpoints/tagsrv20-e5-r10-base-s0/round_10 \
  --data_dir c:/Users/Administrator/Desktop/bishe/datasets/cifar10 \
  --output outputs/experiments/2026-11-05_same-target-candidates/checks/replay-r10-base \
  --targets 1,2,3,4,5,7,9,10,14,15,18 --max-candidates 6 --replay-epochs 1
```

实验一交付（图与 Word）：

```bash
python scripts/make_run020_figures.py
python scripts/build_run020_docx.py
```
