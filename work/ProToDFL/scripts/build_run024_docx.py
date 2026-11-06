"""论文级配置复现第一趟：纯数据报告 docx。

数据全部来自 outputs/paper100/detail.json，不做结论性解读。
"""
from __future__ import annotations

import json
from pathlib import Path

from docx_helpers import bullet, caption, h1, new_doc, subtitle, table, title

ROOT = Path(__file__).resolve().parents[1]
DETAIL = json.loads((ROOT / "outputs/paper100/detail.json").read_text(encoding="utf-8"))
DOCS = ROOT.parents[1] / "docs/2026-11-06"
TARGET = DOCS / "22_论文级复现第一趟_数据报告_2026-11-06.docx"

R = {k: DETAIL[f"r{k}"] for k in (1, 250, 500)}


def pp(x, sign=True):
    return f"{x:+.2f}" if sign else f"{x:.2f}"


def pct(x):
    return f"{x * 100:.2f}%"


def rho(x):
    return f"{x:+.3f}" if x == x else "-"


def build():
    doc = new_doc()
    title(doc, "论文级配置复现 · 第一趟 数据报告")
    subtitle(doc, "RUN-024　|　2026-11-05 训练，2026-11-06 离线评估　|　CIFAR-10 / α=0.3 / 100 客户端 / 500 轮")

    h1(doc, "1 实验配置")
    table(doc, ["项", "论文设置", "本次设置"], [
        ["数据集", "CIFAR-10", "CIFAR-10"],
        ["客户端数", "100", "100"],
        ["数据划分", "Dirichlet α=0.3，10 组组内同质", "my_part，α=0.3，10 组"],
        ["拓扑", "随机连通图，每点 10 邻居", "cs=random，10 邻居"],
        ["通信轮数", "500", "500（501 次迭代）"],
        ["本地 epoch", "5", "5"],
        ["batch size", "128", "128"],
        ["优化器 / 学习率", "SGD 0.1", "SGD 0.1，lr_decay=1.0"],
        ["共享范围", "只共享特征提取器", "aggregate_scope=base"],
        ["快照轮次", "—", "r1 / r250 / r500"],
        ["原型刷新", "—", "proto_update_each=0（既定口径）"],
        ["原型动量", "—", "0.1（代码默认）"],
    ], widths=[3.2, 6.4, 6.4], size=8.5)
    caption(doc, "表 1  实验配置。前 10 项逐条对照 ICME 论文 Experiment Setup；后 2 项为论文未给值的实现细节。")

    h1(doc, "2 运行记录")
    table(doc, ["项", "值"], [
        ["训练进度", "501 / 501，0 报错"],
        ["总耗时", "14 h 25 m 33 s（103.66 s/it）"],
        ["硬件", "单卡 3090，峰值显存 5976 MiB"],
        ["快照", "r1 / r250 / r500，各 201 文件、各 4.2G"],
        ["离线评估", "每快照 100 目标 × 20 候选 = 2000 对，约 5 分钟"],
        ["评估口径", "单伙伴、只聚合 base、Δacc = 聚合后 − 聚合前"],
        ["数据文件", "outputs/paper100/r{1,250,500}.json，detail.json"],
    ], widths=[4.0, 12.0], size=9)
    caption(doc, "表 2  运行记录。三个快照已按要求在评估完成后删除。")

    h1(doc, "3 三快照总体指标")
    rows = [
        ["配对数 / 目标数"] + [f"{R[k]['n_pairs']} / {R[k]['n_targets']}" for k in R],
        ["有效 KL 配对"] + [str(R[k]["n_valid_kl"]) for k in R],
        ["退化头比例"] + [pct(R[k]["degenerate_frac"]) for k in R],
        ["聚合前精度（均值）"] + [f"{R[k]['pre_acc_mean']:.3f}" for k in R],
        ["聚合前精度范围"] + [f"{R[k]['pre_acc_min']:.3f}–{R[k]['pre_acc_max']:.3f}" for k in R],
        ["收益均值 Δacc (pp)"] + [pp(R[k]["delta_mean_pp"]) for k in R],
        ["收益中位数 (pp)"] + [pp(R[k]["delta_median_pp"]) for k in R],
        ["收益标准差 (pp)"] + [pp(R[k]["delta_std_pp"], False) for k in R],
        ["收益范围 (pp)"] + [f"{pp(R[k]['delta_min_pp'])} ~ {pp(R[k]['delta_max_pp'])}" for k in R],
        ["零 / 正 / 负迁移 (%)"] + [
            f"{pct(R[k]['zero_frac'])[:-1]} / {pct(R[k]['positive_frac'])[:-1]} / {pct(R[k]['negative_frac'])[:-1]}"
            for k in R
        ],
        ["混算 Spearman"] + [rho(R[k]["overall_rho"]) for k in R],
        ["目标内 Spearman 均值"] + [rho(R[k]["mean_target_rho"]) for k in R],
        ["目标内 ρ 正 / 负 / 可算"] + [
            f"{R[k]['n_targets_rho_positive']} / {R[k]['n_targets_rho_negative']} / {R[k]['n_targets_evaluated']}"
            for k in R
        ],
        ["KL 首选命中"] + [f"{R[k]['top1_hit']}/100" for k in R],
        ["首选 regret 均值 (pp)"] + [pp(R[k]["top1_regret_mean_pp"], False) for k in R],
        ["首选 regret 中位 (pp)"] + [pp(R[k]["top1_regret_median_pp"], False) for k in R],
    ]
    table(doc, ["指标", "r1", "r250", "r500"], rows, widths=[6.0, 3.4, 3.4, 3.4], size=8.5)
    caption(doc, "表 3  三快照总体指标。20 个候选下随机命中基线为 5%。主统计基于全部 2000 行，未套用退化头过滤。")

    h1(doc, "4 聚合前预测类别数分布")
    pred_rows = []
    for n in range(1, 9):
        pred_rows.append([str(n)] + [str(R[k]["unique_pred_dist"].get(str(n), 0)) for k in R])
    table(doc, ["预测类别数", "r1", "r250", "r500"], pred_rows,
          widths=[4.0, 4.0, 4.0, 4.0], size=9)
    caption(doc, "表 4  聚合前模型在目标客户端留出验证集上预测出的不同类别数（每快照 2000 条配对记录）。")

    h1(doc, "5 KL 五分箱")
    bins = R[1]["kl_bins"]
    bin_rows = []
    for i, b in enumerate(bins):
        rng = f"[{b['kl_min']:.2f}, {b['kl_max']:.2f}]"
        bin_rows.append([
            rng,
            str(b["n"]),
            pp(R[1]["kl_bins"][i]["mean_delta_pp"]),
            pp(R[250]["kl_bins"][i]["mean_delta_pp"]),
            pp(R[500]["kl_bins"][i]["mean_delta_pp"]),
            pct(R[500]["kl_bins"][i]["zero_frac"]),
        ])
    table(doc, ["KL 区间", "配对数", "r1 Δacc (pp)", "r250 Δacc (pp)", "r500 Δacc (pp)", "r500 零收益"],
          bin_rows, widths=[3.8, 1.6, 2.6, 2.6, 2.6, 2.8], size=8.5)
    caption(doc, "表 5  KL 五分箱平均收益。三个快照的分箱边界完全一致。")

    h1(doc, "6 同组与跨组")
    same, cross = R[1]["same_group"], R[1]["cross_group"]
    table(doc, ["类别", "配对数", "平均 KL", "r1 Δacc (pp)", "r250 Δacc (pp)", "r500 Δacc (pp)", "r500 负迁移"], [
        ["同组", same["n"], f"{same['mean_kl']:.2f}"] + [pp(R[k]["same_group"]["mean_delta_pp"]) for k in R]
        + [pct(R[500]["same_group"]["negative_frac"])],
        ["跨组", cross["n"], f"{cross['mean_kl']:.2f}"] + [pp(R[k]["cross_group"]["mean_delta_pp"]) for k in R]
        + [pct(R[500]["cross_group"]["negative_frac"])],
    ], widths=[2.0, 1.8, 2.2, 2.5, 2.5, 2.5, 2.5], size=8.5)
    caption(doc, "表 6  my_part 的 10 组划分下，同组与跨组候选的平均 KL 及收益。平均 KL 在三个快照相同。")

    h1(doc, "7 按图跳数")
    hop_map = {
        k: {h["hop"]: h for h in R[k]["hop_stats"]} for k in R
    }
    hop_rows = []
    for hop in range(1, 9):
        cells = [str(hop)]
        for k in R:
            h = hop_map[k].get(hop)
            cells.append(f"{h['n']} / {pp(h['mean_delta_pp'])}" if h else "—")
        hop_rows.append(cells)
    table(doc, ["跳数", "r1 (n / Δacc pp)", "r250 (n / Δacc pp)", "r500 (n / Δacc pp)"],
          hop_rows, widths=[2.2, 4.6, 4.6, 4.6], size=8.5)
    caption(doc, "表 7  候选在图中的最短跳数与平均收益。r1 的跳数分布为 1–8，r250 / r500 为 1–4。")

    h1(doc, "8 口径")
    for text in [
        "Δacc 在目标客户端 20% 留出验证集上测量；聚合前精度是本地偏斜验证集精度，不是全局测试精度。",
        "单伙伴、只聚合特征提取器；每目标从其余 99 个客户端中按固定种子随机抽 20 个候选。",
        "主统计基于全部 2000 行重算，未套用评估脚本的退化头过滤；退化比例单独列出。",
        "三个快照的 2000 对 KL 逐对相同（0 个不同），原型按既定口径保持初始化后固定。",
    ]:
        bullet(doc, text, size=9.5)

    h1(doc, "9 文件")
    for text in [
        "原始数据：work/ProToDFL/outputs/paper100/r{1,250,500}.json（及同名 CSV）",
        "汇总：work/ProToDFL/outputs/paper100/detail.json",
        "统计脚本：work/ProToDFL/scripts/analyze_paper100_detail.py",
        "完整实验记录：docs/2026-11-06/21_论文级复现第一趟_2026-11-06.md",
    ]:
        bullet(doc, text, size=9.5)

    DOCS.mkdir(parents=True, exist_ok=True)
    doc.save(TARGET)
    print("written:", TARGET)


if __name__ == "__main__":
    build()
