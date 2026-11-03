# NLP-TextCNN_reproduction

课程项目 P01：复现 Yoon Kim (EMNLP 2014) *Convolutional Neural Networks for Sentence Classification* 在 MR 电影评论数据集上的 TextCNN 情感分类实验。

- 原论文：<https://aclanthology.org/D14-1181/>（ACL Anthology D14-1181，EMNLP 2014, pp. 1746–1751）
- 作者代码与数据：<https://github.com/yoonkim/CNN_sentence>
- **中期报告：[`docs/midterm_report.md`](docs/midterm_report.md)** ← 建议先读这份

## 快速结论

| 设置 | 窗口 | 种子 | Accuracy 均值 | Macro-F1 均值 |
|---|---:|---:|---:|---:|
| CNN-rand | 3,4,5 | 13/42/2026 | 0.7172 | 0.7163 |
| CNN-static（SVD 冻结） | 3,4,5 | 13/42/2026 | 0.7094 | 0.7060 |
| **CNN-non-static（SVD 微调）** | 3,4,5 | 13/42/2026 | **0.7499** | **0.7496** |
| CNN-non-static 窗口消融 | 2,4,5 | 13/42/2026 | 0.7571 | 0.7566 |

> 本组为**缩规模复现**：固定 80/10/10 划分（原文 10 折 CV），50 维训练集 SVD 词向量替代原文 300 维 Google News word2vec，CPU、5 epoch。**结果不能与原文表 2 的 76.1 / 81.0 / 81.5 直接对比。**

## 目录结构

```text
docs/midterm_report.md      # 中期报告（合并精简版）
code/textcnn_experiment.py  # 数据准备 + 词表 + TextCNN + 训练/评估/落盘
configs/                    # 三种设置的配置快照
data/README.md              # 数据来源与划分说明
data/processed/             # 固定划分数据 + manifest.json（SHA-256）
results/                    # 汇总表、训练日志、逐条预测、案例
contributions.md            # 成员贡献与 AI 使用说明
```

## 快速开始

```powershell
pip install -r requirements.txt

# 数据准备（下载作者仓库 MR 语料并生成固定划分）
python .\code\textcnn_experiment.py --prepare-data-only

# 三种词向量设置 × 3 个种子
python .\code\textcnn_experiment.py --mode rand       --seeds 13 42 2026 --epochs 5
python .\code\textcnn_experiment.py --mode static     --seeds 13 42 2026 --epochs 5
python .\code\textcnn_experiment.py --mode non_static --seeds 13 42 2026 --epochs 5

# 窗口单因素实验
python .\code\textcnn_experiment.py --mode non_static --seeds 13 42 2026 --epochs 5 --filter-sizes 2 4 5
```

## 结果与证据入口

| 文件 | 说明 |
|---|---|
| `results/summary.md` / `summary_by_mode.csv` | 按条件聚合（均值 / 范围） |
| `results/all_runs.csv` | 12 次运行明细（含耗时） |
| `results/<run_id>/train_log.jsonl` | 逐 epoch 训练日志 |
| `results/<run_id>/test_predictions.csv` | 测试集 1069 条逐条预测（id / text / label / pred / prob_pos） |
| `results/cases.csv` / `cases.md` | 代表性失败、修正与回退案例 |

## 环境

Python 3.12 + `torch` / `scikit-learn` / `numpy` / `pandas` / `requests`。普通 CPU 即可完成全部 12 次运行（合计约 159 s）。

## 数据与合规

MR 语料（`rt-polarity.pos` / `rt-polarity.neg`）来自作者公开仓库；仓库内不含受限数据、模型权重或密钥。AI 使用说明见 `contributions.md`。
