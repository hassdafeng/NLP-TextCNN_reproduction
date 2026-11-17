# NLP-TextCNN_reproduction

课程项目 P01：复现 Yoon Kim (EMNLP 2014) *Convolutional Neural Networks for Sentence Classification* 在 MR 电影评论数据集上的 TextCNN 情感分类实验。

- 原论文：<https://aclanthology.org/D14-1181/>（ACL Anthology D14-1181，EMNLP 2014, pp. 1746–1751）
- 作者代码与数据：<https://github.com/yoonkim/CNN_sentence>
- 中期报告：[`docs/midterm_report.md`](docs/midterm_report.md)
- 数据修正后的核验：[`docs/mr_lf_verification.md`](docs/mr_lf_verification.md)

## 中期结果

| 设置 | 窗口 | 种子 | Accuracy 均值 | Macro-F1 均值 |
|---|---:|---:|---:|---:|
| CNN-rand | 3,4,5 | 13/42/2026 | 0.7172 | 0.7163 |
| CNN-static（SVD 冻结） | 3,4,5 | 13/42/2026 | 0.7094 | 0.7060 |
| **CNN-non-static（SVD 微调）** | 3,4,5 | 13/42/2026 | **0.7499** | **0.7496** |
| CNN-non-static 窗口消融 | 2,4,5 | 13/42/2026 | 0.7571 | 0.7566 |

> 本组为**缩规模复现**：固定 80/10/10 划分（原文 10 折 CV），50 维训练集 SVD 词向量替代原文 300 维 Google News word2vec，CPU、5 epoch。**结果不能与原文表 2 的 76.1 / 81.0 / 81.5 直接对比。**

以上是原中期划分的历史结果。修正后实验用同一份 10,662 条语料的新划分，按原配置重跑。新结果见 [`results/mr_lf/summary.md`](results/mr_lf/summary.md)，新旧划分的样本组成不同。

## 目录结构

```text
docs/midterm_report.md      # 中期报告（合并精简版）
code/textcnn_experiment.py  # 数据准备 + 词表 + TextCNN + 训练/评估/落盘
configs/                    # 三种设置的配置快照
data/README.md              # 数据来源与划分说明
data/processed/             # 中期历史划分
data/processed_lf/          # 修正后的固定划分、样本 ID 与校验清单
results/                    # 中期结果及修正后的 mr_lf/ 子目录
```

## 快速开始

```powershell
# 本次复跑环境为 Windows x64、Python 3.12，使用锁定依赖
python -m pip install -r results/mr_lf/environment.txt
$env:OMP_NUM_THREADS='3'
$env:MKL_NUM_THREADS='3'
$env:OPENBLAS_NUM_THREADS='3'

# 数据准备（下载作者仓库 MR 语料并生成固定划分）
python .\code\textcnn_experiment.py --prepare-data-only

# 三种词向量设置 × 3 个种子
python .\code\textcnn_experiment.py --mode rand       --seeds 13 42 2026 --epochs 5
python .\code\textcnn_experiment.py --mode static     --seeds 13 42 2026 --epochs 5
python .\code\textcnn_experiment.py --mode non_static --seeds 13 42 2026 --epochs 5

# 窗口单因素实验
python .\code\textcnn_experiment.py --mode non_static --seeds 13 42 2026 --epochs 5 --filter-sizes 2 4 5

# 从保存的划分、日志和逐样本预测复算，不训练模型
python .\code\verify_mr_lf.py
```

## 修正后的结果与证据

| 文件 | 说明 |
|---|---|
| [`results/mr_lf/summary.md`](results/mr_lf/summary.md) / `summary_by_mode.csv` | 按条件聚合（均值 / 范围） |
| [`results/mr_lf/all_runs.csv`](results/mr_lf/all_runs.csv) | 12 次运行明细，含数据清单摘要和耗时 |
| `results/mr_lf/<run_id>/train_log.jsonl` | 逐 epoch 训练日志 |
| `results/mr_lf/<run_id>/test_predictions.csv` | 测试集 1,067 条逐条预测 |
| [`results/mr_lf/verification.json`](results/mr_lf/verification.json) / [`cases.csv`](results/mr_lf/cases.csv) | 独立复算、窗口配对与案例 |

## 环境

Python 3.12 + `torch` / `scikit-learn` / `numpy` / `pandas` / `requests`。本次运行使用 CPU，主要版本和完整依赖见核验记录及环境清单。

## 数据与合规

MR 语料（`rt-polarity.pos` / `rt-polarity.neg`）来自作者公开仓库；仓库内不含受限数据、模型权重或密钥。
