# P01 TextCNN 复现中期报告

---

## 摘要

1. 复现对象：Yoon Kim (EMNLP 2014)《Convolutional Neural Networks for Sentence Classification》第 3.3 节模型变体与表 2 中 MR 数据集上的 CNN-rand / CNN-static / CNN-non-static 三种词向量设置。
2. 数据与实现：从作者仓库获取 MR 原始语料，固定 80/10/10 分层划分，用 PyTorch 重写 TextCNN 全流程（数据准备 → 词表 → 训练 → 验证选模型 → 测试 → 逐条预测落盘）。
3. 已有结果（3 个随机种子 13/42/2026）：CNN-non-static 最好，Accuracy 0.7499、Macro-F1 0.7496；CNN-rand 0.7172 / 0.7163；CNN-static 0.7094 / 0.7060；窗口 `[2,4,5]` 的 non-static 为 0.7571 / 0.7566。
4. 关键判断：在本组缩规模设置下，"词向量可训练"优于"随机初始化"和"冻结词向量"；但 CNN-static 未超过 CNN-rand，与原文趋势不一致，原因是使用了 50 维训练集 SVD 词向量而非原文的 300 维 Google News word2vec。
5. 当前结果属于**缩规模 + 替代词向量复现**，不能与原文表 2 的 76.1 / 81.0 / 81.5 直接对齐；下一项单因素实验为卷积窗口 `[3,4,5]` vs `[2,4,5]` 的正式对照。

---

## 1. 原论文与仓库代码位置

### 1.1 原论文

| 项 | 内容 |
|---|---|
| 题目 | Convolutional Neural Networks for Sentence Classification |
| 作者 | Yoon Kim（New York University） |
| 会议 | EMNLP 2014, pages 1746–1751 |
| 论文编号 | ACL Anthology **D14-1181** |
| 在线地址 | <https://aclanthology.org/D14-1181/> |
| 本地文件 | `D14-1181.pdf`（与复现目录同级） |
| 复现目标位置 | 第 3.3 节 Model Variations + 表 2 的 **MR 列** |

### 1.2 作者代码与数据

| 项 | 内容 |
|---|---|
| 作者代码仓库 | <https://github.com/yoonkim/CNN_sentence>（Theano 实现） |
| 作者数据 | 仓库内 `rt-polarity.pos` / `rt-polarity.neg`（Pang & Lee 2005 MR 电影评论） |
| 本组实验脚本 | [`code/textcnn_experiment.py`](../code/textcnn_experiment.py)（PyTorch 重新实现） |
| 本组结果入口 | [`results/summary.md`](../results/summary.md)、[`results/summary_by_mode.csv`](../results/summary_by_mode.csv) |

### 1.3 原文目标数值（表 2，MR 列，Accuracy %）

| 模型 | MR |
|---|---:|
| CNN-rand | 76.1 |
| CNN-static | 81.0 |
| CNN-non-static | **81.5** |
| CNN-multichannel | 81.1 |

原文在 MR 上使用 10 折交叉验证（表 1 中 MR 的 Test 一栏标注 CV），词向量为 Google News word2vec（1000 亿词、300 维、CBOW）。

---

## 2. 复现内容与范围

### 2.1 已完成的工作

1. **论文研读与范围界定**：只复现 MR 数据集上的三种词向量设置 + 一个卷积窗口单因素实验，不追全部 7 个数据集。
2. **数据获取与固定划分**：脚本自动下载作者仓库的 MR 语料，用 `split_seed=2026` 做分层 80/10/10 划分，保存样本 ID 与 `manifest.json`（含原始文件 SHA-256），保证可复查。
3. **PyTorch 版 TextCNN**：嵌入层 → 三种窗口卷积 → ReLU → max-over-time pooling → dropout → 全连接 → softmax，串成数据准备、训练、验证选模、测试评估、逐条预测落盘一条龙。
4. **三种词向量设置 × 3 个随机种子（13/42/2026）× 5 epoch**：共 3 组主对照 + 1 组窗口消融，12 次运行全部完成。
5. **结果归档与错误分析**：汇总表、单次训练日志、测试集逐条预测（含正类概率）、代表性失败/边界案例。

### 2.2 明确不做（超出范围）

- 原文其余 6 个数据集（SST-1/2、Subj、TREC、CR、MPQA）；
- CNN-multichannel 变体；
- MR 的 10 折交叉验证；
- 论文的全量超参网格搜索。

### 2.3 与原文的关系

**缩规模复现**：保留模型结构与主要超参，但在数据划分和词向量上做了适配普通 CPU 的替换，因此结论只用于"趋势与相对差异"的对照，不用于复现原文绝对分数。

---

## 3. 方法：TextCNN 与三种词向量设置

### 3.1 模型流程

```
句子 → 分词 → 词 ID 序列（截断/补齐到 64）
     → 词向量矩阵 E ∈ R^{|V|×50}
     → Conv1d（窗口 3/4/5，每种 100 个特征图）→ ReLU
     → 沿时间维 max-pooling（max-over-time）
     → 拼接 300 维 → Dropout(0.5) → Linear(300→2) → Softmax
```

### 3.2 三种设置（对应原文 §3.3）

| 设置 | 词向量初始化 | 是否训练 | 本组实现 |
|---|---|---|---|
| CNN-rand | 随机 `U(-0.25, 0.25)` | 是 | `random_trainable` |
| CNN-static | 训练集 SVD 词向量 | 否（冻结） | `local_svd_frozen` |
| CNN-non-static | 训练集 SVD 词向量 | 是（微调） | `local_svd_trainable` |

SVD 词向量的构造方式：对训练集文本做 TF-IDF（词 × 文档矩阵转置），用 `TruncatedSVD` 降到 50 维，再逐词 L2 归一化；`<unk>` 用其余词向量的均值初始化。**该词向量只在训练集上拟合**，验证/测试集不参与。

### 3.3 实现要点

- 优化器 Adadelta（lr = 1.0），与原文一致；batch 50；dropout 0.5；窗口 3/4/5 × 100 特征图；
- 每个 epoch 在验证集上算 Macro-F1，保存**验证集 Macro-F1 最高**的模型权重，最后只在测试集评估一次；
- 随机性控制：`random / numpy / torch` 全局种子 + DataLoader generator 种子，均跟随实验种子。

---

## 4. 实验设置差异（原文 vs 本组）

| 维度 | 原文（Kim 2014） | 本组 | 影响 |
|---|---|---|---|
| 数据集规模 | MR 10662 句 | 原始 10662 句，处理后可训练样本 10685 条（含切分碎片，见 §10.1） | 轻微 |
| 数据划分 | 10 折交叉验证 | 固定 80/10/10 分层划分（seed 2026）：8548 / 1068 / 1069 | 指标不可与原文直接比 |
| 词向量来源 | Google News word2vec（100B 词、CBOW） | 训练集无监督 TF-IDF + SVD | **最大差异** |
| 词向量维度 | 300 | 50 | 表征能力明显下降 |
| 词表大小 | 18765（含预训练覆盖） | 16587（min_freq = 1，仅训练集） | 轻微 |
| 句子长度 | 未截断（MR 平均 20 词） | 截断/补齐至 64 | 轻微 |
| 卷积窗口 | 3/4/5 | 3/4/5（消融改为 2/4/5） | 一致/单因素 |
| 每窗口特征图 | 100 | 100 | 一致 |
| Dropout | 0.5 | 0.5 | 一致 |
| L2 约束 | 所有词向量权重范数 ≤ 3 | 仅全连接层权重 max-norm ≤ 3（实现简化） | 轻微 |
| Batch size | 50 | 50 | 一致 |
| 优化器 | Adadelta | Adadelta（lr = 1.0） | 一致（lr 未细调） |
| 训练轮数 | early stopping（dev 上） | 固定 5 epoch + 按 dev Macro-F1 选模型 | 训练预算更小 |
| 评价指标 | Accuracy | Accuracy + Macro-F1（均值与范围） | 增加类别均衡视角 |
| 设备 | GPU | 普通 CPU（12 次运行合计约 159 s，单次 9.8–15.3 s） | 限制规模 |

**一句话结论**：结构、窗口、特征图数、dropout、batch、优化器与原文一致；词向量（来源 + 维度）、数据划分（固定划分 vs CV）、训练预算（5 epoch）是三项主要差异。

---

## 5. 数据与实验流程

### 5.1 数据

- 来源：`https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.pos|neg`
- 原始文件 SHA-256：
  - `rt-polarity.neg` = `4ace77d558c3714723843f1d65b60c01e3417b208180f0728808d76ad0eeeaca`
  - `rt-polarity.pos` = `2da124ec187a9d5a29c9f04e91c540e02baed5af8868f550a26bd6fd4dbf8bf0`
- 划分（`data/processed/manifest.json`）：

| 集合 | 条数 | 负例 | 正例 |
|---|---:|---:|---:|
| train | 8548 | 4273 | 4275 |
| dev | 1068 | 534 | 534 |
| test | 1069 | 534 | 535 |

### 5.2 运行命令

```powershell
# 只准备数据
python .\code\textcnn_experiment.py --prepare-data-only

# 三组主对照（每组 3 个种子）
python .\code\textcnn_experiment.py --mode rand       --seeds 13 42 2026 --epochs 5
python .\code\textcnn_experiment.py --mode static     --seeds 13 42 2026 --epochs 5
python .\code\textcnn_experiment.py --mode non_static --seeds 13 42 2026 --epochs 5

# 窗口单因素实验（只改窗口）
python .\code\textcnn_experiment.py --mode non_static --seeds 13 42 2026 --epochs 5 --filter-sizes 2 4 5
```

### 5.3 每次运行的产物

```
results/<run_id>/
  config.json            # 完整超参（含 max_len、lr、split_seed 等）
  train_log.jsonl        # 每个 epoch 的 train_loss / dev_accuracy / dev_macro_f1
  summary.json           # 本次运行的测试集指标与耗时
  test_predictions.csv   # 测试集逐条预测（核心“逐条输出”）
results/all_runs.csv           # 12 次运行明细
results/summary_by_mode.csv    # 按条件聚合（均值 / 最小 / 最大）
results/summary.md             # 人读版汇总表
results/cases.csv, cases.md    # 代表性案例
```

---

## 6. 已有结果（三种设置的并排对照）

### 6.1 主结果（3 个种子的均值与范围）

| 条件 | 窗口 | 种子 | Accuracy 均值 | Accuracy 范围 | Macro-F1 均值 | Macro-F1 范围 | 平均耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN-rand | 3,4,5 | 13/42/2026 | 0.7172 | 0.6932–0.7362 | 0.7163 | 0.6926–0.7361 | 14.8 s |
| CNN-static（SVD 冻结） | 3,4,5 | 13/42/2026 | 0.7094 | 0.6950–0.7287 | 0.7060 | 0.6946–0.7287 | 10.8 s |
| **CNN-non-static（SVD 微调）** | 3,4,5 | 13/42/2026 | **0.7499** | 0.7474–0.7540 | **0.7496** | 0.7468–0.7540 | 15.2 s |
| CNN-non-static 窗口消融 | 2,4,5 | 13/42/2026 | 0.7571 | 0.7502–0.7615 | 0.7566 | 0.7499–0.7614 | 12.1 s |

### 6.2 单次运行明细（`results/all_runs.csv`）

| run_id | seed | Accuracy | Macro-F1 | 耗时 (s) |
|---|---:|---:|---:|---:|
| rand_seed13 | 13 | 0.7362 | 0.7361 | 14.75 |
| rand_seed42 | 42 | 0.7222 | 0.7204 | 14.76 |
| rand_seed2026 | 2026 | 0.6932 | 0.6926 | 14.75 |
| static_seed13 | 13 | 0.7287 | 0.7287 | 10.80 |
| static_seed42 | 42 | 0.6950 | 0.6947 | 10.98 |
| static_seed2026 | 2026 | 0.7044 | 0.6946 | 10.75 |
| non_static_seed13 | 13 | 0.7484 | 0.7481 | 15.30 |
| non_static_seed42 | 42 | 0.7474 | 0.7468 | 15.27 |
| non_static_seed2026 | 2026 | 0.7540 | 0.7540 | 15.14 |
| non_static_w245_seed13 | 13 | 0.7596 | 0.7586 | 14.51 |
| non_static_w245_seed42 | 42 | 0.7502 | 0.7499 | 11.95 |
| non_static_w245_seed2026 | 2026 | 0.7615 | 0.7614 | 9.82 |

### 6.3 观察

1. **CNN-non-static 稳定最优**：均值 0.7499 高于 rand（+3.3 pp）与 static（+4.1 pp）；种子间标准差仅 0.35 pp，是四组里最稳的。
2. **CNN-static 没有超过 CNN-rand**：与原文（81.0 vs 76.1）方向相反。合理解释是 50 维 SVD 词向量承载的语义信息远弱于 300 维 word2vec，冻结后模型无法补偿；而 rand 虽然初始化差，但 8548 条训练样本足以让 50 维词向量自由适配任务。
3. **rand / static 种子波动大**（标准差 2.2 pp / 1.7 pp），说明这两种设置对初始化/优化路径敏感；non-static 的稳定性本身可以视为一条有意义的结论。
4. **窗口消融初值**：`[2,4,5]` 相对 `[3,4,5]`，Accuracy +0.72 pp、Macro-F1 +0.70 pp，且 **3 个种子方向全部为正**（+1.12 / +0.28 / +0.75 pp）；但幅度与种子波动同量级，暂不宣称显著，详见 §9。

---

## 7. 运行生成的逐条输出

### 7.1 文件与字段

每个运行目录下的 `test_predictions.csv` 是测试集 1069 条的逐条预测，字段：

| 字段 | 含义 |
|---|---|
| `id` | 样本 ID（如 `mr-neg-00038`，与 `data/processed/test.csv` 对齐） |
| `text` | 原始句子 |
| `label` | 金标（0 = 负例，1 = 正例） |
| `pred_label` | 模型预测标签 |
| `prob_pos` | 判为正类的 softmax 概率 |
| `config` | 条件名（rand / static / non_static） |
| `seed` | 随机种子 |

示例路径：[`results/non_static_seed2026/test_predictions.csv`](../results/non_static_seed2026/test_predictions.csv)。

### 7.2 真实抽样（`non_static_seed2026` 与 `non_static_w245_seed2026`）

| 样本 ID | 金标 | non-static 预测 | P(正) | w245 预测 | P(正) | 句子（截断） |
|---|---:|---:|---:|---:|---:|---|
| `mr-pos-00025` | 1 | 1 | 0.7850 | 1 | 0.9063 | occasionally melodramatic , it's also extremely effective . |
| `mr-neg-00000` | 0 | 0 | 0.0056 | 0 | 0.0028 | simplistic , silly and tedious . |
| `mr-pos-00000` | 1 | 0 | 0.1474 | 0 | 0.2465 | the rock is destined to be the 21st century's new " conan " … |
| `mr-neg-00038` | 0 | 1 | 0.6677 | 1 | 0.6769 | … if you're just in the mood for a fun -- but bad -- movie … |
| `mr-neg-00128` | 0 | 1 | 0.5731 | 0 | 0.2377 | none of this is half as moving as the filmmakers seem to think . |
| `mr-neg-00213` | 0 | 0 | 0.3877 | 1 | 0.5181 | [n]o matter how much good will the actors generate … |
| `mr-neg-00325` | 0 | 0 | 0.4814 | 1 | 0.6792 | … a movie comes along to remind us of how very bad a motion picture can truly be … |

> 说明：以上记录直接取自逐条预测文件，未做任何改写；概率值可在对应 CSV 中按 `id` 检索。

### 7.3 如何复查

```powershell
python -c "import pandas as pd; d=pd.read_csv('results/non_static_seed2026/test_predictions.csv'); print(d[d.id.isin(['mr-neg-00038','mr-neg-00128'])])"
```

---

## 8. 两例失败或边界

### 8.1 案例 A（失败）：`mr-neg-00038` —— 转折结构被局部正向词盖过

| 项 | 内容 |
|---|---|
| 文本 | `. . . if you're just in the mood for a fun -- but bad -- movie , you might want to catch freaks as a matinee .` |
| 金标 | 负例（0） |
| 各设置预测 | CNN-rand = 0 ✅；CNN-static = 0 ✅；**CNN-non-static = 1 ❌**（P(正) = 0.6677）；窗口 2,4,5 = 1 ❌（0.6769） |
| 失败类型 | 转折 / 让步结构：句中有 "fun" 与 "bad" 两个极性冲突词，最终语义由转折词 "but" 决定 |

**分析**：max-over-time pooling 只保留每个窗口内最强的激活，而 "fun" 这类强正向词的局部 n-gram 激活强度高于 "but bad" 的转折组合；50 维 SVD 向量对 "but" 的句法/逻辑作用刻画不足，导致两种可训练设置都被带偏。冻结词向量的 static 反而判对，说明这条样本上"更新词向量"并非总是加分。

**改进方向**：引入转折词表或对否定/转折标记做位置编码；也可以在词向量层面替换为含更丰富句法信息的预训练向量。

### 8.2 案例 B（边界）：`mr-neg-00128` —— 否定被短窗口修正，但同类回退同时出现

| 项 | 内容 |
|---|---|
| 文本 | `none of this is half as moving as the filmmakers seem to think .` |
| 金标 | 负例（0） |
| 各设置预测 | CNN-non-static = 1 ❌（P(正) = 0.5731）；**窗口 2,4,5 = 0 ✅**（0.2377） |
| 失败类型 | 否定 + 比较结构（"none of this is half as moving as…"），字面有 "moving"（感人）正向词 |

**分析**：默认窗口最小为 3，`none of this`（3 词）刚好可被覆盖但 "none of" 与 "is half as moving" 的组合更靠 2 词窗口；加入宽度 2 后，像 "none of"、"not half" 这类否定短语能被直接捕捉，因此该样本被修正（概率从 0.5731 降到 0.2377）。

**边界证据**：窗口 2 并非只带来收益。同一设置下：

- `mr-neg-00213`：non-static 判对（0.3877）→ w245 判错（0.5181）；
- `mr-neg-00325`：non-static 判对（0.4814）→ w245 判错（0.6792）。

三条 `window_fix` 与三条 `window_regression`（另见 `mr-neg-00200`、`mr-neg-00420`、`mr-neg-00222`）数量相当，这解释了为什么 `[2,4,5]` 的整体提升只有 0.7 pp：**收益来自净修正的样本数，而非每条样本都变好**。该现象是下一项单因素实验要重点验证的对象。

---

## 9. 下一项单因素实验：卷积窗口 `[3,4,5]` vs `[2,4,5]`

### 9.1 设计（严格控制变量）

| 变量 | 设置 |
|---|---|
| 唯一变化 | `filter_sizes`：`[3,4,5]` → `[2,4,5]` |
| 固定不变 | 数据划分（seed 2026）、词向量构造、embedding 维度 50、dropout、batch、Adadelta、epochs = 5、模型选择规则 |
| 重复 | 每种窗口各 3 个种子（已完成初跑） |
| 指标 | Accuracy、Macro-F1 的均值 ± 标准差、配对差值；错误类型计数（否定/转折样本的修正与回退数） |

### 9.2 现状（已完成初跑）

| 种子 | `[3,4,5]` Acc | `[2,4,5]` Acc | 配对差值 |
|---|---:|---:|---:|
| 13 | 0.7484 | 0.7596 | +1.12 pp |
| 42 | 0.7474 | 0.7502 | +0.28 pp |
| 2026 | 0.7540 | 0.7615 | +0.75 pp |
| 均值 | 0.7499 | 0.7571 | **+0.72 pp** |

Macro-F1 同样为 +0.70 pp（0.7496 → 0.7566）。三个种子方向一致，但幅度小于种子间波动（w245 组标准差 0.60 pp），因此**当前只能判为"有正向趋势，未达显著"**。

### 9.3 判定标准

1. 把种子扩到 5–8 个，计算配对差值的均值与 95% 区间；若均值下界 > 0，则判定稳定收益。
2. 固定其他变量，把 epochs 从 5 增加到 10 复跑一次，排除"欠拟合使窗口差异被放大/掩盖"的混杂；训练日志显示 5 epoch 时 train_loss 仍从 0.658 降到 0.337、dev 指标仍在上行，欠拟合风险确实存在。
3. 追加 `[2,3,4]` 做二次单因素，拆解收益到底来自"最短窗口变短"还是"去掉窗口 5"，避免把两个变化混在一起解释。
4. 用案例计数验证：若 `[2,4,5]` 的净收益确实来自否定短语，则 `window_fix` 数应显著多于 `window_regression` 数（当前为 3 : 3，样本量太小）。

### 9.4 预算与产出

- CPU 单次运行约 10–15 s；8 种子 × 2 窗口 × 10 epoch ≈ 40 分钟，可在本机完成。
- 产出：更新 `results/summary_by_mode.csv`、`results/summary.md`、`results/cases.csv`，并在最终报告中单列一节"窗口消融"。

---

## 10. 已知问题、风险与边界

### 10.1 数据切分的碎片问题（已定位，待修复）

原始语料用 latin-1 解码时含 `0x85`（NEXT LINE, NEL）控制字符，Python 的 `str.splitlines()` 会把它当成换行，导致：

- `rt-polarity.pos`：应有 5331 句 → 被切成 5344 条；
- `rt-polarity.neg`：应有 5331 句 → 被切成 5341 条；
- 总计 10662 句 → **10685 条样本**，多出 21 条碎片。

`data/processed/train.csv` 中可见 15 条长度 < 15 的记录（含空文本片段，如 `mr-neg-00473`、`mr-neg-01562`），即碎片样本。影响约 0.2%，不改变"non-static 最好"的趋势判断，但会轻微扰动指标。

**修复方案**：改用 `text.split("\n")` 并按 10662 条校验，重新生成划分后复跑 12 次实验（约 3 分钟）。

### 10.2 替代词向量（最主要限制）

SVD 50 维 vs 原文 Google News word2vec 300 维，是 CNN-static 表现不如预期的直接原因，也是当前结果不能与原文 81.0 / 81.5 对比的根本原因。

**改进方向**：缓存正式预训练词向量（Google News 或 GloVe），把 static / non-static 两组替换后重跑，作为"接近原文设置"的补充对照。

### 10.3 固定划分 vs 10 折交叉验证

固定 80/10/10 让结果可复现、可复查，但测试集只有 1069 条，单条样本对指标的影响约 0.09 pp，种子间差异容易被噪声放大。

**缓解**：所有结论都以"3 种子均值 + 范围 + 配对差值"给出，不依赖单次运行。

### 10.4 训练预算偏小

5 epoch、CPU、无 early stopping 调参，模型可能欠拟合（train_loss 仍在下降）。

**缓解**：统一扩到 10 epoch，并固定"按 dev Macro-F1 选模型"的规则，保证可比。

### 10.5 实现层面的简化

- L2 max-norm 只施加于全连接层权重（原文约束所有词向量权重）；
- `<unk>` 向量用其余词向量均值初始化（原文随机初始化未登录词）；
- 词表仅来自训练集（16587），未与预训练词表求交。

以上三点都已在代码注释和本文档中标注，不影响主对照的相对比较。

---

## 附录 A：文件清单

| 路径 | 内容 |
|---|---|
| `code/textcnn_experiment.py` | 数据准备、词表、SVD/随机词向量、TextCNN、训练与评估、结果落盘 |
| `configs/*.json` | 三种设置的配置快照 |
| `data/README.md` | 数据来源与划分说明 |
| `data/processed/manifest.json` | 原始文件 SHA-256、划分种子、样本数 |
| `data/processed/{train,dev,test}.csv` | 固定划分数据（含样本 ID） |
| `results/all_runs.csv` | 12 次运行明细 |
| `results/summary_by_mode.csv` / `summary.md` | 按条件聚合结果 |
| `results/*_seed*/` | 每次运行的 config / train_log / summary / 逐条预测 |
| `results/cases.csv` / `cases.md` | 代表性失败与修正案例 |

## 附录 B：复现所需的软件环境

- Python 3.12；`torch`、`scikit-learn`、`numpy`、`pandas`、`requests`（见 `requirements.txt`）；
- 普通 CPU 即可完成全部 12 次运行；GPU 可缩短多种子实验时间。

## 附录 C：代码来源与数据合规

- 原作者代码/数据：<https://github.com/yoonkim/CNN_sentence>；本组使用 PyTorch 重新实现模型与训练流程，未复制作者 Theano 代码。
- 未上传任何受限数据或大模型权重；仓库中只保留公开的 MR 语料与实验产物。
