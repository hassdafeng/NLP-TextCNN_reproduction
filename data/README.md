# 数据说明

本实验使用 MR 句子情感分类语料（sentence polarity dataset v1.0），即 5,331 条正例和 5,331 条负例。数据文件来自 Kim 的 [CNN_sentence](https://github.com/yoonkim/CNN_sentence) 仓库。原始版本与引用信息见 [Cornell 数据页](https://www.cs.cornell.edu/people/pabo/movie-review-data/)。

- `rt-polarity.pos`：正例句子
- `rt-polarity.neg`：负例句子

缺少原始文件时，`python code/textcnn_experiment.py --prepare-data-only` 会从作者仓库下载到 `data/raw/`。脚本按物理行读取，并以种子 2026 生成固定 80/10/10 分层划分：

- `data/processed_lf/train.csv`
- `data/processed_lf/dev.csv`
- `data/processed_lf/test.csv`

样本 ID 的末尾是原始文件内从 0 开始的行序号。`manifest.json` 记录原始文件 SHA-256（换行统一为 LF）、三份 CSV 的 SHA-256、划分随机种子、样本数量和标签映射。原论文对 MR 报告的是 10 折交叉验证，本课程复现实验使用固定划分，分数不可直接与原文表 2 对齐。

中期划分保存在 `data/processed/`。当时 `.splitlines()` 把评论中的 23 个 NEL 字符误识别成换行，生成了 10,685 条记录。修正后按文件物理行切分，样本 ID 和测试集以相应版本的清单为准。
