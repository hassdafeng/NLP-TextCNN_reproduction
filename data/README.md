# 数据说明

本实验使用作者维护仓库中的 MR 情感分类数据：

- `rt-polarity.pos`：正例句子
- `rt-polarity.neg`：负例句子

数据由脚本 `code/textcnn_experiment.py` 下载到 `data/raw/`。脚本生成固定 80/10/10 分层划分：

- `data/processed/train.csv`
- `data/processed/dev.csv`
- `data/processed/test.csv`

`manifest.json` 记录原始文件 SHA-256、划分随机种子、样本数量和标签映射。原论文对 MR 报告的是 10-fold CV；本课程复现实验使用固定划分，分数不可直接与原文表 2 绝对对齐。
