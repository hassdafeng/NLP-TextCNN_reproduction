import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "results"

URLS = {
    "pos": "https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.pos",
    "neg": "https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.neg",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)
PAD = "<pad>"
UNK = "<unk>"


@dataclass
class RunConfig:
    mode: str
    seed: int
    embedding_dim: int = 50
    max_len: int = 64
    min_freq: int = 1
    batch_size: int = 50
    epochs: int = 5
    lr: float = 1.0
    dropout: float = 0.5
    filter_sizes: Tuple[int, ...] = (3, 4, 5)
    num_filters: int = 100
    l2_weight: float = 0.0
    l2_max_norm: float = 3.0
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    split_seed: int = 2026
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_mr() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for label, url in URLS.items():
        out = RAW_DIR / f"rt-polarity.{label}"
        if out.exists():
            continue
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        out.write_bytes(response.content)


def read_raw_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1").splitlines()


def build_split(cfg: RunConfig) -> None:
    download_mr()
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SPLIT_DIR / "manifest.json"
    split_paths = [SPLIT_DIR / f"{name}.csv" for name in ("train", "dev", "test")]
    if manifest_path.exists() and all(p.exists() for p in split_paths):
        return

    rows = []
    for label_name, label_id in [("pos", 1), ("neg", 0)]:
        for i, text in enumerate(read_raw_lines(RAW_DIR / f"rt-polarity.{label_name}")):
            rows.append({"id": f"mr-{label_name}-{i:05d}", "text": text, "label": label_id})
    df = pd.DataFrame(rows)
    train_df, temp_df = train_test_split(
        df,
        test_size=cfg.val_ratio + cfg.test_ratio,
        random_state=cfg.split_seed,
        stratify=df["label"],
    )
    dev_share = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - dev_share,
        random_state=cfg.split_seed,
        stratify=temp_df["label"],
    )
    for name, part in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        part.sort_values("id").to_csv(SPLIT_DIR / f"{name}.csv", index=False, encoding="utf-8")

    manifest = {
        "source": URLS,
        "raw_sha256": {p.name: sha256_file(p) for p in RAW_DIR.glob("rt-polarity.*")},
        "split_seed": cfg.split_seed,
        "val_ratio": cfg.val_ratio,
        "test_ratio": cfg.test_ratio,
        "counts": {
            "train": int(len(train_df)),
            "dev": int(len(dev_df)),
            "test": int(len(test_df)),
        },
        "label_map": {"neg": 0, "pos": 1},
        "note": "Fixed stratified 80/10/10 split for the course reproduction package. The original paper reports 10-fold CV on MR.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLIT_DIR / f"{name}.csv", keep_default_na=False)


def build_vocab(texts: Iterable[str], min_freq: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for text in texts:
        for tok in tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
    vocab = {PAD: 0, UNK: 1}
    for tok, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokenize(text)[:max_len]]
    ids.extend([vocab[PAD]] * (max_len - len(ids)))
    return ids


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Dict[str, int], max_len: int):
        self.df = df.reset_index(drop=True)
        self.x = torch.tensor([encode(t, vocab, max_len) for t in self.df["text"]], dtype=torch.long)
        self.y = torch.tensor(self.df["label"].to_numpy(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class TextCNN(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, trainable_embedding: bool, cfg: RunConfig):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=not trainable_embedding,
            padding_idx=0,
        )
        dim = embedding_matrix.shape[1]
        self.convs = nn.ModuleList([nn.Conv1d(dim, cfg.num_filters, k) for k in cfg.filter_sizes])
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.num_filters * len(cfg.filter_sizes), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            feat = torch.relu(conv(emb))
            pooled.append(torch.max(feat, dim=2).values)
        z = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(z)


def random_embeddings(vocab_size: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(-0.25, 0.25, size=(vocab_size, dim)).astype("float32")
    matrix[0] = 0.0
    return matrix


def svd_embeddings(train_texts: Sequence[str], vocab: Dict[str, int], dim: int, seed: int) -> np.ndarray:
    ordered_vocab = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1]) if tok not in (PAD, UNK)]
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        token_pattern=None,
        vocabulary={tok: i for i, tok in enumerate(ordered_vocab)},
        lowercase=False,
    )
    x = vectorizer.fit_transform(train_texts).T
    n_components = min(dim, max(2, min(x.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    word_vectors = svd.fit_transform(x).astype("float32")
    if n_components < dim:
        word_vectors = np.pad(word_vectors, ((0, 0), (0, dim - n_components)))
    norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
    word_vectors = word_vectors / np.maximum(norms, 1e-8)
    matrix = random_embeddings(len(vocab), dim, seed)
    for tok, local_idx in vectorizer.vocabulary_.items():
        matrix[vocab[tok]] = word_vectors[local_idx]
    matrix[0] = 0.0
    matrix[1] = matrix[2:].mean(axis=0)
    return matrix.astype("float32")


def make_embeddings(cfg: RunConfig, train_df: pd.DataFrame, vocab: Dict[str, int]) -> Tuple[np.ndarray, bool, str]:
    if cfg.mode == "rand":
        return random_embeddings(len(vocab), cfg.embedding_dim, cfg.seed), True, "random_trainable"
    if cfg.mode == "static":
        return svd_embeddings(train_df["text"].tolist(), vocab, cfg.embedding_dim, cfg.seed), False, "local_svd_frozen"
    if cfg.mode == "non_static":
        return svd_embeddings(train_df["text"].tolist(), vocab, cfg.embedding_dim, cfg.seed), True, "local_svd_trainable"
    raise ValueError(f"Unknown mode: {cfg.mode}")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    gold, pred, prob_pos = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred.extend(np.argmax(probs, axis=1).tolist())
            prob_pos.extend(probs[:, 1].tolist())
            gold.extend(y.numpy().tolist())
    return (
        accuracy_score(gold, pred),
        f1_score(gold, pred, average="macro"),
        np.array(pred),
        np.array(prob_pos),
    )


def enforce_max_norm(model: nn.Module, max_norm: float) -> None:
    with torch.no_grad():
        w = model.fc.weight
        norms = w.norm(2, dim=1, keepdim=True)
        desired = torch.clamp(norms, max=max_norm)
        w.mul_(desired / (norms + 1e-8))


def run_one(cfg: RunConfig) -> Dict[str, object]:
    set_seed(cfg.seed)
    build_split(cfg)
    train_df, dev_df, test_df = load_split("train"), load_split("dev"), load_split("test")
    vocab = build_vocab(train_df["text"], cfg.min_freq)
    emb, trainable_embedding, emb_source = make_embeddings(cfg, train_df, vocab)

    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = TextCNN(emb, trainable_embedding, cfg).to(device)
    train_loader = DataLoader(
        TextDataset(train_df, vocab, cfg.max_len),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    dev_loader = DataLoader(TextDataset(dev_df, vocab, cfg.max_len), batch_size=cfg.batch_size)
    test_dataset = TextDataset(test_df, vocab, cfg.max_len)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2_weight)
    loss_fn = nn.CrossEntropyLoss()
    best_state = None
    best_dev_f1 = -math.inf
    rows = []
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            enforce_max_norm(model, cfg.l2_max_norm)
            total_loss += loss.item() * len(y)
        dev_acc, dev_f1, _, _ = evaluate(model, dev_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_df),
            "dev_accuracy": dev_acc,
            "dev_macro_f1": dev_f1,
        }
        rows.append(row)
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_f1, test_pred, test_prob = evaluate(model, test_loader, device)
    elapsed = time.time() - start

    default_windows = (3, 4, 5)
    window_suffix = "" if cfg.filter_sizes == default_windows else "_w" + "".join(map(str, cfg.filter_sizes))
    run_id = f"{cfg.mode}{window_suffix}_seed{cfg.seed}"
    run_dir = RESULT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "train_log.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    out_df = test_df.copy()
    out_df["pred_label"] = test_pred
    out_df["prob_pos"] = test_prob
    out_df["config"] = cfg.mode
    out_df["seed"] = cfg.seed
    out_df.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8")

    summary = {
        "run_id": run_id,
        "mode": cfg.mode,
        "filter_sizes": ",".join(map(str, cfg.filter_sizes)),
        "condition": cfg.mode if cfg.filter_sizes == (3, 4, 5) else f"{cfg.mode}_w{''.join(map(str, cfg.filter_sizes))}",
        "seed": cfg.seed,
        "embedding_source": emb_source,
        "embedding_trainable": trainable_embedding,
        "vocab_size": len(vocab),
        "train_size": len(train_df),
        "dev_size": len(dev_df),
        "test_size": len(test_df),
        "best_dev_macro_f1": best_dev_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "elapsed_sec": elapsed,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def summarize() -> None:
    summaries = []
    for path in sorted(RESULT_DIR.glob("*_seed*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        config_path = path.parent / "config.json"
        if "filter_sizes" not in item and config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            filters = cfg.get("filter_sizes", [3, 4, 5])
            item["filter_sizes"] = ",".join(map(str, filters))
        if "condition" not in item:
            filters = item.get("filter_sizes", "3,4,5").replace(",", "")
            item["condition"] = item["mode"] if filters == "345" else f"{item['mode']}_w{filters}"
        summaries.append(item)
    if not summaries:
        return
    df = pd.DataFrame(summaries)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_DIR / "all_runs.csv", index=False, encoding="utf-8")
    agg = (
        df.groupby("condition")
        .agg(
            mode=("mode", "first"),
            filter_sizes=("filter_sizes", "first"),
            seeds=("seed", lambda x: ",".join(map(str, sorted(x)))),
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_min=("test_accuracy", "min"),
            accuracy_max=("test_accuracy", "max"),
            macro_f1_mean=("test_macro_f1", "mean"),
            macro_f1_min=("test_macro_f1", "min"),
            macro_f1_max=("test_macro_f1", "max"),
            elapsed_sec_mean=("elapsed_sec", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(RESULT_DIR / "summary_by_mode.csv", index=False, encoding="utf-8")

    md = ["# TextCNN MR Results", ""]
    md.append("| condition | windows | seeds | accuracy mean | accuracy range | macro-F1 mean | macro-F1 range |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in agg.iterrows():
        md.append(
            f"| {r['condition']} | {r['filter_sizes']} | {r['seeds']} | {r['accuracy_mean']:.4f} | "
            f"{r['accuracy_min']:.4f}-{r['accuracy_max']:.4f} | {r['macro_f1_mean']:.4f} | "
            f"{r['macro_f1_min']:.4f}-{r['macro_f1_max']:.4f} |"
        )
    (RESULT_DIR / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rand", "static", "non_static"], default="rand")
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2026])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--filter-sizes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--prepare-data-only", action="store_true")
    args = parser.parse_args()

    base_cfg = RunConfig(
        mode=args.mode,
        seed=args.seeds[0],
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        max_len=args.max_len,
        filter_sizes=tuple(args.filter_sizes),
        device=args.device,
    )
    build_split(base_cfg)
    if args.prepare_data_only:
        print(f"Data prepared under {SPLIT_DIR}")
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        cfg = RunConfig(
            mode=args.mode,
            seed=seed,
            epochs=args.epochs,
            embedding_dim=args.embedding_dim,
            max_len=args.max_len,
            filter_sizes=tuple(args.filter_sizes),
            device=args.device,
        )
        summary = run_one(cfg)
        print(json.dumps(summary, ensure_ascii=False))
        sys.stdout.flush()
    summarize()


if __name__ == "__main__":
    main()
