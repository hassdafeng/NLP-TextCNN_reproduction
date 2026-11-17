"""Independently verify the corrected MR runs from saved files, without training."""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (13, 42, 2026)
CONDITIONS = {
    "rand": ("rand", [3, 4, 5]),
    "static": ("static", [3, 4, 5]),
    "non_static": ("non_static", [3, 4, 5]),
    "non_static_w245": ("non_static", [2, 4, 5]),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_split(root):
    split_dir = root / "data" / "processed_lf"
    manifest_path = split_dir / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest["record_format"] == "mr_lf_v1", "Unexpected MR record format")
    require(manifest["label_map"] == {"neg": 0, "pos": 1}, "Unexpected label map")

    raw_records = {}
    for suffix, label in (("neg", 0), ("pos", 1)):
        path = root / "data" / "raw" / f"rt-polarity.{suffix}"
        raw = path.read_bytes()
        raw_hash = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
        require(raw_hash == manifest["raw_sha256"][path.name], f"Changed raw data: {path}")
        try:
            raw.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            encoding = "latin-1"
        # Split physical records as bytes so an embedded NEL remains inside its sentence.
        lines = raw.split(b"\n")
        if lines[-1] == b"":
            lines.pop()
        require(len(lines) == 5331, f"Expected 5,331 physical lines in {path}")
        for index, line in enumerate(lines):
            sentence = line.removesuffix(b"\r").decode(encoding)
            require(bool(sentence.strip()), f"Blank source record: {path}:{index + 1}")
            raw_records[f"mr-{suffix}-{index:05d}"] = (sentence, label)

    splits = {}
    seen = set()
    for split in ("train", "dev", "test"):
        path = split_dir / f"{split}.csv"
        require(digest(path) == manifest["csv_sha256"][path.name], f"Changed split: {path}")
        rows = read_csv(path)
        require(len(rows) == manifest["counts"][split], f"Incorrect split count: {path}")
        records = {}
        for row in rows:
            sample_id = row["id"]
            record = (row["text"], int(row["label"]))
            require(sample_id not in seen, f"Duplicate or overlapping ID: {sample_id}")
            require(raw_records.get(sample_id) == record, f"Physical source mismatch: {sample_id}")
            seen.add(sample_id)
            records[sample_id] = record
        splits[split] = records
    require(seen == raw_records.keys(), "Splits do not cover the full physical corpus")
    return manifest, digest(manifest_path), splits


def metrics(predictions):
    confusion = Counter((gold, prediction) for gold, prediction, _ in predictions.values())
    accuracy = (confusion[0, 0] + confusion[1, 1]) / len(predictions)
    errors = confusion[0, 1] + confusion[1, 0]
    f1 = []
    for label in (0, 1):
        hits = 2 * confusion[label, label]
        f1.append(hits / (hits + errors) if hits + errors else 0.0)
    return accuracy, mean(f1)


def verify_run(path, condition, seed, manifest, manifest_hash, splits):
    config = read_json(path / "config.json")
    summary = read_json(path / "summary.json")
    mode, windows = CONDITIONS[condition]
    require((config["mode"], config["seed"], config["filter_sizes"]) == (mode, seed, windows),
            f"Configuration does not match {path.name}")
    require((summary["run_id"], summary["mode"], summary["seed"]) == (path.name, mode, seed),
            f"Summary identity mismatch: {path.name}")
    for key in ("split_seed", "val_ratio", "test_ratio"):
        require(config[key] == manifest[key], f"Split configuration mismatch: {path.name} {key}")
    require(summary["split_manifest_sha256"] == manifest_hash, f"Wrong split used by {path.name}")
    for split, records in splits.items():
        require(summary[f"{split}_size"] == len(records), f"Wrong {split} size: {path.name}")

    test_set = splits["test"]
    predictions = {}
    for row in read_csv(path / "test_predictions.csv"):
        sample_id = row["id"]
        gold, pred = int(row["label"]), int(row["pred_label"])
        probability = float(row["prob_pos"])
        require(sample_id not in predictions and test_set.get(sample_id) == (row["text"], gold),
                f"Prediction record mismatch: {path.name} {sample_id}")
        require(row["config"] == mode and int(row["seed"]) == seed,
                f"Prediction provenance mismatch: {path.name} {sample_id}")
        require(pred in (0, 1) and 0.0 <= probability <= 1.0,
                f"Invalid prediction: {path.name} {sample_id}")
        predictions[sample_id] = (gold, pred, probability)
    require(predictions.keys() == test_set.keys(), f"Incomplete test coverage: {path.name}")

    accuracy, macro_f1 = metrics(predictions)
    for field, expected in (("test_accuracy", accuracy), ("test_macro_f1", macro_f1)):
        require(math.isclose(summary[field], expected, rel_tol=0, abs_tol=1e-12),
                f"Recomputed {field} mismatch: {path.name}")
    log = [json.loads(line) for line in (path / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    require([row["epoch"] for row in log] == list(range(1, config["epochs"] + 1)),
            f"Unexpected epoch sequence: {path.name}")
    require(all(0 <= row["dev_macro_f1"] <= 1 for row in log), f"Invalid dev score: {path.name}")
    best = max(log, key=lambda row: row["dev_macro_f1"])
    require(math.isclose(summary["best_dev_macro_f1"], best["dev_macro_f1"], rel_tol=0, abs_tol=1e-12),
            f"Best dev Macro-F1 mismatch: {path.name}")
    return config, predictions, {"accuracy": accuracy, "macro_f1": macro_f1, "best_dev_epoch": best["epoch"]}


def paired_cases(runs, test_set):
    pairs = []
    buckets = {(outcome, label): [] for outcome in ("fixed", "regressed") for label in (0, 1)}
    for seed in SEEDS:
        before_config, before, before_metrics = runs["non_static", seed]
        after_config, after, after_metrics = runs["non_static_w245", seed]
        changed = {key for key in before_config.keys() | after_config.keys()
                   if before_config.get(key) != after_config.get(key)}
        require(changed == {"filter_sizes"}, f"Window pair changes additional settings: seed {seed}")
        counts = Counter()
        for sample_id, (gold, pred_before, prob_before) in before.items():
            _, pred_after, prob_after = after[sample_id]
            if pred_before == pred_after:
                counts["both_correct" if pred_before == gold else "both_wrong"] += 1
                continue
            outcome = "fixed" if pred_after == gold else "regressed"
            counts[outcome] += 1
            buckets[outcome, gold].append({
                "id": sample_id, "seed": seed, "outcome": outcome, "gold": gold,
                "pred_w345": pred_before, "pred_w245": pred_after,
                "prob_pos_w345": prob_before, "prob_pos_w245": prob_after,
                "text": test_set[sample_id][0],
            })
        pairs.append({
            "seed": seed, "n": len(test_set),
            "fixed": counts["fixed"], "regressed": counts["regressed"],
            "both_correct": counts["both_correct"], "both_wrong": counts["both_wrong"],
            "accuracy_delta_pp": (counts["fixed"] - counts["regressed"]) * 100 / len(test_set),
            "macro_f1_delta_pp": 100 * (after_metrics["macro_f1"] - before_metrics["macro_f1"]),
        })

    # Round-robin over outcome and label, ordered by sample ID and then seed.
    groups = [iter(sorted(rows, key=lambda row: (row["id"], row["seed"]))) for rows in buckets.values()]
    cases, used = [], set()
    while len(cases) < 12 and groups:
        for group in groups.copy():
            row = next((item for item in group if item["id"] not in used), None)
            if row is None:
                groups.remove(group)
            else:
                cases.append(row)
                used.add(row["id"])
            if len(cases) == 12:
                break
    return pairs, cases, {f"{outcome}_{label}": len(rows) for (outcome, label), rows in buckets.items()}


def main():
    manifest, manifest_hash, splits = verify_split(ROOT)
    result_dir = ROOT / "results" / "mr_lf"
    runs = {}
    grouped = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            run_id = f"{condition}_seed{seed}"
            runs[condition, seed] = verify_run(result_dir / run_id, condition, seed, manifest, manifest_hash, splits)
        grouped[condition] = {}
        for metric in ("accuracy", "macro_f1"):
            values = [runs[condition, seed][2][metric] for seed in SEEDS]
            grouped[condition][metric] = {"mean": mean(values), "min": min(values), "max": max(values)}
    reference = runs["rand", SEEDS[0]][0]
    expected = {key: value for key, value in reference.items() if key not in ("mode", "seed", "filter_sizes")}
    for (condition, seed), (config, _, _) in runs.items():
        actual = {key: value for key, value in config.items() if key not in ("mode", "seed", "filter_sizes")}
        require(actual == expected, f"Shared settings differ: {condition}_seed{seed}")

    pairs, cases, buckets = paired_cases(runs, splits["test"])
    fields = ("id", "seed", "outcome", "gold", "pred_w345", "pred_w245", "prob_pos_w345", "prob_pos_w245", "text")
    with (result_dir / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(cases)
    result = {
        "split_manifest_sha256": manifest_hash,
        "counts": {split: len(rows) for split, rows in splits.items()},
        "label_counts": {split: Counter(label for _, label in rows.values()) for split, rows in splits.items()},
        "runs": {f"{condition}_seed{seed}": stats for (condition, seed), (_, _, stats) in runs.items()},
        "grouped_metrics": grouped,
        "window_pairs": pairs,
        "available_case_predictions_by_outcome_and_gold": buckets,
        "selected_distinct_cases": len(cases),
    }
    (result_dir / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(runs)} runs over {sum(result['counts'].values()):,} samples. Saved {len(cases)} paired cases.")


if __name__ == "__main__":
    main()
