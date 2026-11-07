"""Verify deterministic MNIST partitions and persisted Flower experiment results.

Run from the project root with the project's Python interpreter.  This script is
deliberately separate from the Flower application so it cannot affect training.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tomllib
from pathlib import Path

import torch
from torch.utils.data import Subset

from pytorchexample.task import Net, RANDOM_SEED, load_data


ROOT = Path(__file__).resolve().parents[1]


def subset_indices(dataset) -> set[int]:
    """Flatten nested torch Subsets to indices in the original MNIST dataset."""
    indices = list(range(len(dataset)))
    current = dataset
    while isinstance(current, Subset):
        indices = [current.indices[index] for index in indices]
        current = current.dataset
    return set(indices)


def verify_partitions() -> None:
    """Assert the three deterministic 16k/4k non-overlapping partitions."""
    partitions: list[set[int]] = []
    for partition_id in range(3):
        trainloader, valloader = load_data(partition_id, 3, 32)
        assert len(trainloader.dataset) == 16_000
        assert len(valloader.dataset) == 4_000
        train_indices = subset_indices(trainloader.dataset)
        validation_indices = subset_indices(valloader.dataset)
        assert len(train_indices) == 16_000
        assert len(validation_indices) == 4_000
        assert not train_indices.intersection(validation_indices)
        partitions.append(train_indices | validation_indices)

    assert all(len(partition) == 20_000 for partition in partitions)
    assert not partitions[0].intersection(partitions[1])
    assert not partitions[0].intersection(partitions[2])
    assert not partitions[1].intersection(partitions[2])
    assert len(set.union(*partitions)) == 60_000

    # Recreating the loaders must preserve exactly the same client membership.
    repeat_train, repeat_val = load_data(0, 3, 32)
    assert partitions[0] == (
        subset_indices(repeat_train.dataset) | subset_indices(repeat_val.dataset)
    )


def verify_results(results_dir: Path) -> dict:
    """Validate five rounds of real metrics and the saved model state dictionary."""
    summary_path = results_dir / "experiment_summary.json"
    metrics_path = results_dir / "federated_metrics.csv"
    assert summary_path.is_file(), f"Missing summary: {summary_path}"
    assert metrics_path.is_file(), f"Missing metrics CSV: {metrics_path}"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["random_seed"] == RANDOM_SEED
    assert summary["dataset"] == "MNIST"
    assert summary["num_clients"] == 3
    assert summary["num_server_rounds"] == 5
    assert summary["local_epochs"] == 1
    assert summary["aggregation_strategy"] == "FedAvg"
    assert summary["device"] == "cpu"

    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    assert [int(row["round"]) for row in rows] == [0, 1, 2, 3, 4, 5]
    for row in rows[1:]:
        for field in ("global_loss", "global_accuracy", "client_train_loss", "client_validation_loss", "client_validation_accuracy"):
            assert row[field] not in (None, ""), f"Missing {field} in round {row['round']}"
            assert float(row[field]) >= 0.0
        assert float(row["global_accuracy"]) <= 1.0
        assert float(row["client_validation_accuracy"]) <= 1.0

    final_row = rows[-1]
    final_metrics = summary["final_global_metrics"]
    assert abs(float(final_row["global_loss"]) - float(final_metrics["loss"])) < 1e-12
    assert abs(float(final_row["global_accuracy"]) - float(final_metrics["accuracy"])) < 1e-12

    model_path = Path(summary["model_checkpoint"])
    if not model_path.is_absolute():
        model_path = results_dir.parent / model_path
    assert model_path.is_file(), f"Missing model checkpoint: {model_path}"
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model = Net()
    model.load_state_dict(state_dict)
    assert next(model.parameters()).device.type == "cpu"
    return summary


def verify_config() -> None:
    """Validate the fixed experiment configuration from pyproject.toml."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    run_config = config["tool"]["flwr"]["app"]["config"]
    assert run_config["num-server-rounds"] == 5
    assert run_config["local-epochs"] == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    verify_config()
    verify_partitions()
    summary = verify_results(args.results_dir.resolve())
    print(
        "Verification passed: 3 deterministic non-overlapping client partitions, "
        "5 Flower rounds, required metrics, and loadable CPU checkpoint. "
        f"Final accuracy={summary['final_global_metrics']['accuracy']}."
    )


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
