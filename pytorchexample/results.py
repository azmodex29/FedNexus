"""Persist completed Flower experiment outputs using metrics returned by FedAvg."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import flwr
import torch

from pytorchexample.task import RANDOM_SEED


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _results_dir() -> Path:
    """Return the caller-selected results directory or a local fallback."""
    return Path(os.environ.get("FL_RESULTS_DIR", PROJECT_ROOT / "results")).resolve()


def _portable_path(path: Path) -> str:
    """Store project-relative artifact paths so manifests work after cloning."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _metric_value(metrics: dict, key: str) -> float | int | str | None:
    """Return a scalar metric only when Flower supplied it."""
    value = metrics.get(key)
    return value if isinstance(value, (float, int, str)) else None


def persist_completed_experiment(result, run_config: dict) -> dict:
    """Write real aggregated metrics and final global parameters after FedAvg ends."""
    root = _results_dir()
    models_dir = root / "models"
    logs_dir = root / "run_logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    all_rounds = sorted(
        set(result.train_metrics_clientapp)
        | set(result.evaluate_metrics_clientapp)
        | set(result.evaluate_metrics_serverapp)
    )
    for server_round in all_rounds:
        train_metrics = dict(result.train_metrics_clientapp.get(server_round, {}))
        client_eval_metrics = dict(result.evaluate_metrics_clientapp.get(server_round, {}))
        global_metrics = dict(result.evaluate_metrics_serverapp.get(server_round, {}))
        rows.append(
            {
                "round": server_round,
                "global_loss": _metric_value(global_metrics, "loss"),
                "global_accuracy": _metric_value(global_metrics, "accuracy"),
                "client_train_loss": _metric_value(train_metrics, "train_loss"),
                "client_train_examples": _metric_value(train_metrics, "num-examples"),
                "client_validation_loss": _metric_value(client_eval_metrics, "eval_loss"),
                "client_validation_accuracy": _metric_value(client_eval_metrics, "eval_acc"),
                "client_validation_examples": _metric_value(
                    client_eval_metrics, "num-examples"
                ),
            }
        )

    metrics_path = root / "federated_metrics.csv"
    fieldnames = list(rows[0]) if rows else ["round"]
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    model_path = models_dir / "final_global_model.pt"
    torch.save(result.arrays.to_torch_state_dict(), model_path)

    timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
    final_row = next((row for row in reversed(rows) if row["round"] > 0), None)
    summary = {
        "status": "completed",
        "completed_at": timestamp,
        "random_seed": RANDOM_SEED,
        "flower_version": flwr.__version__,
        "dataset": "MNIST",
        "num_clients": 3,
        "num_server_rounds": int(run_config["num-server-rounds"]),
        "local_epochs": int(run_config["local-epochs"]),
        "aggregation_strategy": "FedAvg",
        "device": "cpu",
        "metrics_file": _portable_path(metrics_path),
        "model_checkpoint": _portable_path(model_path),
        "final_global_metrics": {
            "loss": final_row["global_loss"] if final_row else None,
            "accuracy": final_row["global_accuracy"] if final_row else None,
        },
    }
    summary_path = root / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log_path = logs_dir / "completed_run.log"
    log_path.write_text(
        "Flower FedAvg run completed successfully.\n"
        f"Completed at: {timestamp}\n"
        f"Rounds: {run_config['num-server-rounds']}\n"
        f"Clients: 3\n"
        f"Final global loss: {summary['final_global_metrics']['loss']}\n"
        f"Final global accuracy: {summary['final_global_metrics']['accuracy']}\n",
        encoding="utf-8",
    )
    return summary
