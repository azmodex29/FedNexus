"""Safe, local-only loading and validation for dashboard artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from pytorchexample.task import Net, load_data


REQUIRED_SUMMARY_FIELDS = {
    "status",
    "completed_at",
    "dataset",
    "num_clients",
    "num_server_rounds",
    "local_epochs",
    "aggregation_strategy",
    "device",
    "final_global_metrics",
}
REQUIRED_METRIC_COLUMNS = {"round", "global_loss", "global_accuracy"}


@dataclass
class ExperimentData:
    """Validated dashboard inputs. Errors are displayed, never hidden."""

    root: Path
    summary: dict | None = None
    metrics: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)
    paths: dict[str, Path] = field(default_factory=dict)


def project_root() -> Path:
    """Allow tests to redirect the root while defaulting to this repository."""
    override = os.environ.get("FL_DASHBOARD_PROJECT_ROOT")
    return Path(override).resolve() if override else Path(__file__).resolve().parents[1]


def _safe_artifact(root: Path, candidate: str | Path, fallback: Path) -> Path:
    """Return a project-contained artifact path; reject external/traversal paths."""
    path = Path(candidate) if candidate else fallback
    if not path.is_absolute():
        path = root / path
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return fallback
    return path.resolve()


def load_experiment(root: Path | None = None) -> ExperimentData:
    """Load only genuine experiment artifacts, collecting actionable errors."""
    root = (root or project_root()).resolve()
    results = root / "results"
    data = ExperimentData(root=root)
    summary_path = results / "experiment_summary.json"
    metrics_path = results / "federated_metrics.csv"
    log_path = results / "run_logs" / "completed_run.log"

    for label, path in (("summary", summary_path), ("metrics", metrics_path), ("log", log_path)):
        data.paths[label] = path
        if not path.is_file():
            data.errors.append(f"Required {label} file is missing: `{path.relative_to(root)}`")

    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            missing = REQUIRED_SUMMARY_FIELDS - set(summary)
            if missing:
                data.errors.append("Summary is missing fields: " + ", ".join(sorted(missing)))
            elif not isinstance(summary["final_global_metrics"], dict):
                data.errors.append("Summary field `final_global_metrics` must be an object.")
            else:
                data.summary = summary
                data.paths["checkpoint"] = _safe_artifact(
                    root,
                    summary.get("model_checkpoint", ""),
                    results / "models" / "final_global_model.pt",
                )
        except (json.JSONDecodeError, OSError) as error:
            data.errors.append(f"Could not read experiment summary: {error}")

    if metrics_path.is_file():
        try:
            metrics = pd.read_csv(metrics_path)
            missing = REQUIRED_METRIC_COLUMNS - set(metrics.columns)
            if missing:
                data.errors.append("Metrics CSV is missing columns: " + ", ".join(sorted(missing)))
            elif metrics.empty:
                data.errors.append("Metrics CSV contains no rows.")
            else:
                metrics["round"] = pd.to_numeric(metrics["round"], errors="raise")
                data.metrics = metrics.sort_values("round").reset_index(drop=True)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            data.errors.append(f"Could not read metrics CSV: {error}")
    return data


def checkpoint_details(data: ExperimentData) -> dict:
    """Load this project's state dictionary with PyTorch's restricted loader."""
    path = data.paths.get("checkpoint")
    if path is None or not path.is_file():
        return {"ok": False, "message": "Checkpoint file is unavailable."}
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model = Net()
        model.load_state_dict(state_dict)
        return {
            "ok": True,
            "message": "Loaded safely as a CPU state dictionary.",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "path": path,
        }
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        return {"ok": False, "message": f"Checkpoint could not be loaded: {error}"}


def client_partition_rows(num_clients: int) -> pd.DataFrame:
    """Read actual local partition sizes without training or modifying data."""
    rows = []
    for client_id in range(num_clients):
        trainloader, valloader = load_data(client_id, num_clients, batch_size=32)
        rows.append(
            {
                "Client ID": f"Client {client_id + 1}",
                "Local samples": len(trainloader.dataset) + len(valloader.dataset),
                "Training samples": len(trainloader.dataset),
                "Validation samples": len(valloader.dataset),
                "Partition": "Unique, non-overlapping MNIST subset",
            }
        )
    return pd.DataFrame(rows)
