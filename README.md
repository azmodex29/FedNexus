# Federated Learning Using Flower and PyTorch

A CPU-only, local demonstrator of federated MNIST classification using Flower's
modern `ServerApp` / `ClientApp` Message API, PyTorch, and FedAvg. Three
simulated clients train a small CNN on disjoint local data partitions while a
Flower server aggregates their model updates.

Federated learning is useful when training data should remain at its source:
clients perform local optimization and exchange model updates rather than raw
examples. This repository demonstrates that workflow for teaching and
experimentation; it is not a production privacy system.

## Features

- Three deterministic, non-overlapping virtual MNIST clients.
- Five real Flower FedAvg rounds with one local epoch per client and CPU-only
  PyTorch execution.
- Global server evaluation on the held-out MNIST test set.
- Persisted metrics, experiment summary, completion log, and model checkpoint.
- A local Streamlit dashboard with charts, architecture visualization,
  checkpoint inspection, and original-artifact downloads.
- Separate scripts for experiment verification and dashboard smoke testing.

## Architecture

```text
MNIST train set (60,000 examples)
            │ deterministic shuffle / split
 ┌──────────┼──────────┐
 ▼          ▼          ▼
Client 1   Client 2   Client 3
20,000     20,000     20,000 examples each
16k train / 4k validation per client
 └──── local CNN training (1 epoch) ────┘
                    │ model updates
                    ▼
          Flower Server + FedAvg
                    │ global model
                    └──── distributed to all clients
```

The three clients are simulated processes on one physical computer. The server
initializes and distributes a global CNN, each client trains only on its own
16,000-example training split, and FedAvg computes a weighted aggregate of the
returned model parameters. The server evaluates the aggregated model after each
round against the separate 10,000-example MNIST test set.

## Verified experiment result

The committed result artifacts describe a completed five-round run. They are
the source for the following values and are independently checked by
`scripts/verify_experiment.py`.

| Round | Global loss | Global accuracy |
| ---: | ---: | ---: |
| 1 | 1.894248 | 30.26% |
| 2 | 0.128651 | 96.40% |
| 3 | 0.090655 | 97.54% |
| 4 | 0.085512 | 97.84% |
| 5 | 0.079284 | **98.11%** |

The saved output also includes initial global-model evaluation (round 0) and
weighted aggregate client-local training/validation metrics.

## Team and contributors

This is a five-member university team project. The contributors are listed in
[AUTHORS.md](AUTHORS.md): Baranya Mishra, Suhani Sinha, Devansh Aarav Das,
Srujan Sanket Mishra, and Ashish Kalyan Pattanayak. No individual roles or
specific contributions are assigned in this repository.

The intended copyright holder for the team's original project material is
**The Federated Learning Project Contributors**. This designation does not
claim ownership of Flower, MNIST, or other third-party software and materials.

## Dashboard screenshot

<!-- Add an annotated screenshot here after capturing it locally, for example:
![Dashboard overview](docs/images/dashboard-overview.png)
-->

Screenshot placeholder: launch the local dashboard below and capture the
overview screen before publishing if a visual preview is desired.

## Technology stack

- Python 3.11
- Flower 1.43 with the Simulation Runtime (Ray)
- PyTorch 2.15 CPU build and TorchVision
- Streamlit, Pandas, and Altair
- MNIST

## Requirements

- Python 3.11 (the project was tested with Python 3.11.7 on Windows 11).
- A CPU capable of running three local worker processes; no CUDA/GPU is used.
- Internet access for the first package and MNIST download.
- Local disk space for the virtual environment and downloaded MNIST data.

The verified run used an Intel i7-1255U with approximately 16 GiB RAM.

## Installation

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2
.\.venv\Scripts\python.exe -m pip install -e .
```

The CPU PyTorch command is intentionally separate so a new Windows setup does
not select a CUDA build. `pyproject.toml` declares the remaining project
dependencies. On macOS or Linux, use the virtual environment's `bin/python`
path instead of `Scripts\python.exe` and consult PyTorch's installation guide
for the appropriate wheel source.

## Run the federated simulation

MNIST downloads to the ignored `data/` directory on its first use.

```powershell
.\.venv\Scripts\flwr.exe run . --federation-config="num-supernodes=3 client-resources-num-cpus=1 client-resources-num-gpus=0 init-args-num-cpus=3 init-args-num-gpus=0" --stream
```

This starts three simulated clients, assigns one CPU and zero GPUs per client,
uses five rounds, and runs one local epoch per round. On Windows terminals that
cannot render Flower's Unicode status output, run `chcp 65001` first.

## Verify a completed experiment

```powershell
.\.venv\Scripts\python.exe scripts\verify_experiment.py
```

The verifier checks deterministic/disjoint partitions, configured rounds,
metric fields and consistency, and safe CPU checkpoint loading.

## Launch the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). The dashboard reads the
current `results/` files at page load. After a new experiment completes,
refresh the page to inspect its new persisted outputs.

To run the dashboard smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\test_dashboard.py
```

## Output structure

```text
results/
├── federated_metrics.csv          # global and aggregated client metrics
├── experiment_summary.json        # run metadata and final global metrics
├── models/final_global_model.pt   # PyTorch state dictionary (about 84 KiB)
└── run_logs/completed_run.log     # concise completion record
```

The small final checkpoint and the verified result artifacts are retained in
Git. Downloaded MNIST data, virtual environments, cache directories, temporary
run logs, and unsuccessful transient-run directories are excluded.

## Privacy and limitations

- Raw local examples are not intentionally transmitted by this Flower
  workflow, but plain FedAvg does **not** provide formal privacy guarantees.
- This project does not implement secure aggregation, differential privacy,
  authentication, transport hardening, Byzantine robustness, or production
  monitoring.
- The three clients are simulated on one machine, so this is not a network or
  cross-device deployment.
- The persisted client metrics are weighted aggregates; separate per-client
  measurements are not retained for this run.
- Results are for MNIST and this configuration only; they should not be
  generalized to other datasets or deployments without new experiments.

## Attribution and license

The Flower application structure was adapted from the official Flower PyTorch
quickstart and then modified for this MNIST experiment, deterministic
partitioning, evaluation, persistence, and dashboard. File-level notices mark
the adapted implementation files; the dashboard, verification scripts, and
project documentation are this project's own additions. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and dependency
notes. The repository includes the Apache License 2.0 text required for the
upstream adaptation. The team's intended license for its original material is
MIT, subject to confirmation that all five contributors agree and have the
authority to license their contributions. Until then, no project MIT license
has been added. The intended initial GitHub visibility is private.
