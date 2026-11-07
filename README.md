# FedNexus — A Distributed Federated Learning Framework

FedNexus is a CPU-only university project demonstrating federated MNIST image classification with Flower, PyTorch, and FedAvg. Three virtual Flower clients train a shared CNN on disjoint local data partitions while a central server aggregates model updates.

> Repository description: A network of independent clients learning collaboratively through a central aggregation system.

Repository: [github.com/azmodex29/FedNexus](https://github.com/azmodex29/FedNexus)

## Problem statement and motivation

Centralized machine learning normally requires collecting data in one place. FedNexus demonstrates an alternative workflow: participants retain their local examples, train locally, and exchange model updates for central aggregation. It is an educational simulation, not a production privacy system.

## Objectives

- Demonstrate a reproducible Flower `ClientApp`/`ServerApp` FedAvg workflow.
- Use deterministic, non-overlapping MNIST client partitions.
- Persist and verify genuine experiment metrics and a CPU-loadable checkpoint.
- Provide a local Streamlit dashboard for inspecting saved results.

## Features

- Three deterministic virtual MNIST clients, each with 20,000 examples.
- Five CPU-only FedAvg rounds with one local epoch per client.
- Central global-model evaluation on the held-out 10,000-example MNIST test set.
- Persisted metrics CSV, JSON summary, completion log, and model checkpoint.
- Dashboard charts, architecture view, checkpoint inspection, and downloads.
- Independent experiment verification and dashboard smoke-test scripts.

## Technical architecture

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

The three clients are simulated processes on one physical computer. This is not a deployment across three physical devices or machines.

## Federated learning workflow

1. MNIST training data is deterministically split into three disjoint client partitions, then each partition is split 80/20 for local training and validation.
2. The server initializes and sends global CNN parameters to participating clients.
3. Each client trains for one epoch only on its local training subset.
4. Clients return model updates and weighted metrics; raw examples are not intentionally transmitted by this workflow.
5. Flower applies FedAvg, then the server evaluates the aggregated global model on the separate MNIST test set.

## Flower client-server architecture and FedAvg

`pytorchexample/client_app.py` implements Flower `ClientApp` training and local evaluation handlers. `pytorchexample/server_app.py` configures `ServerApp`, starts FedAvg, and performs central evaluation. The strategy computes a sample-weighted aggregate of returned client model parameters; it does not implement secure aggregation or differential privacy.

## Technology stack

- Python 3.11, Flower 1.43, and the Flower Simulation Runtime (Ray)
- PyTorch 2.15 CPU build and TorchVision
- Streamlit, Pandas, Altair, and NumPy
- MNIST

## Verified experiment results

The committed artifacts in `results/` describe the completed run and are validated by `scripts/verify_experiment.py`.

| Round | Global loss | Global accuracy |
| ---: | ---: | ---: |
| 1 | 1.894248 | 30.26% |
| 2 | 0.128651 | 96.40% |
| 3 | 0.090655 | 97.54% |
| 4 | 0.085512 | 97.84% |
| 5 | 0.079284 | **98.11%** |

Final global evaluation: loss `0.07928350340710817`, accuracy `0.9811` (98.11%). The saved output also includes round-0 global evaluation and weighted aggregate client-local metrics.

## Team and proposed technical responsibility areas

These are proposed responsibility areas for the university team; they do not assert individual authorship of particular files, commits, or features. The dashboard and documentation are shared project deliverables.

| Member | Technical responsibility area | Scope |
| --- | --- | --- |
| Baranya Mishra | Federated Learning Architecture & Optimization Engineer | Federated workflow design, FedAvg configuration, convergence analysis, and aggregation-parameter evaluation. |
| Suhani Sinha | Machine Learning & Model Evaluation Engineer | CNN architecture, MNIST model optimization, training/generalization evaluation, and accuracy/loss analysis. |
| Devanssh Arav Das | Python & Data Engineering Engineer | Dataset preprocessing, deterministic client distribution, PyTorch training/validation pipelines, serialization, and reproducibility utilities. |
| Srujan Sanket Mishra | Distributed Systems & Federated Client Engineer | Flower ClientApp/ServerApp components, message exchange, simulation resources, client participation, synchronization, and execution reliability. |
| Ashish Kalyan Pattanayak | Experiment Engineering & System Validation Engineer | Workflow tests, partition/isolation validation, metrics persistence, checkpoint validation, reproducibility, and resource-performance analysis. |

See [AUTHORS.md](AUTHORS.md) for the contributor list and the same proposed technical responsibility areas.

## Hardware requirements

- Python 3.11 and a CPU capable of running three local worker processes.
- No CUDA/GPU is used.
- Internet access is required for the first package and MNIST download.
- Disk capacity for the virtual environment and ignored downloaded MNIST data.

The verified run used an Intel i7-1255U with approximately 16 GiB RAM.

## Installation

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2
.\.venv\Scripts\python.exe -m pip install -e .
```

The CPU PyTorch command is separate so a new Windows setup does not select a CUDA build. On macOS or Linux, use the environment's `bin/python` path and the appropriate CPU wheel source from PyTorch.

## Run the experiment

MNIST downloads to the ignored `data/` directory on first use.

```powershell
.\.venv\Scripts\flwr.exe run . --federation-config="num-supernodes=3 client-resources-num-cpus=1 client-resources-num-gpus=0 init-args-num-cpus=3 init-args-num-gpus=0" --stream
```

This starts three simulated clients with one CPU and zero GPUs per client. If a Windows terminal cannot render Flower's status output, run `chcp 65001` first.

## Verify the experiment

```powershell
.\.venv\Scripts\python.exe scripts\verify_experiment.py
```

The verifier checks deterministic and disjoint partitions, configured rounds, metric consistency, and safe CPU checkpoint loading.

## Launch the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). To run its smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\test_dashboard.py
```

## Project directory structure

```text
FedNexus/
├── app.py                         # Streamlit dashboard entry point
├── dashboard/                     # Artifact validation and dashboard helpers
├── pytorchexample/                # Flower/PyTorch application package
├── scripts/                       # Experiment verifier and dashboard smoke test
├── results/                       # Verified metrics, summary, log, checkpoint
├── AUTHORS.md                     # Team and proposed responsibility areas
├── THIRD_PARTY_LICENSES/          # Preserved upstream license text
└── THIRD_PARTY_NOTICES.md         # Upstream attribution and license scope
```

## Results and checkpoint

`results/federated_metrics.csv`, `results/experiment_summary.json`, `results/run_logs/completed_run.log`, and `results/models/final_global_model.pt` are retained as small, useful evidence of the verified experiment. The final checkpoint is a CPU-loadable PyTorch state dictionary of about 84 KiB.

## Privacy, security, and known limitations

- Plain FedAvg does not provide formal privacy guarantees.
- Secure aggregation, differential privacy, authentication, transport hardening, Byzantine robustness, and production monitoring are not implemented.
- The clients are simulated on one computer; network and cross-device behavior are not evaluated.
- Results apply to MNIST and this fixed configuration only.
- Individual client metrics were not retained; persisted client metrics are weighted aggregates.

## Third-party attribution

The Flower application structure was adapted from Flower's official PyTorch quickstart and modified for this experiment. Flower-derived material remains subject to Apache-2.0; its preserved license text and notices are in [THIRD_PARTY_LICENSES/FLOWER-APACHE-2.0.txt](THIRD_PARTY_LICENSES/FLOWER-APACHE-2.0.txt) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Other dependencies and MNIST remain subject to their respective licenses.

## License

Original FedNexus project material is licensed under the [MIT License](LICENSE), copyright © 2026 The FedNexus Project Contributors. This license does not replace or alter licenses for Flower-derived material, MNIST, or dependencies.
