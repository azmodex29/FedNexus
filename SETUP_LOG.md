# Setup Log

All timestamps below use the user-provided client date (2026-11-06, Asia/Kolkata). Command output is summarized faithfully; detailed environment values are retained in `PROJECT_PROGRESS.md`.

## 2026-11-06 — Environment inspection

| Check | Actual result |
| --- | --- |
| Parent workspace contents | No entries were listed by `Get-ChildItem -Force`. |
| Operating system | Windows 11 Home Single Language, 64-bit, build 26300. |
| PowerShell | PowerShell Core 7.6.9. |
| CPU/RAM | Intel Core i7-1255U, 10 cores / 12 logical processors; approximately 15.72 GiB RAM. |
| Disk | C: drive had 25.21 GB free. |
| Python | Python 3.10.9, 3.11.7, 3.12, and 3.13 were discovered. |
| Git | `git version 2.55.0.windows.3`. |
| Network | Ping to `pypi.org` returned `True`; HTTPS HEAD request returned HTTP 200. |
| Python 3.11 virtual environment | Created successfully at `.venv`; interpreter reports Python 3.11.7 and pip 23.2.1. |
| Package-index lookup | PyPI resolved flwr 1.43.0, torch 2.15.0, and torchvision 0.29.2. |

## 2026-11-06 — Dependencies and application scaffolding

| Check | Actual result |
| --- | --- |
| Official compatibility review | Flower documentation identifies 1.43.0 as the stable release, requires Python 3.11+, and specifies `flwr[simulation]` for the Simulation Runtime. PyTorch documentation supports Python 3.11+ and its official CPU wheel index was used for the installed CPU builds. |
| Installed packages | Flower 1.43.0; Ray 2.55.1; torch 2.15.0+cpu; torchvision 0.29.2+cpu; Streamlit 1.67.0; pandas 3.1.1; NumPy 2.4.6; matplotlib 3.11.2. Scikit-learn was not installed because this app does not use it. |
| CPU verification | `torch.cuda.is_available()` returned `False`; a CPU tensor operation and a small CNN forward pass both succeeded. |
| Flower verification | `flwr --version` returned `Flower version: 1.43.0`; imports of `ClientApp`, `ServerApp`, `FedAvg`, the Message API records, and Ray succeeded. |
| Application base | The official Flower PyTorch quickstart already present in the project was retained as the base and adapted in place. |
| Dataset/model | MNIST with a small single-channel CNN; MNIST downloaded successfully. |
| Client partitions | Three deterministic non-overlapping partitions of 20,000 training examples, each split into 16,000 local-train and 4,000 local-validation examples. |
| Simulation configuration | `num-server-rounds = 5`, `local-epochs = 1`, FedAvg, and a documented supported `flwr run --federation-config` invocation with three supernodes, one CPU per worker, and zero GPU allocation. |
| Validation | Editable installation, `pip check`, source compilation, CLI help, package imports, model forward pass, and data-loader shape/size checks all passed. |

The full federated experiment was deliberately not started in this phase, so no accuracy or loss has been claimed.

## 2026-11-07 — Phase 3 execution and verification

| Check | Actual result |
| --- | --- |
| Completed Flower run | A native Flower simulation completed with 3 simulated clients, five FedAvg rounds, one local epoch, and CPU-only resources. |
| Final central evaluation | Round 5 loss: `0.07928350340710817`; accuracy: `0.9811` on the 10,000-example MNIST test set. |
| Per-round outputs | Real global and aggregated client metrics were persisted in `results/federated_metrics.csv` for initial evaluation and rounds 1–5. |
| Checkpoint | `results/models/final_global_model.pt` loaded successfully into the MNIST CNN on CPU. |
| Independent validation | `scripts/verify_experiment.py` passed partition disjointness, 20,000/16,000/4,000 sizes, determinism, configuration, required result fields, five-round completion, summary consistency, and checkpoint loading. |
| Console-stream limitation | The fresh run created 3 SuperNodes, then its Windows console log stream ended with a `charmap` Unicode-encoding error. The underlying simulation continued and completed at `2026-11-07T07:03:09.648007+05:30`; the results above are from that run. Use a UTF-8 terminal / `chcp 65001` when streaming logs. |

## 2026-11-07 — Phase 4 dashboard

| Check | Actual result |
| --- | --- |
| Dashboard | Created local Streamlit entry point `app.py` with validated, dynamic result loading from the project's `results/` directory. |
| Content | Implemented overview, global/client-local metric charts, virtual-client/FedAvg architecture diagram, partition table, CNN/checkpoint details, original metric table, and four downloads. |
| Safety | The checkpoint is loaded only with `torch.load(..., weights_only=True)` on CPU and only from a path constrained to the project directory. |
| Automated dashboard test | `scripts/test_dashboard.py` passed: valid artifacts render without Streamlit exceptions, six tabs and four download controls are present, and an empty temporary results root reports validation errors instead of placeholders. |
| App startup | Streamlit started successfully at `http://127.0.0.1:8501`. Browser automation reached the local app shell; its environment did not expose the WebSocket-rendered page content, so visual completion was not claimed. |
| Compatibility | Streamlit warned that `websockets 17.4` may conflict with Flower-compatible `uvicorn 0.49.0`. The compatible alternative (`websockets 16.1.1`) was installed and recorded in `pyproject.toml`; `pip check` then passed. |

## Activation command for a future PowerShell session

```powershell
Set-Location 'C:\Users\mishr\Documents\Projects\Major\federated-learning-flower'
.\.venv\Scripts\Activate.ps1
```

If PowerShell's local execution policy prevents activation, use the virtual-environment interpreter directly instead of changing a system policy:

```powershell
.\.venv\Scripts\python.exe -m pip --version
```

## 2026-11-07 — GitHub-readiness audit

| Check | Actual result |
| --- | --- |
| Code audit | No hard-coded absolute paths or hard-coded result metrics were found in the application or dashboard. Deterministic non-overlapping partitioning and Flower FedAvg configuration were preserved. The evaluation helper now explicitly sets evaluation mode and uses device-aligned loss evaluation. |
| Dependencies | Added direct `altair>=5,<7` declaration because `app.py` imports Altair directly. Existing pinned runtime dependencies installed successfully. |
| Repository hygiene | `.gitignore` excludes virtual environments, downloaded MNIST data, caches, Ray/Flower runtime files, local logs, environment files, temporary files, and OS/editor metadata. Verified result artifacts remain included. |
| Artifact retention | `results/models/final_global_model.pt` is 85,543 bytes and remains included with the real metrics CSV, summary JSON, and completion log. |
| Attribution | Confirmed Flower's upstream Apache License 2.0 and added modified-source notices for quickstart-derived modules. The repository's intended project copyright holder and license have not been chosen. |
| Final verification | `scripts/verify_experiment.py`, `scripts/test_dashboard.py`, `pip check`, and source compilation completed successfully. A search of non-ignored project content found no credential or token material. |
| Git | Existing repository had no commits and no remotes. A local initial checkpoint commit was prepared; no push or remote creation was performed. |

