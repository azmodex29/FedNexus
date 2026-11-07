# Project Progress

## Current phase

Phase 5 — GitHub-readiness audit and local repository preparation **complete**.
Publishing and presentation preparation remain out of scope pending user confirmation.

## Completed tasks

- Inspected the parent workspace; no pre-existing content was listed.
- Inspected Windows version, PowerShell, CPU, RAM, available disk capacity, Python installations, Git, and network connectivity.
- Created the dedicated project directory without modifying unrelated workspace content.
- Selected Python 3.11.7 and created a functioning isolated `.venv`.
- Consulted current official Flower and PyTorch installation documentation.
- Resolved current package releases from PyPI without installing application dependencies.
- Created the required project-management files.
- Verified current official Flower 1.43 documentation and official PyTorch CPU-installation guidance.
- Verified the existing installation: Flower 1.43.0 with Ray 2.55.1, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, Streamlit 1.67.0, pandas 3.1.1, NumPy 2.4.6, and matplotlib 3.11.2.
- Reconciled the generated official Flower PyTorch quickstart with the project requirements: current `ServerApp`/`ClientApp` Message API, MNIST, FedAvg, three clients, five server rounds, and one local epoch.
- Replaced the CIFAR-10 model/data pipeline with a CPU-only MNIST CNN. Client partitions are deterministic and non-overlapping: 20,000 examples per client, then 16,000 local-training and 4,000 local-validation examples.
- Installed the local project in editable mode and validated source compilation, package consistency, Flower/Ray API imports, Flower CLI, CPU-only status, and an MNIST CNN forward pass.
- Executed a complete native Flower simulation with three simulated SuperNodes, five FedAvg rounds, one local epoch per client per round, CPU-only PyTorch, client-local validation, and central global evaluation.
- Persisted real global and aggregated client metrics, a JSON experiment manifest, a completion log, and a CPU-loadable final global-model checkpoint.
- Added `scripts/verify_experiment.py`, which independently validates all three deterministic partitions, config invariants, all six CSV rows (initial plus five rounds), required metrics, summary consistency, and checkpoint loading.
- Ran the verifier successfully against the fresh completed experiment. Final central MNIST accuracy was `0.9811`; final loss was `0.07928350340710817`.
- Built `app.py` and the `dashboard/` data layer. The dashboard dynamically validates and reads the saved CSV, JSON, log, and checkpoint; it does not hardcode experiment metrics.
- Implemented overview, interactive training charts, federated architecture visualization, client partition comparison, model/checkpoint inspection, complete metrics table, and original-artifact downloads.
- Added `scripts/test_dashboard.py`. Streamlit's supported `AppTest` rendered the dashboard without exceptions, found six tabs and all four download controls, and verified graceful handling of a missing-results directory.
- Resolved a local Streamlit WebSocket compatibility warning without breaking Flower: retained Flower-compatible `uvicorn==0.49.0` and constrained `websockets` to `16.1.1` through `websockets>=16.1.1,<17` in `pyproject.toml`; `pip check` passes.
- Audited all tracked source, configuration, experiment artifacts, documentation,
  and Git state for GitHub readiness. The initial repository has no commits and
  no remotes; its configured local Git identity is available for a checkpoint
  commit.
- Retained the verified results and `85,543`-byte final model checkpoint because
  they are useful, portable experiment evidence and well within normal GitHub
  limits.
- Added Altair as a direct dependency because the dashboard imports it directly;
  expanded `.gitignore` for temporary files and environment files; and made the
  evaluation helper explicitly use model evaluation mode.
- Confirmed no hard-coded absolute paths or hard-coded experiment metrics in
  application/dashboard code, and no exposed credentials or tokens in tracked
  project content.
- Added prominent file-level notices to Flower quickstart-derived modules and
  clarified that Apache-2.0 upstream attribution is preserved while the
  project's own license/copyright holder still needs user confirmation.
- Recorded the five-member university team in `AUTHORS.md` and added a Team /
  Contributors section to the README without assigning individual roles or
  contributions. The intended copyright holder is `The Federated Learning
  Project Contributors`; an MIT license remains conditional on all five
  contributors agreeing and having authority to license their contributions.
- Preserved the existing Flower Apache-2.0 attribution and license text. No
  project MIT license, remote, visibility change, or publishing action was made.

## Failed tasks

- Initial attempt to create the project directory and use it as the command working directory in a single command failed before execution: the working directory did not yet exist (`os error 267`). The directory was then created in a separate verified command; virtual-environment creation subsequently succeeded.
- The fresh streamed run's console log stream ended early with `'charmap' codec can't encode characters in position 70-72'` after creating all three SuperNodes. The underlying Flower simulation continued and completed successfully at `2026-11-07T07:03:09.648007+05:30`, writing its verified results to `results/`. Use a UTF-8 terminal (or `chcp 65001`) for streamed output.

## Outstanding tasks

- Obtain agreement and licensing authority confirmation from all five
  contributors before adding a project MIT license or publishing. No remote has
  been added and no publication action has occurred.

## Exact commands already executed

```powershell
Get-ChildItem -Force | Select-Object Mode,Length,LastWriteTime,Name | Format-Table -AutoSize
$PSVersionTable | Format-List; Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture | Format-List; Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | Format-List; Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | Format-List; Get-PSDrive -PSProvider FileSystem | Select-Object Name,Root,@{Name='FreeGB';Expression={[math]::Round($_.Free/1GB,2)}},@{Name='UsedGB';Expression={[math]::Round($_.Used/1GB,2)}} | Format-Table -AutoSize
Get-Command python,py,git,powershell -ErrorAction SilentlyContinue | Select-Object Name,Source,Version | Format-Table -AutoSize; where.exe python; where.exe py; where.exe git; python --version; py -0p; git --version
Test-Connection -ComputerName pypi.org -Count 1 -Quiet
Invoke-WebRequest -Uri 'https://pypi.org/simple/flwr/' -Method Head -UseBasicParsing -TimeoutSec 15
New-Item -ItemType Directory -Path 'C:\Users\mishr\Documents\Projects\Major\federated-learning-flower' -ErrorAction SilentlyContinue
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip --version
.\.venv\Scripts\python.exe -m pip index versions flwr
.\.venv\Scripts\python.exe -m pip index versions torch
.\.venv\Scripts\python.exe -m pip index versions torchvision
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\flwr.exe --version
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m compileall -q pytorchexample
.\.venv\Scripts\python.exe -c "from pytorchexample.task import Net; import torch; assert Net()(torch.randn(2, 1, 28, 28)).shape == (2, 10)"
.\.venv\Scripts\python.exe -c "from pytorchexample.task import load_data; loaders=[load_data(i, 3, 32) for i in range(3)]; assert all(len(train.dataset) == 16000 and len(val.dataset) == 4000 for train, val in loaders)"
.\.venv\Scripts\flwr.exe run --help
```

## Current environment details

- OS: Windows 11 Home Single Language, 64-bit, version 10.0.26300 (build 26300).
- Shell: PowerShell Core 7.6.9.
- CPU: 12th Gen Intel Core i7-1255U, 10 cores / 12 logical processors.
- RAM: 16,874,135,552 bytes (approximately 15.72 GiB).
- Disk: C: had 25.21 GB free at inspection.
- Internet: ICMP connectivity to `pypi.org` succeeded; HTTPS request to PyPI returned HTTP 200.
- Git: 2.55.0.windows.3.
- Python installations: 3.10.9, 3.11.7, 3.12, and 3.13; the Windows launcher default is 3.13.
- Project environment: `.venv` uses Python 3.11.7; pip 23.2.1.
- Installed releases: `flwr` 1.43.0 (`ray` 2.55.1), `torch` 2.15.0+cpu, `torchvision` 0.29.2+cpu, `streamlit` 1.67.0, `pandas` 3.1.1, `numpy` 2.4.6, and `matplotlib` 3.11.2.
- PyTorch reports `torch.cuda.is_available() == False`; CPU tensor operations and a CNN forward pass returned a `(2, 10)` tensor.
- The local MNIST dataset was downloaded successfully during partition validation.

## Next action required

Review the local checkpoint commit and obtain all five contributors' agreement
and licensing-authority confirmation before adding the intended MIT license or
publishing. Do not push or create a remote without a new request.

