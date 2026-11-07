# Project Plan — Privacy-Preserving Distributed Image Classification Using Federated Learning

## Scope

Build a CPU-only local academic demonstrator using Flower, PyTorch, three simulated Flower clients, one Flower server, FedAvg, MNIST, and a Streamlit metrics dashboard.

## Technology decisions

- **Python 3.11.7:** Flower's current installation documentation requires Python 3.11 or later. Flower's simulation guidance recommends Python 3.11 as the upper bound because its simulation dependency Ray has known compatibility considerations with newer Python versions.
- **Flower 1.43.0:** installed with the `simulation` extra (Ray 2.55.1) for Flower's supported Simulation Runtime and current `ServerApp`/`ClientApp` Message API.
- **PyTorch 2.15.0+cpu and torchvision 0.29.2+cpu:** installed from the official PyTorch CPU wheel index. CUDA is unavailable and the application explicitly selects `cpu`.
- **MNIST + small CNN:** selected and implemented. The 60,000 training examples are deterministically split into three non-overlapping 20,000-example client partitions; each is split 80/20 for local train/validation.

## Phases

1. **Environment setup** — inspect machine, create an isolated Python 3.11 virtual environment, document findings. *(Completed)*
2. **Dependency installation and application scaffolding** — install pinned Flower simulation, CPU PyTorch/torchvision, Streamlit, and visualization dependencies; generate and adapt the modern Flower quickstart. *(Completed)*
3. **Federated application execution** — run and validate the modern Flower application using a server, three simulated clients, FedAvg, client data partitions, and a PyTorch CNN. *(Completed)*
4. **Experiment tracking** — persist actual per-round loss and accuracy metrics plus final evaluation results. *(Completed)*
5. **Dashboard** — build a Streamlit dashboard to display real saved experiment data and visualizations. *(Completed)*
6. **Validation and documentation** — run a complete CPU experiment, validate outputs, and write setup/run/architecture documentation. *(Completed)*
7. **GitHub readiness** — audit the implementation, preserve verified artifacts, prepare repository hygiene and attribution, and create a local checkpoint commit without publishing. *(Completed 2026-11-07)*

## Phase 7 outcome

- Retained the verified CSV, JSON summary, completion log, and 85,543-byte CPU
  checkpoint; all are small enough to version in Git.
- Added direct Altair dependency declaration, broader temporary-file and
  environment-secret exclusions, and explicit evaluation mode in the model test
  helper.
- Confirmed that the local repository has no prior history or remote and is
  ready for its first local commit. Publishing remains intentionally deferred.
- Preserved Flower Apache-2.0 attribution and added file-level modified-source
  notices. The project copyright holder and license choice remain a user
  decision before public publication.

## Sources consulted

- Flower installation documentation: https://flower.ai/docs/framework/main/en/how-to-install-flower.html
- Flower virtual-environment guidance: https://flower.ai/docs/framework/1.39/en/contributor-how-to-set-up-a-virtual-env.html
- PyTorch local installation guide: https://pytorch.org/get-started/locally/

