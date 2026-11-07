# Third-Party Notices

## Flower quickstart adaptation

The Flower application structure in `pytorchexample/` began as an adaptation of
the official Flower PyTorch quickstart. `task.py`, `client_app.py`, and
`server_app.py` carry file-level modification notices and were changed for this
project's CPU-only MNIST experiment, deterministic partitioning, global
evaluation, and result persistence. Flower is distributed under the Apache
License 2.0. The upstream project and license are available at:

- https://github.com/adap/flower
- https://github.com/adap/flower/blob/main/LICENSE

The repository retains the Apache License 2.0 text in `LICENSE` to meet the
upstream attribution requirement. The project's own license and copyright
holder have not yet been confirmed; publishing should wait for that decision.

## Runtime dependencies

This project depends on third-party packages including Flower, Ray, PyTorch,
TorchVision, Streamlit, Pandas, NumPy, Altair, and their transitive
dependencies. They are not copied into this repository; each remains subject
to its own license. Consult the package metadata and upstream repositories
before redistributing bundled dependency artifacts.

## MNIST

MNIST is downloaded locally by TorchVision at runtime and is intentionally not
committed to this repository. Dataset information is available from the
[MNIST database](https://yann.lecun.com/exdb/mnist/).
