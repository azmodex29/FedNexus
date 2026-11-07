# Third-Party Notices

## License scope

The root [LICENSE](LICENSE) is the MIT License for original FedNexus project
material, copyright © 2026 The FedNexus Project Contributors. It does not
replace, relicense, or claim ownership of third-party material.

## Flower quickstart adaptation

The Flower application structure in `pytorchexample/` began as an adaptation of
the official Flower PyTorch quickstart. `task.py`, `client_app.py`, and
`server_app.py` carry file-level modification notices and were changed for this
CPU-only MNIST experiment, deterministic partitioning, global evaluation, and
result persistence.

Flower is distributed under the Apache License 2.0. The Apache-2.0 license text
previously retained at the repository root is preserved verbatim at
[`THIRD_PARTY_LICENSES/FLOWER-APACHE-2.0.txt`](THIRD_PARTY_LICENSES/FLOWER-APACHE-2.0.txt).
The upstream project and license are available at:

- https://github.com/adap/flower
- https://github.com/adap/flower/blob/main/LICENSE

## Runtime dependencies

FedNexus depends on third-party packages including Flower, Ray, PyTorch,
TorchVision, Streamlit, Pandas, NumPy, Altair, and their transitive
dependencies. They are not copied into this repository; each remains subject to
its own license. Consult package metadata and upstream repositories before
redistributing bundled dependency artifacts.

## MNIST

MNIST is downloaded locally by TorchVision at runtime and is intentionally not
committed to this repository. Dataset information is available from the
[MNIST database](https://yann.lecun.com/exdb/mnist/).
