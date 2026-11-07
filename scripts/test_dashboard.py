"""Local smoke tests for dashboard data validation and Streamlit rendering."""

from __future__ import annotations

import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard.data import load_experiment


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    valid = load_experiment(ROOT)
    assert not valid.errors, valid.errors
    assert valid.summary is not None and valid.metrics is not None
    assert len(valid.metrics) == 6
    assert valid.paths["checkpoint"].is_file()
    with tempfile.TemporaryDirectory() as temporary:
        invalid = load_experiment(Path(temporary))
        assert invalid.errors and invalid.summary is None and invalid.metrics is None

    app = AppTest.from_file(str(ROOT / "app.py"))
    app.run(timeout=60)
    assert not app.exception, app.exception
    assert len(app.tabs) == 6
    assert len(app.download_button) == 4
    print("Dashboard tests passed: valid artifacts render, downloads are present, and missing artifacts report errors.")


if __name__ == "__main__":
    main()
