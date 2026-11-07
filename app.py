"""Interactive local dashboard for the completed Flower MNIST experiment."""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.data import checkpoint_details, client_partition_rows, load_experiment


st.set_page_config(page_title="FedNexus | Flower", page_icon="✦", layout="wide")


def _fmt_percent(value: float | None) -> str:
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def _fmt_number(value: float | None) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _metric_line(frame: pd.DataFrame, fields: list[tuple[str, str]], title: str, y_title: str):
    tidy = frame[["round", *[field for field, _ in fields]]].melt(
        "round", var_name="Metric", value_name="Value"
    ).dropna()
    labels = {field: label for field, label in fields}
    tidy["Metric"] = tidy["Metric"].map(labels)
    return (
        alt.Chart(tidy, title=title)
        .mark_line(point=alt.OverlayMarkDef(size=55), strokeWidth=3)
        .encode(
            x=alt.X("round:O", title="Federated round"),
            y=alt.Y("Value:Q", title=y_title),
            color=alt.Color("Metric:N", title=None),
            tooltip=[alt.Tooltip("round:O", title="Round"), alt.Tooltip("Metric:N"), alt.Tooltip("Value:Q", format=".4f")],
        )
        .properties(height=290)
        .interactive()
    )


def _architecture() -> None:
    st.html(
        """
        <style>
          .flow {font-family:Arial,sans-serif; display:flex; align-items:center; gap:12px; padding:15px 6px; min-width:780px}
          .node {border-radius:14px; padding:16px 14px; width:145px; min-height:86px; box-sizing:border-box; text-align:center; border:1px solid #cbd5e1; background:#f8fafc; color:#172554}
          .client {background:#eff6ff; border-color:#93c5fd}.server {background:#ecfdf5; border-color:#6ee7b7; width:178px}.arrow {color:#2563eb; font-size:27px; font-weight:bold}
          .caption {font-size:12px;color:#475569;margin-top:7px}.title {font-weight:700;font-size:15px}
        </style>
        <div class="flow">
          <div class="node client"><div class="title">Client 1</div><div class="caption">Local MNIST<br/>CNN training</div></div><div class="arrow">⇄</div>
          <div class="node client"><div class="title">Client 2</div><div class="caption">Local MNIST<br/>CNN training</div></div><div class="arrow">⇄</div>
          <div class="node server"><div class="title">Central Flower Server</div><div class="caption">FedAvg aggregation<br/>Global CNN evaluation</div></div><div class="arrow">⇄</div>
          <div class="node client"><div class="title">Client 3</div><div class="caption">Local MNIST<br/>CNN training</div></div>
        </div>
        """,
    )


def _downloads(data) -> None:
    mapping = [
        ("Metrics CSV", data.paths.get("metrics"), "text/csv"),
        ("Experiment summary JSON", data.paths.get("summary"), "application/json"),
        ("Final model checkpoint", data.paths.get("checkpoint"), "application/octet-stream"),
        ("Completed run log", data.paths.get("log"), "text/plain"),
    ]
    for label, path, mime in mapping:
        if path and path.is_file():
            st.download_button(label, data=path.read_bytes(), file_name=path.name, mime=mime, width="stretch")
        else:
            st.caption(f"{label}: unavailable")


def main() -> None:
    data = load_experiment()
    st.markdown("""<style>
        .hero {padding:1.25rem 1.5rem 1.1rem;border:1px solid #dbeafe;border-radius:18px;background:linear-gradient(110deg,#eff6ff,#f8fafc)}
        .hero h1 {margin:0;color:#0f172a;font-size:2rem}.hero p {margin:.45rem 0 0;color:#475569;font-size:1.02rem}
    </style>""", unsafe_allow_html=True)
    st.markdown("<div class='hero'><h1>FedNexus</h1><p>A distributed federated learning framework for MNIST classification with three virtual clients, local CNN training, and FedAvg aggregation.</p></div>", unsafe_allow_html=True)
    st.write("")

    if data.errors:
        st.error("The dashboard could not validate all experiment artifacts.")
        for error in data.errors:
            st.warning(error)
        st.info("No placeholder metrics are shown. Restore the indicated files and refresh the page.")
        return

    summary, metrics = data.summary, data.metrics
    final = summary["final_global_metrics"]
    st.success(f"Experiment status: {summary['status'].title()} · completed {summary['completed_at']}")
    overview = st.columns(5)
    overview[0].metric("Final global accuracy", _fmt_percent(final.get("accuracy")))
    overview[1].metric("Final global loss", _fmt_number(final.get("loss")))
    overview[2].metric("Simulated clients", int(summary["num_clients"]))
    overview[3].metric("Federated rounds", int(summary["num_server_rounds"]))
    overview[4].metric("Aggregation", summary["aggregation_strategy"])

    tab_overview, tab_progress, tab_architecture, tab_clients, tab_model, tab_results = st.tabs(
        ["Overview", "Training progress", "Architecture", "Clients", "Model", "Results & downloads"]
    )
    with tab_overview:
        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("Experiment overview")
            st.table(pd.DataFrame([
                ("Dataset", summary["dataset"]), ("Aggregation strategy", summary["aggregation_strategy"]),
                ("Execution device", summary["device"].upper()), ("Local epochs per round", summary["local_epochs"]),
                ("Completion timestamp", summary["completed_at"]), ("Flower version", summary.get("flower_version", "Not recorded")),
            ], columns=["Property", "Value"]).astype(str))
        with right:
            st.subheader("What this demonstrates")
            st.write("Each virtual client trains only on its own deterministic MNIST partition. Flower aggregates model updates with FedAvg; the raw client examples are not sent to the server.")
            st.caption("This is a one-computer simulation, not three physically separate devices. FedAvg alone does not provide secure aggregation or differential privacy.")
    with tab_progress:
        st.subheader("Global model evaluation")
        global_metrics = metrics.dropna(subset=["global_loss", "global_accuracy"])
        col_a, col_b = st.columns(2)
        with col_a: st.altair_chart(_metric_line(global_metrics, [("global_accuracy", "Global accuracy")], "Global accuracy", "Accuracy"), width="stretch")
        with col_b: st.altair_chart(_metric_line(global_metrics, [("global_loss", "Global loss")], "Global loss", "Cross-entropy loss"), width="stretch")
        local_fields = [(field, label) for field, label in [("client_train_loss", "Client training loss"), ("client_validation_loss", "Client validation loss"), ("client_validation_accuracy", "Client validation accuracy")] if field in metrics and metrics[field].notna().any()]
        st.subheader("Aggregated client-local metrics")
        if local_fields:
            loss_fields = [item for item in local_fields if "loss" in item[0]]
            accuracy_fields = [item for item in local_fields if "accuracy" in item[0]]
            if loss_fields: st.altair_chart(_metric_line(metrics, loss_fields, "Client-local loss", "Loss"), width="stretch")
            if accuracy_fields: st.altair_chart(_metric_line(metrics, accuracy_fields, "Client-local validation accuracy", "Accuracy"), width="stretch")
            st.caption("These are weighted aggregates returned by the participating clients, not separate per-client measurements.")
        else:
            st.info("No client-local metrics were saved for this run.")
    with tab_architecture:
        st.subheader("Federated training architecture")
        _architecture()
        st.write("At each round, the server distributes the global CNN. Clients train locally on their private partitions, return model updates, and the server applies FedAvg before redistributing the aggregated model.")
        st.info("The three clients are isolated simulated processes sharing one physical computer; they are not separate physical machines.")
    with tab_clients:
        st.subheader("Client partition comparison")
        st.dataframe(client_partition_rows(int(summary["num_clients"])), width="stretch", hide_index=True)
        st.caption("All 60,000 MNIST training examples are partitioned exactly once: 20,000 per client, then 16,000 train / 4,000 validation. Per-client metric values were not persisted; the progress tab reports their actual weighted aggregates.")
    with tab_model:
        st.subheader("CNN model information")
        details = checkpoint_details(data)
        model_left, model_right = st.columns(2)
        with model_left:
            st.code("Input: 1 × 28 × 28\nConv2d(1, 8, kernel=5) → ReLU → MaxPool2d(2)\nConv2d(8, 16, kernel=5) → ReLU → MaxPool2d(2)\nLinear(256, 64) → ReLU → Linear(64, 10)", language="text")
        with model_right:
            st.metric("Output classes", 10)
            st.metric("Trainable parameters", f"{details['parameters']:,}" if details.get("ok") else "Unavailable")
            st.write(f"Dataset: {summary['dataset']}")
            st.write(f"Checkpoint: `{data.paths.get('checkpoint', 'Unavailable')}`")
            (st.success if details.get("ok") else st.error)(details["message"])
    with tab_results:
        st.subheader("Complete saved metrics")
        st.dataframe(metrics, width="stretch", hide_index=True)
        st.subheader("Download original artifacts")
        _downloads(data)
        with st.expander("Completed run log"):
            st.code(data.paths["log"].read_text(encoding="utf-8"), language="text")
        with st.expander("Experiment summary JSON"):
            st.code(json.dumps(summary, indent=2), language="json")


if __name__ == "__main__":
    main()
