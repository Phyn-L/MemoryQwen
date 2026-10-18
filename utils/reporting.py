from __future__ import annotations

import io
import math

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

EVAL_SECTIONS = {
    "qa_task": "eval_qa",
    "recon_task": "eval_recon",
    "recon_autoregressive": "eval_recon",
}
METRIC_LABELS = {"em": "EM", "f1": "F1", "rouge_l": "ROUGE-L"}
THREE_LINE = box.Box(" ━━ \n" "    \n" " ━━ \n" "    \n" "    \n" "    \n" "    \n" " ━━ \n")


def table_panel(table):
    caption, title = table.caption, table.title
    table.caption = None
    table.title = None
    width = min(120, Console(width=120).measure(table).maximum + 4)
    # The table is rendered inside a titled panel. Restoring ``table.title`` here
    # makes Rich render the same title twice: once on the panel and once above the
    # table itself.
    table.caption = caption
    return Panel(table, title=title, width=width, expand=False, border_style="cyan")


def render_rich(renderable) -> str:
    console = Console(record=True, file=io.StringIO(), force_terminal=False, width=120)
    console.print(renderable)
    return console.export_text().rstrip()


def _render_table(title, headers, rows, *, caption=None) -> str:
    table = Table(title=title, caption=caption, box=THREE_LINE, header_style="bold cyan")
    for index, header in enumerate(headers):
        table.add_column(header, justify="left" if index == 0 else "right", style="cyan" if index == 0 else "white")
    for row in rows:
        table.add_row(*row)
    return render_rich(table_panel(table))


def format_evaluation_report(
    task_metrics, *, selection=None, seed=None, step=None, max_contexts=None,
    max_new_tokens=None,
) -> str:
    if not task_metrics:
        return ""
    details = []
    for label, value in (("Step", step), ("max contexts/dataset", max_contexts),
                         ("max new tokens", max_new_tokens), ("global sample seed", seed)):
        if value is not None:
            details.append(f"{label} {value:,}")
    caption = " · ".join(details) or None
    qa_counts = {} if selection is None else dict(selection.qa_counts)
    context_counts = {} if selection is None else dict(selection.context_counts)
    lines = []
    for task_name, metrics in task_metrics.items():
        reconstruction = "recon_ce" in metrics
        count_key = "contexts" if reconstruction else "count"
        dataset_names = sorted(
            {
                key.removeprefix("dataset/").rsplit("/", 1)[0]
                for key in metrics
                if key.startswith("dataset/") and key.endswith(f"/{count_key}")
            }
            | qa_counts.keys()
            | context_counts.keys()
        )
        rows = []
        if reconstruction:
            for dataset in [*dataset_names, "overall"]:
                prefix = "" if dataset == "overall" else f"dataset/{dataset}/"
                ce = float(metrics.get(f"{prefix}recon_ce", 0.0))
                ppl = (
                    metrics.get("recon_ppl", math.exp(min(ce, 20.0)))
                    if not prefix
                    else math.exp(min(ce, 20.0))
                )
                rows.append([
                    dataset,
                    f"{ce:.4f}",
                    f"{ppl:.2f}",
                    f"{metrics.get(f'{prefix}recon_token_accuracy', 0.0):.4f}",
                    f"{metrics.get(f'{prefix}contexts', 0):,.0f}",
                ])
            lines.append(_render_table(
                f"Evaluation · {task_name} · teacher-forced",
                ["dataset", "recon_ce", "recon_ppl", "token_acc", "contexts"],
                rows,
            ))
            continue
        for dataset in [*dataset_names, "overall"]:
            overall = dataset == "overall"
            prefix = "" if overall else f"dataset/{dataset}/"
            qa = (
                sum(metrics.get(f"dataset/{name}/count", 0) for name in dataset_names)
                if overall
                else metrics.get(f"{prefix}count", 0)
            )
            contexts = len(selection.contexts) if overall and selection is not None else context_counts.get(dataset)
            sampled_qa = selection.qa_rows if overall and selection is not None else qa_counts.get(dataset, qa)
            rows.append([
                dataset,
                f"{metrics.get(f'{prefix}em', 0.0):.4f}",
                f"{metrics.get(f'{prefix}f1', 0.0):.4f}",
                f"{metrics.get(f'{prefix}rouge_l', 0.0):.4f}",
                "-" if contexts is None else f"{contexts:,}",
                f"{sampled_qa:,.0f}",
            ])
        lines.append(_render_table(
            f"Evaluation · {task_name}",
            ["dataset", "EM", "F1", "ROUGE-L", "contexts", "QA"],
            rows,
            caption=caption,
        ))
    return "\n".join(lines)


def eval_log_payloads(task_metrics=None):
    if task_metrics is None:
        return []
    payloads = []
    for task_name, metrics in task_metrics.items():
        section = EVAL_SECTIONS[task_name]
        payload = {}
        for key, value in metrics.items():
            if key.startswith("dataset/"):
                dataset, metric = key.removeprefix("dataset/").rsplit("/", 1)
                if metric == "count":
                    continue
                dataset = dataset.replace("/", "_").replace("(", "_").replace(")", "")
                payload[f"{dataset}_{METRIC_LABELS.get(metric, metric)}"] = value
            elif key != "count" and isinstance(value, (int, float)) and not isinstance(value, bool):
                payload[key] = value
        if payload:
            payloads.append((section, payload))
    return payloads
