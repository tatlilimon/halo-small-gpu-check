"""Print the result tables from the files modal_app.py wrote to results/<pass>/.

The first table lists the cards. Then the runs are grouped by recipe and added
arguments, one table per group, smallest card first, one row per pass.
Throughput, step time and memory are the figures Halo logs with
enable_efficiency_metrics; they leave out the first two steps as warm-up.

The last column compares each run with one reference: the H100 run of its group
in the first pass, a setup Halo documents as supported. It is the largest
difference between the two runs' training losses over the steps. The H100 rows
of later passes show how much the reference setup itself differs between runs.
Halo logs the loss with four significant digits.
"""

import ast
import json
import re
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
ERROR_LINE = re.compile(r"^[\w.]+(?:Error|Exception): .*$", re.MULTILINE)
COLUMNS = [
    "Card",
    "Pass",
    "Outcome",
    "Attention",
    "Tokens/s",
    "Seconds/step (range)",
    "Peak allocated, GiB",
    "Reserved, GiB",
]


def logged(log: str, first_key: str) -> list[dict]:
    """Halo prints each metrics record on a line of its own as a Python dict."""
    return [ast.literal_eval(line) for line in log.splitlines() if line.startswith(f"{{'{first_key}'")]


def read_run(path: Path) -> dict:
    run = json.loads(path.read_text())
    log = path.with_suffix(".log").read_text(encoding="utf-8")
    run["pass"] = path.parent.name
    run["card"], memory, run["driver"] = run["gpu"].split(", ")
    run["mib"] = int(memory.removesuffix(" MiB"))
    backend = re.search(r"Attention: (\S+)", log)
    run["backend"] = backend.group(1) if backend else "n/a"
    steps = logged(log, "loss")
    run["losses"] = [float(step["loss"]) for step in steps]
    run["step_times"] = [float(step["step_time_seconds"]) for step in steps if "step_time_seconds" in step]
    if run["exit_code"] == 0:
        run["outcome"] = "completed"
        run["summary"] = logged(log, "train_runtime")[-1]
    else:
        # The first sentence of the exception that ended the run.
        run["outcome"] = "failed: " + ERROR_LINE.findall(log)[-1].split(". ")[0].rstrip(".")
    return run


def markdown(header: list[str], alignment: str, rows: list[list[str]]) -> str:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    return "\n".join([line(header), alignment, *map(line, rows)])


def cards_table(runs: list[dict]) -> str:
    types: dict[str, list[dict]] = {}
    for run in runs:
        types.setdefault(run["requested_gpu"], []).append(run)
    rows = [
        [
            gpu,
            ", ".join(sorted({r["card"] for r in members})),
            ", ".join(sorted({f"{r['mib']:,} MiB" for r in members})),
            ", ".join(sorted({r["image"] for r in members})),
        ]
        for gpu, members in sorted(types.items(), key=lambda item: item[1][0]["mib"])
    ]
    return markdown(["Modal GPU type", "Card reported by nvidia-smi", "Memory", "Image"], "|---|---|---:|---|", rows)


def loss_difference(run: dict, reference: dict | None) -> str:
    if run is reference:
        return "reference"
    if reference is None or len(run["losses"]) != len(reference["losses"]):
        return "n/a"
    return f"{max(abs(a - b) for a, b in zip(run['losses'], reference['losses'])):.3f}"


def group_table(runs: list[dict]) -> str:
    runs = sorted(runs, key=lambda r: (r["mib"], r["pass"]))
    reference = next((r for r in runs if r["requested_gpu"] == "H100" and r["exit_code"] == 0), None)
    last_column = f"Loss difference to H100, pass {reference['pass']}" if reference else "Loss difference to H100"
    rows = []
    for run in runs:
        cells = [run["card"], run["pass"], run["outcome"], f"`{run['backend']}`"]
        if "summary" in run:
            summary = run["summary"]
            times = run["step_times"]
            cells += [
                f"{float(summary['avg_tokens_per_second']):,.0f}",
                f"{float(summary['avg_step_time_seconds']):.2f} ({min(times):.2f} to {max(times):.2f})",
                f"{float(summary['training_peak_allocated_gb']):.2f}",
                f"{float(summary['reserved_gb']):.2f}",
            ]
        else:
            cells += ["n/a"] * 4
        cells.append(loss_difference(run, reference))
        rows.append(cells)
    return markdown([*COLUMNS, last_column], "|---|---|---|---|---:|---:|---:|---:|---:|", rows)


def main() -> None:
    runs = [read_run(path) for path in sorted(RESULTS.glob("*/*.json"))]
    print(f"Cards:\n\n{cards_table(runs)}\n")
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        groups.setdefault((run["arguments"], run["recipe"]), []).append(run)
    for (arguments, recipe), members in sorted(groups.items()):
        title = f"{recipe} recipe with `{arguments}`" if arguments else f"{recipe} recipe as shipped"
        print(f"{title}:\n\n{group_table(members)}\n")
    print("NVIDIA driver:", ", ".join(sorted({run["driver"] for run in runs})))
    print("Model and dataset snapshots:", ", ".join(sorted({s for run in runs for s in run["snapshots"]})))


if __name__ == "__main__":
    main()
