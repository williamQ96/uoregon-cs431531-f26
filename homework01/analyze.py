"""Aggregate actual raw runs and generate the report's reproducible figures."""
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path, help="Directory containing raw.csv; metadata.json is optional")
parser.add_argument("--report", type=Path, default=Path("report"))
args = parser.parse_args()
args.report.mkdir(parents=True, exist_ok=True)
figures = args.report / "figures"
figures.mkdir(exist_ok=True)
with (args.run / "raw.csv").open(newline="") as stream:
    raw = list(csv.DictReader(stream))
groups = defaultdict(list)
for row in raw:
    groups[(row["method"], int(row["samples"]), int(row["threads"]))].append(row)
summary = {}
for key, rows in sorted(groups.items()):
    if len(rows) < 20 or len({r["run"] for r in rows}) != len(rows):
        raise ValueError(f"Missing independent repetition records: {key}")
    times = [float(r["seconds"]) for r in rows]
    errors = [float(r["pi"]) - math.pi for r in rows]
    if any(t <= 0 or not math.isfinite(t) for t in times):
        raise ValueError("Invalid timer measurement")
    # The same indexed MC samples must produce exactly the same count at any T.
    if key[0] == "monte_carlo":
        one_thread = groups.get((key[0], key[1], 1))
        if one_thread is None or [(r["seed"], r["pi"]) for r in rows] != [(r["seed"], r["pi"]) for r in one_thread]:
            raise ValueError("Monte Carlo changed with thread count")
    summary[key] = dict(method=key[0], samples=key[1], threads=key[2], repeats=len(rows),
                        seconds_mean=statistics.mean(times), seconds_std=statistics.stdev(times),
                        seconds_min=min(times), seconds_max=max(times),
                        pi_mean=statistics.mean(float(r["pi"]) for r in rows),
                        abs_error_mean=statistics.mean(abs(e) for e in errors),
                        rmse=math.sqrt(statistics.mean(e*e for e in errors)))
with (args.run / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(next(iter(summary.values()))))
    writer.writeheader()
    writer.writerows(summary.values())

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .22, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
colors = {"serial": "#444444", "critical": "#c94b40", "atomic": "#cc871b",
          "reduction": "#196f97", "monte_carlo": "#39834c"}
threads = sorted({t for m, n, t in summary if m == "reduction"})
lock_size = max(n for m, n, t in summary if m == "critical")
max_size = max(n for m, n, t in summary if m == "serial")
def save(fig, name):
    fig.tight_layout(pad=.3)
    fig.savefig(figures / (name + ".pdf"))
    fig.savefig(figures / (name + ".png"), dpi=220)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(3.35, 1.85))
for method in ("critical", "atomic", "reduction"):
    values = [summary[(method, lock_size, t)] for t in threads]
    ax.errorbar(threads, [r["seconds_mean"] for r in values],
                yerr=[r["seconds_std"] for r in values], marker="o", markersize=3,
                capsize=2, label=method.capitalize(), color=colors[method], linewidth=1.2)
ax.axhline(summary[("serial", lock_size, 1)]["seconds_mean"], color=colors["serial"],
           linestyle="--", linewidth=1, label="Serial")
ax.set(xscale="log", yscale="log", xlabel="Threads", ylabel="Wall time (s)")
ax.set_xticks(threads, [str(t) for t in threads])
ax.legend(fontsize=7, ncol=2, loc="best")
save(fig, "synchronization")

fig, ax = plt.subplots(figsize=(3.35, 1.85))
for method in ("reduction", "monte_carlo"):
    baseline = summary[(method, max_size, 1)]["seconds_mean"]
    values = [summary[(method, max_size, t)]["seconds_mean"] for t in threads]
    ax.plot(threads, [baseline / value for value in values], marker="o", markersize=3,
            label=method.replace("_", " ").capitalize(), color=colors[method], linewidth=1.2)
ax.plot(threads, threads, "--", color="gray", label="Ideal", linewidth=1)
ax.set(xscale="log", xlabel="Threads", ylabel="Speedup")
ax.set_xticks(threads, [str(t) for t in threads])
ax.legend(fontsize=7, loc="best")
save(fig, "scaling")

fig, ax = plt.subplots(figsize=(3.35, 1.85))
sizes = sorted(n for m, n, t in summary if m == "serial")
mc = [summary[("monte_carlo", n, max(threads))] for n in sizes]
ax.loglog(sizes, [summary[("serial", n, 1)]["abs_error_mean"] for n in sizes],
          "o-", markersize=3, color=colors["serial"], label="Midpoint absolute error")
ax.loglog(sizes, [r["rmse"] for r in mc], "s-", markersize=3,
          color=colors["monte_carlo"], label="Monte Carlo RMSE")
ax.loglog(sizes, [math.sqrt((4*math.pi-math.pi**2)/n) for n in sizes],
          "--", linewidth=1, color="#a65fa0", label=r"MC theory $\propto N^{-1/2}$")
ax.set(xlabel="Slices / guesses", ylabel="Error relative to pi")
ax.legend(fontsize=7, loc="best")
save(fig, "accuracy")

metadata_path = args.run / "metadata.json"
metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
# Published results retain job identifiers in metrics.json; local runs also
# have metadata.json. Neither is needed to recompute the numerical results.
metrics_path = args.run / "metrics.json"
previous_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
tmax = max(threads)
metrics = {
    "records": len(raw), "configurations": len(summary),
    "repeats": min(row["repeats"] for row in summary.values()),
    "job_id": metadata.get("environment", {}).get("SLURM_JOB_ID", previous_metrics.get("job_id")),
    "hostname": metadata.get("hostname", previous_metrics.get("hostname")),
    "largest_n": max_size, "lock_n": lock_size, "max_threads": tmax,
    "reduction_speedup": summary[("reduction", max_size, 1)]["seconds_mean"] / summary[("reduction", max_size, tmax)]["seconds_mean"],
    "reduction_vs_serial_speedup": summary[("serial", max_size, 1)]["seconds_mean"] / summary[("reduction", max_size, tmax)]["seconds_mean"],
    "mc_speedup": summary[("monte_carlo", max_size, 1)]["seconds_mean"] / summary[("monte_carlo", max_size, tmax)]["seconds_mean"],
    "serial_error_small": summary[("serial", sizes[0], 1)]["abs_error_mean"],
    "serial_error_large": summary[("serial", max_size, 1)]["abs_error_mean"],
    "mc_rmse_small": mc[0]["rmse"], "mc_rmse_large": mc[-1]["rmse"],
    "critical_slowdown": summary[("critical", lock_size, tmax)]["seconds_mean"] / summary[("serial", lock_size, 1)]["seconds_mean"],
    "atomic_slowdown": summary[("atomic", lock_size, tmax)]["seconds_mean"] / summary[("serial", lock_size, 1)]["seconds_mean"],
}
(args.run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
