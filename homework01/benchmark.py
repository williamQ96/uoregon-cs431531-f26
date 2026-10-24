"""Run the measurement grid; retain every repetition, not just averages."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--repeats", type=int, default=20)
parser.add_argument("--max-threads", type=int, default=32)
parser.add_argument("--output", type=Path, default=Path("results"))
args = parser.parse_args()
if args.repeats < 20 or args.max_threads < 1:
    parser.error("Use at least 20 repetitions and a positive thread limit")
if "SLURM_JOB_ID" not in os.environ:
    parser.error("Run this experiment inside a Talapas Slurm allocation")
if args.max_threads > int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
    parser.error("Thread count exceeds the Slurm CPU allocation")
args.output.mkdir(parents=True, exist_ok=False)
threads = [n for n in (1, 2, 4, 8, 16, 32) if n <= args.max_threads]
sizes = [10000, 100000, 1000000, 10000000, 100000000]
configs = []
for n in sizes:
    configs.append(("serial", n, 1))
    for method in ("reduction", "monte_carlo"):
        configs.extend((method, n, t) for t in threads)
    if n <= 1000000:
        for method in ("critical", "atomic"):
            configs.extend((method, n, t) for t in threads)
# Fixed randomized configuration order reduces systematic warm-up/order bias.
random.Random(431531).shuffle(configs)
def command_output(argv):
    result = subprocess.run(argv, capture_output=True, text=True)
    return result.stdout + result.stderr
metadata = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(), "platform": platform.platform(),
    "compiler": command_output(["gcc", "--version"]),
    "cpu": command_output(["lscpu"]),
    "slurm_job": command_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]]),
    "environment": {key: value for key, value in os.environ.items()
                    if key.startswith("OMP_") or key in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT")},
    "repeats": args.repeats, "seed_base": 431531, "configuration_order": configs,
    "command": " ".join(__import__("sys").argv),
    "source_sha256": __import__("hashlib").sha256(Path("pi.c").read_bytes()).hexdigest(),
    "executable_sha256": __import__("hashlib").sha256(Path("pi").read_bytes()).hexdigest(),
    "compile_flags": "make default: -O3 -g -std=c99 -Wall -Wextra -Wpedantic -fopenmp -lm",
}
(args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
with (args.output / "raw.csv").open("w", newline="") as output:
    writer = None
    for i, (method, n, count) in enumerate(configs, 1):
        result = subprocess.run(["./pi", "--csv", method, str(n), str(count), str(args.repeats), "431531"],
                                capture_output=True, text=True, check=True)
        reader = csv.DictReader(io.StringIO(result.stdout))
        rows = list(reader)
        if len(rows) != args.repeats:
            raise RuntimeError("Missing benchmark repetitions")
        if writer is None:
            writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
            writer.writeheader()
        writer.writerows(rows)
        output.flush()
        print(f"{i}/{len(configs)} {method} n={n} threads={count}", flush=True)
print(f"Completed {len(configs) * args.repeats} measurements", flush=True)
