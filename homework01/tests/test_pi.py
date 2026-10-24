"""Black-box correctness checks, including a separate integer PRNG oracle."""
import argparse
import csv
import io
import math
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--exe", default="./pi")
args = parser.parse_args()
exe = str(Path(args.exe).resolve())
env = dict(os.environ, OMP_NUM_THREADS="4", OMP_DYNAMIC="FALSE")

def run(method, n, threads=1, repeats=1, seed=431531):
    result = subprocess.run([exe, "--csv", method, str(n), str(threads), str(repeats), str(seed)],
                            env=env, check=True, capture_output=True, text=True)
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert len(rows) == repeats
    for row in rows:
        assert int(row["threads"]) == threads
        assert math.isfinite(float(row["pi"])) and float(row["seconds"]) >= 0
    return rows

mask = (1 << 64) - 1
def uniform(counter, seed):
    x = (seed + 0x9e3779b97f4a7c15 * (counter + 1)) & mask
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
    x ^= x >> 31
    return (x >> 11) / 2**53

for n in (1, 3, 1000, 100000):
    reference = 4 * math.fsum(math.sqrt(1 - ((i + 0.5) / n)**2) for i in range(n)) / n
    serial = float(run("serial", n)[0]["pi"])
    assert abs(serial - reference) < 2e-12, (n, serial, reference)
    for method in ("critical", "atomic", "reduction"):
        for threads in (1, 2, 4):
            estimate = float(run(method, n, threads)[0]["pi"])
            assert abs(estimate - reference) < 2e-12, (method, n, threads, estimate)

for seed in (0, 431531, (1 << 64) - 1):
    n = 1001  # Unequal chunks and a non-divisible sample count.
    expected = 4 * sum(uniform(2*i, seed)**2 + uniform(2*i+1, seed)**2 <= 1 for i in range(n)) / n
    for threads in (1, 2, 4):
        actual = float(run("monte_carlo", n, threads, seed=seed)[0]["pi"])
        assert actual == expected, (seed, threads, actual, expected)

mc = run("monte_carlo", 1000000, 4, repeats=3)
assert len({row["seed"] for row in mc}) == 3
for row in mc:
    assert abs(float(row["pi"]) - math.pi) < 0.02

for argv in (["0"], ["-1"], ["1.5"], ["abc"], ["18446744073709551616"],
             ["--csv", "unknown", "100", "1", "1", "0"],
             ["--csv", "atomic", "100", "0", "1", "0"],
             ["--csv", "serial", "100", "2", "1", "0"],
             ["--csv", "atomic", "100", "1", "0", "0"]):
    assert subprocess.run([exe, *argv], env=env, capture_output=True).returncode != 0, argv
assert subprocess.run([exe, "--help"], env=env, capture_output=True).returncode == 0
assert subprocess.run([exe, "1000", "1001", "0"], env=env, capture_output=True).returncode == 0
print("PASS: integration oracle, all synchronization variants, counter-RNG oracle,")
print("      thread-count reproducibility, independent seeds, CLI and boundary cases.")
