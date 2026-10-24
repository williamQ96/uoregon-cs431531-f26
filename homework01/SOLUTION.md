# Homework 01: implementation and reproduction

## What is implemented

`pi.c` contains five methods:

1. `serial`: midpoint integration of the unit quarter-circle, multiplied by four.
2. `critical`: `omp parallel for` with one named critical-section update per slice.
3. `atomic`: the same integration with one atomic update per slice (`calcPi_P1`).
4. `reduction`: an additional optimized integration baseline using private sums.
5. `monte_carlo`: uniformly sample the unit square and reduce an integer hit count.

The circle-area interpretation is explicitly

    pi ~= (4/N) * sum(sqrt(1 - ((i + 1/2)/N)^2)),  i = 0,...,N-1.

Lecture 5, slides 43-46, introduces rectangle integration of the semicircle
`sqrt(1-x*x)` on `[-1,1]`. That integral is pi/2. By symmetry,
`pi = 2 * integral[-1,1] = 4 * integral[0,1]` for this integrand. The code uses
N midpoint rectangles on `[0,1]`; this gives the same area-based estimate of pi
without evaluating both halves. The endpoint derivative is singular; its
discretization error differs from that of the alternative integral of
`4/(1+x*x)`.

The original `calcPi_Serial`, `calcPi_P1`, and `calcPi_P2` names are retained.
`calcPi_P2` is a fixed-seed Monte Carlo wrapper; the CLI supports arbitrary seeds.
The original TSC helper files remain for reference but are not used. Timings use
OpenMP's portable wall-clock timer, avoiding a one-second TSC calibration and
CPU-frequency assumptions on every process invocation. No `-ffast-math` or
architecture-specific compiler flag is used.

Monte Carlo uses SplitMix64's mixing function with a sample-index counter. Each
sample uses two distinct counter inputs; no `rand()` state or shared RNG lock
is used. A seed and sample count define the same points at every thread count.
This makes the integer hit count reproducible across schedules. It does not
make estimates from different thread counts independent statistical replicates.
Sampling `[0,1)^2` uses the same area ratio as the full-square dart experiment
in Lecture 5, slide 47: the probability of a hit is pi/4.

The critical and atomic updates follow Lecture 5, slides 37-38, and Lecture 6,
slides 33-34. The additional reduction baseline and static loop schedule are
covered in Lecture 6, slides 67-68 and 80-81.

## Build and correctness checks

On a Linux compute allocation with GCC, OpenMP and Python 3:

```bash
make
make test
OMP_NUM_THREADS=4 ./pi 1000000 1000000 431531
./pi --csv atomic 100000 4 20 431531
```

The CSV mode is `--csv METHOD samples threads repeats seed`. It performs an
untimed warm-up and emits one record per timed run. Each repetition uses
`seed + run`; the integration methods ignore the seed. Timed sections include
the algorithm and OpenMP fork/join/barriers, and exclude printing and disk I/O.
Serial mode requires one thread. Invalid/zero/overflowing counts are rejected.
Very large accepted counts are a numeric input range, not a practical runtime
promise. All estimates and absolute errors use `acos(-1)` as the reference.

`tests/test_pi.py` checks independent numerical quadrature, all synchronization
variants at 1/2/4 threads, a Python integer RNG oracle (including seed wraparound),
uneven work sizes, thread-count invariance, distinct repetition seeds, a Monte
Carlo accuracy sanity bound, and command-line error cases.

The recorded experiment and its correctness checks ran on Talapas. A local
check on October 23, 2026 also passed under WSL Ubuntu with GCC 15.2.0 and
`-Werror`. The report uses only the recorded Talapas timings.

## Talapas experiment

`pi.srun` requests the `cs399` account, compute partition, one task with 32 CPUs,
4 GiB memory and a 15-minute limit. The `7713` constraint selects AMD Milan
hardware. These settings match the recorded job; account and hardware
availability must be checked before submitting a new run.

```bash
# From homework01 on a Talapas login node; output must exist before sbatch.
mkdir -p output
sbatch --test-only pi.srun
sbatch pi.srun
```

The job compiles and runs the tests on its allocated compute node before
benchmarking. `benchmark.py` refuses to run outside Slurm or request more
threads than allocated. Each run uses a fresh `results/run-JOBID` directory.
It also writes a local `metadata.json`, which is excluded from Git. The
published raw data, metrics and job logs are sufficient to regenerate figures;
`analyze.py` does not require that local metadata file.

The full grid uses 20 repetitions, 1/2/4/8/16/32 threads, and 10^4 to 10^8
slices/guesses for serial, reduction and Monte Carlo. Critical and atomic use
10^4 to 10^6 slices to keep contention experiments bounded. There are 101
configurations and 2,020 timed records. Configuration order is shuffled with a
fixed seed; repetitions within a configuration are consecutive. This reduces
order bias without claiming complete randomization of individual trials.
OpenMP uses static scheduling, disabled dynamic teams, core places, spread
binding and passive waiting. CPU allocation is dedicated by Slurm; the whole
physical node is not requested exclusively, so shared-node effects remain.

## Analyze and build the report

Install Python plotting dependencies in a personal environment if necessary:
`numpy` and `matplotlib`. Then, from homework01:

```bash
python3 analyze.py results/run-48008499
make report
```

The analysis verifies at least 20 distinct repetition records per configuration
and exact Monte Carlo agreement across thread counts, then writes `summary.csv`,
`metrics.json`, and PDF/PNG figures. Timing plots use arithmetic means and sample
standard deviations; Monte Carlo accuracy uses RMSE across 20 seeds. Speedup
uses each parallel algorithm's own one-thread mean, not a different workload.
Do not pool the same seeds across thread counts as extra independent samples.

`report/main.tex` is an 11-point, two-column LaTeX report. Its three figures are
generated from the raw Talapas records. The included `report/main.pdf` is the
compiled report. The checked-in starter `output/sample_*` files were not used
as measurements and were left unchanged.

The assignment is submitted through the GitHub repository. Commit and push
any further changes before grading.
