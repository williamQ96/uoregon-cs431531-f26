# Homework 02: inclusive parallel prefix sums

The implementation is based on the current course starter at
[UOCS431531/uoregon-cs431531-f26-assignments](https://github.com/UOCS431531/uoregon-cs431531-f26-assignments),
commit `e76fd075b1faed5206b99c8510c9a7ca639c6ebe`, and lecture 7, slides 43Ã¢â‚¬â€œ65.
`READ.ME`, `common.c`, and `common.h` are retained from the starter.
The original serial recurrence in `prefix_sum` is retained, with a guard for empty input.
The starter command line remains `./prefix <element count> <seed>`.

## Implementations

- `prefix_sum_p1`: doubling offsets 1,2,4,..., O(n log n) work and O(log n) span. It uses the starter's disposable input copy as the second buffer. **This function may overwrite `src`**; input/output must be distinct.
- `prefix_sum_p2`: the lecture's inclusive binary-tree up-sweep/down-sweep, O(n) work and O(log n) span. For n=2^k, it uses `(n-1)+(n-1-k)` additions. It skips missing tree positions for arbitrary lengths rather than padding the array. It preserves `src`.

Both functions use one persistent OpenMP team, static scheduling, and implicit barriers between levels. Empty/null input with n=0 is allowed. As in the starter, values and partial sums must fit C `int` (32 bits on the measured system). The command-line driver checks allocation failures, invalid arguments, and overflow in its nonnegative generated inputs; verification failures produce an unsuccessful exit status.

## Build and verify

```bash
make
make test
./prefix 1048576 5
```

The C algorithms compile with GCC C99/OpenMP. The independent measurement/test driver in `bench/benchmark.cpp` uses C++17 to call those same C functions. Its oracle is `std::partial_sum`; tests cover 1,580 input/thread cases, including 1/2/3/4/8 threads, empty input, lengths around powers of two, signed random values, zeros, ones, and alternating large signed values. Serial and tree are checked for unchanged input; doubling is tested on a disposable copy.

## Talapas measurements and report

The final report uses **Talapas job 48205110**, completed with exit code `0:0` on `n0187.talapas.uoregon.edu`, October 30, 2026. The job allocated 64 CPUs and 4 GiB for at most 20 minutes; actual elapsed time was 45 seconds. The node has two AMD EPYC 7713 processors (128 physical cores, no SMT exposed, eight NUMA domains). The code was compiled with system GCC/G++ 8.5.0; no modules were loaded.

The primary dataset is `results/talapas-48205110/`: four input sizes (2^16, 2^20, 2^24, 2^25), 1/2/4/8/16/32/64 threads, seven repetitions, 60 configurations and 420 measurements. The report's graph, table and numerical statements are generated from this dataset. At n=2^25, serial took 15.61 ms; the tree version's best time was 18.34 ms at 64 threads (0.85x speedup), and doubling's best was 98.12 ms at 64 threads (0.16x). Neither parallel implementation beat serial in the tested configurations. The tree was substantially faster than doubling. A result below 1x is a measured slowdown, not a failed correctness test.

An initial 32-CPU Talapas run, job 48205061, is retained separately (39 configurations, 273 measurements). A requested 128-CPU extension, job 48205088, was predicted to wait until 12:27 PDT and was canceled while still pending; it consumed no compute time. No 128-thread measurements are claimed. Local WSL diagnostic data remains in `results/local-wsl-20261030/`; it is not mixed into the Talapas report.

Each configuration receives one excluded warmup and seven measured trials. Configurations are shuffled reproducibly per repetition. The deterministic bounded input is `x[i] = i % 201 - 100`. A disposable input copy is prepared outside timing for every implementation, matching the starter's timing boundary. Algorithm-internal output copies, any final doubling-buffer copy, team overhead and barriers are timed. Allocation, input generation, reference/checksum validation and CSV I/O are excluded. All arrays use the same parallel first-touch policy. Threads use core places and close binding; requested team sizes and `SLURM_CPUS_PER_TASK` are checked. This memory placement can affect NUMA/cache locality and the serial/parallel ratio.

Raw seconds, checksums, hardware/allocation details, source hashes, Slurm accounting and job stdout/stderr are retained. The final job's six source-file hashes match the local source exactly. `summary.csv` reports medians, first/third quartiles, speedup `T_serial/T_parallel`, and efficiency `speedup/threads`. `source_revision.txt` records the submitted revision; `submitted-prefix.srun` preserves the job script, and `submission.json` preserves resource overrides.

Regenerate figures and the one-page ACM double-column report from the measured dataset:

```bash
python scripts/analyze.py results/talapas-48205110
cd report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

The analysis validates sample counts, input/thread configurations and checksums. It requires the dataset's non-secret `metadata.json` and `environment.txt`. Python dependencies are in `requirements.txt`.
