#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Integrate the unit quarter-circle using midpoint rectangles. */
static double slice(uint64_t i, double width)
{
    double x = ((double)i + 0.5) * width;
    return sqrt(1.0 - x * x);
}

double calcPi_Serial(uint64_t n)
{
    double sum = 0.0, width = 1.0 / (double)n;
    for (uint64_t i = 0; i < n; ++i)
        sum += slice(i, width);
    return 4.0 * width * sum;
}

double calcPi_Critical(uint64_t n)
{
    double sum = 0.0, width = 1.0 / (double)n;
    #pragma omp parallel for schedule(static) default(none) shared(n, width, sum)
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        double term = slice((uint64_t)i, width);
        #pragma omp critical(pi_sum)
        sum += term;
    }
    return 4.0 * width * sum;
}

/* Preserve the starter's integration entry point; P1 uses atomic updates. */
double calcPi_P1(uint64_t n)
{
    double sum = 0.0, width = 1.0 / (double)n;
    #pragma omp parallel for schedule(static) default(none) shared(n, width, sum)
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        double term = slice((uint64_t)i, width);
        #pragma omp atomic update
        sum += term;
    }
    return 4.0 * width * sum;
}

double calcPi_Reduction(uint64_t n)
{
    double sum = 0.0, width = 1.0 / (double)n;
    #pragma omp parallel for schedule(static) default(none) shared(n, width) reduction(+:sum)
    for (int64_t i = 0; i < (int64_t)n; ++i)
        sum += slice((uint64_t)i, width);
    return 4.0 * width * sum;
}

/* SplitMix64's bijective mixer applied to a counter, not shared RNG state.
 * Each sample gets the same pair at every thread count. This is a reproducible
 * simulation PRNG, not a cryptographic generator. Unsigned overflow is defined.
 */
static uint64_t mix64(uint64_t x)
{
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static double uniform01(uint64_t counter, uint64_t seed)
{
    uint64_t bits = mix64(seed + UINT64_C(0x9e3779b97f4a7c15) * (counter + 1));
    return (double)(bits >> 11) * 0x1.0p-53;
}

double calcPi_MonteCarlo(uint64_t n, uint64_t seed)
{
    uint64_t hits = 0;
    #pragma omp parallel for schedule(static) default(none) shared(n, seed) reduction(+:hits)
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        double x = uniform01(2 * (uint64_t)i, seed);
        double y = uniform01(2 * (uint64_t)i + 1, seed);
        hits += (x * x + y * y <= 1.0);
    }
    return 4.0 * (double)hits / (double)n;
}

double calcPi_P2(uint64_t n)
{
    return calcPi_MonteCarlo(n, UINT64_C(431531));
}

static void usage(const char *program)
{
    printf("Usage: %s [steps [guesses [seed]]]\n"
           "       %s --csv METHOD samples threads repeats seed\n"
           "Methods: serial, critical, atomic, reduction, monte_carlo\n"
           "Set OMP_NUM_THREADS for the human-readable mode.\n", program, program);
}

static int parse_uint(const char *text, uint64_t maximum, uint64_t *result)
{
    char *end;
    if (!text[0] || text[0] == '-' || text[0] == '+' || text[0] < '0' || text[0] > '9')
        return 0;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno == ERANGE || *end || value > maximum)
        return 0;
    *result = (uint64_t)value;
    return 1;
}

static double compute(const char *method, uint64_t n, uint64_t seed)
{
    if (!strcmp(method, "serial")) return calcPi_Serial(n);
    if (!strcmp(method, "critical")) return calcPi_Critical(n);
    if (!strcmp(method, "atomic")) return calcPi_P1(n);
    if (!strcmp(method, "reduction")) return calcPi_Reduction(n);
    return calcPi_MonteCarlo(n, seed);
}

static int valid_method(const char *method)
{
    const char *methods[] = {"serial", "critical", "atomic", "reduction", "monte_carlo"};
    for (size_t i = 0; i < sizeof(methods) / sizeof(methods[0]); ++i)
        if (!strcmp(method, methods[i])) return 1;
    return 0;
}

int main(int argc, char **argv)
{
    const double reference = acos(-1.0);
    uint64_t n = 100000, guesses = 100000, seed = 431531;
    omp_set_dynamic(0);
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc > 1 && !strcmp(argv[1], "--csv")) {
        uint64_t threads, repeats;
        if (argc != 7 || !valid_method(argv[2]) ||
            !parse_uint(argv[3], INT64_MAX, &n) || !n ||
            !parse_uint(argv[4], 65536, &threads) || !threads ||
            !parse_uint(argv[5], INT_MAX, &repeats) || !repeats ||
            !parse_uint(argv[6], UINT64_MAX, &seed) ||
            (!strcmp(argv[2], "serial") && threads != 1)) {
            fprintf(stderr, "Invalid CSV arguments. See --help.\n");
            return EXIT_FAILURE;
        }
        omp_set_num_threads((int)threads);
        int actual_threads = 1;
        #pragma omp parallel default(none) shared(actual_threads)
        {
            #pragma omp single
            actual_threads = omp_get_num_threads();
        }
        if (actual_threads != (int)threads) {
            fprintf(stderr, "Requested %d threads, received %d.\n", (int)threads, actual_threads);
            return EXIT_FAILURE;
        }
        /* One untimed warm-up; timed sections include fork/join and barriers. */
        double warmup = compute(argv[2], n < 1000 ? n : 1000, seed);
        if (!isfinite(warmup)) return EXIT_FAILURE;
        puts("method,samples,threads,run,seed,pi,abs_error,seconds");
        for (uint64_t run = 0; run < repeats; ++run) {
            uint64_t run_seed = seed + run;
            double start = omp_get_wtime();
            double estimate = compute(argv[2], n, run_seed);
            double elapsed = omp_get_wtime() - start;
            printf("%s,%" PRIu64 ",%d,%" PRIu64 ",%" PRIu64 ",%.17g,%.17g,%.17g\n",
                   argv[2], n, actual_threads, run, run_seed, estimate,
                   fabs(estimate - reference), elapsed);
            fflush(stdout);
        }
        return EXIT_SUCCESS;
    }
    if (argc > 4 || (argc > 1 && (!parse_uint(argv[1], INT64_MAX, &n) || !n)) ||
        (argc > 2 && (!parse_uint(argv[2], INT64_MAX, &guesses) || !guesses)) ||
        (argc > 3 && !parse_uint(argv[3], UINT64_MAX, &seed))) {
        fprintf(stderr, "Steps and guesses must be positive integers; seed must be unsigned.\n");
        return EXIT_FAILURE;
    }
    if (argc < 3) guesses = n;
    printf("Reference pi: %.17g; OpenMP max threads: %d\n", reference, omp_get_max_threads());
    const char *methods[] = {"serial", "critical", "atomic", "reduction", "monte_carlo"};
    for (size_t i = 0; i < sizeof(methods) / sizeof(methods[0]); ++i) {
        uint64_t samples = i == 4 ? guesses : n;
        double start = omp_get_wtime();
        double estimate = compute(methods[i], samples, seed);
        double elapsed = omp_get_wtime() - start;
        printf("%-12s samples=%" PRIu64 " pi=%.15f abs_error=%.6g seconds=%.9g\n",
               methods[i], samples, estimate, fabs(estimate - reference), elapsed);
    }
    return EXIT_SUCCESS;
}
