#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <limits.h>
#include <errno.h>
#include "common.h"
#include "prefix.h"


void usage(int argc, char** argv);
void verify(int* sol, int* ans, int n);
void prefix_sum(int* src, int* prefix, int n);
void prefix_sum_p1(int* src, int* prefix, int n);
void prefix_sum_p2(int* src, int* prefix, int n);


#ifndef PREFIX_SUM_NO_MAIN
int main(int argc, char** argv)
{
    // get inputs
    uint32_t n = 1048576;
    unsigned int seed = time(NULL);
    if(argc == 3) {
        char* end;
        errno = 0;
        unsigned long long size = strtoull(argv[1], &end, 10);
        if(errno || *end || end == argv[1] || argv[1][0] == '-' || size > INT_MAX) {
            fprintf(stderr, "Element count must be between 0 and INT_MAX\n");
            return EXIT_FAILURE;
        }
        n = (uint32_t)size;
        errno = 0;
        unsigned long long parsed_seed = strtoull(argv[2], &end, 10);
        if(errno || *end || end == argv[2] || argv[2][0] == '-' || parsed_seed > UINT_MAX) {
            fprintf(stderr, "Invalid unsigned random seed\n");
            return EXIT_FAILURE;
        }
        seed = (unsigned int)parsed_seed;
    } else if(argc == 1) {
        usage(argc, argv);
        printf("using %"PRIu32" elements and time as seed\n", n);
    } else {
        usage(argc, argv);
        return EXIT_FAILURE;
    }


    // set up data
    int* prefix_array = (int*) AlignedMalloc(sizeof(int) * n);
    int* input_array = (int*) AlignedMalloc(sizeof(int) * n);
    if(!prefix_array || !input_array) {
        fprintf(stderr, "Input/output allocation failed\n");
        return EXIT_FAILURE;
    }
    srand(seed);
    int64_t total = 0;
    for(uint32_t i = 0; i < n; i++) {
        input_array[i] = rand() % 100;
        total += input_array[i];
    }
    if(total > INT_MAX) {
        fprintf(stderr, "Generated prefix sums would overflow int; use fewer elements\n");
        return EXIT_FAILURE;
    }


    // set up timers
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();


    // execute serial prefix sum and use it as ground truth
    start_t = ReadTSC();
    prefix_sum(input_array, prefix_array, n);
    end_t = ReadTSC();
    printf("Time to do O(N-1) prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));


    // execute parallel prefix sum which uses a NlogN algorithm
    int* input_array1 = (int*) AlignedMalloc(sizeof(int) * n);
    int* prefix_array1 = (int*) AlignedMalloc(sizeof(int) * n);
    if(!input_array1 || !prefix_array1) {
        fprintf(stderr, "Parallel workspace allocation failed\n");
        return EXIT_FAILURE;
    }
    memcpy(input_array1, input_array, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p1(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do O(NlogN) //prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);


    // execute parallel prefix sum which uses a 2(N-1) algorithm
    memcpy(input_array1, input_array, sizeof(int) * n);
    memset(prefix_array1, 0, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p2(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do 2(N-1) //prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);


    // free memory
    AlignedFree(prefix_array);
    AlignedFree(input_array);
    AlignedFree(input_array1);
    AlignedFree(prefix_array1);


    return 0;
}

void usage(int argc, char** argv)
{
    (void)argc;
    fprintf(stderr, "usage: %s <# elements> <rand seed>\n", argv[0]);
}


void verify(int* sol, int* ans, int n)
{
    int err = 0;
    for(int i = 0; i < n; i++) {
        if(sol[i] != ans[i]) {
            err++;
        }
    }
    if(err != 0) {
        fprintf(stderr, "There was an error: %d\n", err);
        exit(EXIT_FAILURE);
    } else {
        fprintf(stdout, "Pass\n");
    }
}
#endif

void prefix_sum(int* src, int* prefix, int n)
{
    if(n <= 0) return;
    prefix[0] = src[0];
    for(int i = 1; i < n; i++) {
        prefix[i] = src[i] + prefix[i - 1];
    }
}

void prefix_sum_p1(int* src, int* prefix, int n)
{
    if(n <= 0) return;
    /* O(n log n) doubling. The starter supplies a disposable input copy,
     * used here as the second buffer. */
    #pragma omp parallel default(none) shared(src, prefix, n)
    {
        int* current = prefix;
        int* next = src;
        #pragma omp for schedule(static)
        for(int i = 0; i < n; ++i) prefix[i] = src[i];
        for(size_t stride = 1; stride < (size_t)n; stride *= 2) {
            #pragma omp for schedule(static)
            for(int i = 0; i < n; ++i) {
                next[i] = current[i];
                if((size_t)i >= stride) next[i] += current[i - stride];
            }
            /* The implicit for-barrier completes writes before the next
             * level. Each thread swaps its own identical pointers. */
            int* temporary = current;
            current = next;
            next = temporary;
        }
        if(current != prefix) {
            #pragma omp for schedule(static)
            for(int i = 0; i < n; ++i) prefix[i] = current[i];
        }
    }
}

void prefix_sum_p2(int* src, int* prefix, int n)
{
    if(n <= 0) return;
    size_t tree_size = 1;
    while(tree_size < (size_t)n) tree_size *= 2;
    /* Inclusive binary-tree up/down scan from lecture 7. Missing positions
     * are skipped, allowing arbitrary n without padded storage. */
    #pragma omp parallel default(none) shared(src, prefix, n, tree_size)
    {
        #pragma omp for schedule(static)
        for(int i = 0; i < n; ++i) prefix[i] = src[i];
        for(size_t stride = 1; stride < tree_size; stride *= 2) {
            const size_t pairs = (size_t)n / (2 * stride);
            #pragma omp for schedule(static)
            for(size_t k = 0; k < pairs; ++k) {
                const size_t right = (k + 1) * 2 * stride - 1;
                prefix[right] += prefix[right - stride];
            }
        }
        /* Unlike an exclusive scan, the root is not reset to zero. */
        for(size_t stride = tree_size / 4; stride > 0; stride /= 2) {
            const size_t first = 3 * stride - 1;
            const size_t count = first < (size_t)n ? 1 + ((size_t)n - 1 - first) / (2 * stride) : 0;
            #pragma omp for schedule(static)
            for(size_t k = 0; k < count; ++k) {
                const size_t right = first + k * 2 * stride;
                prefix[right] += prefix[right - stride];
            }
        }
    }
}
