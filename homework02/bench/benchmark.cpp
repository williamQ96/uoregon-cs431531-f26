#include "../prefix.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace {
using value_type = int;
using Clock = std::chrono::steady_clock;
volatile std::uint64_t observed_checksum = 0;

std::vector<std::size_t> list(const std::string& text) {
    std::vector<std::size_t> out;
    std::istringstream stream(text);
    std::string part;
    while (std::getline(stream, part, ',')) {
        if (part.empty() || part.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("Expected comma-separated positive integers");
        const auto number = std::stoull(part);
        if (!number || number > (std::size_t{1} << 30))
            throw std::runtime_error("Size/thread value out of supported range");
        out.push_back(static_cast<std::size_t>(number));
    }
    if (out.empty()) throw std::runtime_error("Empty size/thread list");
    return out;
}

std::uint64_t checksum(const std::vector<value_type>& data) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto value : data) {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

void check(const std::vector<value_type>& actual,
           const std::vector<value_type>& expected,
           const std::string& algorithm, int threads) {
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if (mismatch.first != actual.end()) {
        throw std::runtime_error(algorithm + " incorrect at index " +
            std::to_string(mismatch.first - actual.begin()) + " n=" +
            std::to_string(actual.size()) + " threads=" + std::to_string(threads));
    }
}

void tests(int max_threads) {
    prefix_sum(nullptr, nullptr, 0);
    prefix_sum_p1(nullptr, nullptr, 0);
    prefix_sum_p2(nullptr, nullptr, 0);
    std::vector<std::size_t> sizes(66);
    std::iota(sizes.begin(), sizes.end(), std::size_t{0});
    for (const auto n : {127,128,129,255,256,257,1023,1024,1025,10007,65535,65536,65537})
        sizes.push_back(static_cast<std::size_t>(n));
    std::mt19937_64 rng(43153102);
    std::size_t cases = 0;
    for (int threads : {1,2,3,4,8}) {
        if (threads > max_threads) continue;
        omp_set_num_threads(threads);
        for (const auto n : sizes) {
            for (int pattern = 0; pattern < 4; ++pattern) {
                std::vector<value_type> input(n), reference(n), output(n), work(n);
                for (std::size_t i = 0; i < n; ++i) {
                    if (pattern == 0) input[i] = static_cast<value_type>(rng() % 2001) - 1000;
                    if (pattern == 1) input[i] = 0;
                    if (pattern == 2) input[i] = 1;
                    if (pattern == 3) input[i] = (i % 2 ? -1 : 1) * value_type{1000000000};
                }
                const auto original = input;
                work = input;
                // Independent standard-library oracle also checks the serial baseline.
                std::partial_sum(input.begin(), input.end(), reference.begin());
                prefix_sum(input.data(), output.data(), n);
                check(output, reference, "serial", threads);
                prefix_sum_p1(work.data(), output.data(), static_cast<int>(n));
                check(output, reference, "doubling", threads);
                prefix_sum_p2(input.data(), output.data(), n);
                check(output, reference, "tree", threads);
                if (input != original) throw std::runtime_error("Input was modified");
                ++cases;
            }
        }
    }
    std::cout << "PASS " << cases << " input/thread cases, all 3 algorithms; "
                 "null/empty, zero, ones, signed random, large signed values, non-powers of two\n";
}

void benchmark(const std::vector<std::size_t>& sizes,
               const std::vector<std::size_t>& counts, int repetitions,
               const std::string& csv_path) {
    if (repetitions < 1 || repetitions > 1000) throw std::runtime_error("Invalid repetition count");
    for (const auto p : counts)
        if (p > static_cast<std::size_t>(omp_get_num_procs()) || p > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Thread request exceeds visible CPU allocation");
    for (const auto p : counts) {
        const char* allocation = std::getenv("SLURM_CPUS_PER_TASK");
        if (allocation && p > std::stoull(allocation))
            throw std::runtime_error("Thread request exceeds SLURM_CPUS_PER_TASK");
        int actual = 0;
        #pragma omp parallel num_threads(static_cast<int>(p)) shared(actual)
        {
            #pragma omp single
            actual = omp_get_num_threads();
        }
        if (actual != static_cast<int>(p))
            throw std::runtime_error("OpenMP did not create the requested thread count");
    }
    std::ofstream csv(csv_path);
    if (!csv) throw std::runtime_error("Cannot open CSV output");
    csv << "algorithm,n,threads,repeat,seconds,checksum\n" << std::setprecision(12);
    std::mt19937_64 rng(43153102);
    for (const auto n : sizes) {
        std::vector<value_type> input(n), reference(n), output(n), work(n);
        // Shared parallel first-touch policy for all implementations.
        omp_set_num_threads(static_cast<int>(*std::max_element(counts.begin(), counts.end())));
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            input[i] = static_cast<value_type>(i % 201) - 100;
            output[i] = 0;
            work[i] = 0;
            reference[i] = 0;
        }
        prefix_sum(input.data(), reference.data(), n);
        const auto expected_checksum = checksum(reference);
        struct Trial { int algorithm; int threads; };
        std::vector<Trial> trials{{0,1}};
        for (const auto threads : counts) {
            trials.push_back({1,static_cast<int>(threads)});
            trials.push_back({2,static_cast<int>(threads)});
        }
        const auto run = [&](const Trial trial) {
            if (trial.algorithm == 0) prefix_sum(work.data(), output.data(), static_cast<int>(n));
            if (trial.algorithm == 1) prefix_sum_p1(work.data(), output.data(), static_cast<int>(n));
            if (trial.algorithm == 2) prefix_sum_p2(work.data(), output.data(), static_cast<int>(n));
        };
        // Warm up each algorithm/thread configuration once; exclude from CSV.
        for (const auto trial : trials) {
            omp_set_num_threads(trial.threads);
            std::copy(input.begin(), input.end(), work.begin());
            run(trial);
            check(output, reference, "warmup", trial.threads);
        }
        for (int repeat = 0; repeat < repetitions; ++repeat) {
            std::shuffle(trials.begin(), trials.end(), rng);
            for (const auto trial : trials) {
                omp_set_num_threads(trial.threads);
                std::copy(input.begin(), input.end(), work.begin());
                const auto begin = Clock::now();
                run(trial);
                const auto end = Clock::now();
                const auto seconds = std::chrono::duration<double>(end - begin).count();
                check(output, reference, "measured trial", trial.threads);
                observed_checksum = checksum(output);
                if (observed_checksum != expected_checksum) throw std::runtime_error("Checksum mismatch");
                const char* name = trial.algorithm == 0 ? "serial" : trial.algorithm == 1 ? "doubling" : "tree";
                csv << name << ',' << n << ',' << trial.threads << ',' << repeat << ','
                    << seconds << ',' << observed_checksum << '\n';
                csv.flush();
            }
        }
        std::cerr << "Completed n=" << n << "\n";
    }
}
}

int main(int argc, char** argv) {
    try {
        omp_set_dynamic(0);
        bool test = false;
        int repetitions = 7;
        int max_threads = std::min(8, omp_get_num_procs());
        auto sizes = list("65536,1048576,16777216");
        auto counts = list("1,2,4,8,16,32");
        std::string csv = "raw.csv";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--test") { test = true; continue; }
            if (arg == "--benchmark") continue;
            if (i + 1 >= argc) throw std::runtime_error("Missing option value");
            const std::string value = argv[++i];
            if (arg == "--sizes") sizes = list(value);
            else if (arg == "--threads") counts = list(value);
            else if (arg == "--repeats") repetitions = std::stoi(value);
            else if (arg == "--max-threads") max_threads = std::stoi(value);
            else if (arg == "--csv") csv = value;
            else throw std::runtime_error("Unknown option: " + arg);
        }
        if (max_threads < 1 || max_threads > omp_get_num_procs())
            throw std::runtime_error("Invalid test thread limit");
        if (test) tests(max_threads);
        else benchmark(sizes, counts, repetitions, csv);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
