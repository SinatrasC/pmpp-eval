// test_radix_multiradix.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <cassert>

extern "C" void radix_sort_multiradix_host(unsigned int* data, int n);

// --- Utilities ---
static void check(cudaError_t e, const char* m) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e));
        std::exit(2);
    }
}

static bool equal_vec(const std::vector<unsigned int>& a, const std::vector<unsigned int>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

static void cpu_radix_sort(std::vector<unsigned int>& data) {
    std::stable_sort(data.begin(), data.end());
}

static void fill_pattern(std::vector<unsigned int>& v, int pat) {
    std::mt19937 rng(1234 + pat);
    switch (pat % 6) {
        case 0: // Random
            {
                std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
                for (size_t i = 0; i < v.size(); ++i) v[i] = dist(rng);
            }
            break;
        case 1: // Ascending
            for (size_t i = 0; i < v.size(); ++i) v[i] = (unsigned int)i;
            break;
        case 2: // Descending
            for (size_t i = 0; i < v.size(); ++i) v[i] = (unsigned int)(v.size() - 1 - i);
            break;
        case 3: // Many duplicates
            for (size_t i = 0; i < v.size(); ++i) v[i] = (unsigned int)(i % 15);  // 15 buckets for multiradix
            break;
        case 4: // Single value
            for (size_t i = 0; i < v.size(); ++i) v[i] = 42;
            break;
        case 5: // Powers of 2 and nearby
            for (size_t i = 0; i < v.size(); ++i) v[i] = (1u << (i % 32)) + (i % 3);
            break;
    }
}

// --- Test ---
int main() {
    printf("radix-multiradix-fullsort-single tests\n");

    struct Case { int n; const char* name; };
    const Case cases[] = {
        {0, "n=0"},
        {1, "n=1"},
        {2, "n=2"},
        {7, "n=7"},
        {32, "n=32"},
        {128, "n=128"},
        {1000, "n=1000"},
        {4096, "n=4096"},
        {10000, "n=10000"},
        {65536, "n=65536"}
    };

    const size_t GUARD = 1024;
    const unsigned int SENT = 0xDEADBEEF;

    int total = 0, passed = 0;

    for (const auto& cs : cases) {
        for (int pat = 0; pat < 6; ++pat) {
            ++total;

            // Host input
            std::vector<unsigned int> data(std::max(cs.n, 0));
            fill_pattern(data, pat);
            std::vector<unsigned int> data_ref = data;
            cpu_radix_sort(data_ref);
            const int n = (int)data.size();

            // Guarded buffer
            unsigned int* d_data_all = nullptr;
            check(cudaMalloc(&d_data_all, (n + 2 * GUARD) * sizeof(unsigned int)), "malloc data");

            std::vector<unsigned int> h_data_guard(n + 2 * GUARD, SENT);
            if (n > 0) std::copy(data.begin(), data.end(), h_data_guard.begin() + GUARD);

            check(cudaMemcpy(d_data_all, h_data_guard.data(), (n + 2 * GUARD) * sizeof(unsigned int), cudaMemcpyHostToDevice), "H2D data");

            unsigned int* d_data = d_data_all + GUARD;

            // Sort
            radix_sort_multiradix_host(d_data, n);
            check(cudaGetLastError(), "sort");
            check(cudaDeviceSynchronize(), "sync");

            // Download
            check(cudaMemcpy(h_data_guard.data(), d_data_all, (n + 2 * GUARD) * sizeof(unsigned int), cudaMemcpyDeviceToHost), "D2H data");

            // Validate
            bool ok = true;

            // Output equals reference
            std::vector<unsigned int> data_got(n);
            if (n > 0) std::copy(h_data_guard.begin() + GUARD, h_data_guard.begin() + GUARD + n, data_got.begin());
            ok = ok && equal_vec(data_got, data_ref);

            // Guards unchanged (detect OOB)
            auto guard_ok = [&](const std::vector<unsigned int>& g) {
                for (size_t i = 0; i < GUARD; i++) {
                    if (g[i] != SENT || g[g.size() - 1 - i] != SENT) return false;
                }
                return true;
            };
            ok = ok && guard_ok(h_data_guard);

            printf("Case %-8s pat=%d -> %s\n", cs.name, pat, ok ? "OK" : "FAIL");
            if (ok) ++passed;

            cudaFree(d_data_all);
        }
    }

    printf("Summary: %d / %d passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}