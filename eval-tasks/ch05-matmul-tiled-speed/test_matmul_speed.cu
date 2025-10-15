// test_matmul_speed.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <string>
#include <functional>

#ifndef TILE
#define TILE 16
#endif

// Prototypes provided by student & reference TUs
extern "C" void matmul_student(const float*, const float*, float*, int,int,int,int);
extern "C" void matmul_ref_tiled(const float*, const float*, float*, int,int,int,int);
extern "C" void matmul_ref_naive(const float*, const float*, float*, int,int,int);

// Utilities
#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while(0)

static void fill_deterministic(std::vector<float>& v, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : v) x = dist(rng);
}

static bool nearly_equal(const float* a, const float* b, size_t n, float atol=1e-4f, float rtol=1e-3f) {
    for (size_t i=0;i<n;++i) {
        float aa = a[i], bb = b[i];
        float diff = std::fabs(aa - bb);
        float tol = atol + rtol * std::max(std::fabs(aa), std::fabs(bb));
        if (diff > tol) {
            // show a few mismatches then bail
            fprintf(stderr, "Mismatch at %zu: ref=%.6f got=%.6f diff=%.6g tol=%.6g\n", i, aa, bb, diff, tol);
            return false;
        }
    }
    return true;
}

static float time_ms(std::function<void()> fn, int warmup=5, int runs=20) {
    for (int i=0;i<warmup;++i) fn();
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    CUDA_OK(cudaEventRecord(start));
    for (int i=0;i<runs;++i) fn();
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float ms=0.f;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
    return ms / runs;
}

static void run_case(int M, int N, int K, int tile, bool &ok, bool &perf_ok) {
    size_t aN = (size_t)M * N;
    size_t bN = (size_t)N * K;
    size_t cN = (size_t)M * K;

    std::vector<float> hA(aN), hB(bN), hC_ref(cN), hC_stu(cN), hC_naive(cN);
    fill_deterministic(hA, 123);
    fill_deterministic(hB, 456);

    float *dA=nullptr, *dB=nullptr, *dC_ref=nullptr, *dC_stu=nullptr, *dC_naive=nullptr;
    CUDA_OK(cudaMalloc(&dA, aN*sizeof(float)));
    CUDA_OK(cudaMalloc(&dB, bN*sizeof(float)));
    CUDA_OK(cudaMalloc(&dC_ref, cN*sizeof(float)));
    CUDA_OK(cudaMalloc(&dC_stu, cN*sizeof(float)));
    CUDA_OK(cudaMalloc(&dC_naive, cN*sizeof(float)));
    CUDA_OK(cudaMemcpy(dA, hA.data(), aN*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB.data(), bN*sizeof(float), cudaMemcpyHostToDevice));

    // Correctness: compare student vs reference tiled
    matmul_ref_tiled(dA, dB, dC_ref, M, N, K, tile);
    matmul_student  (dA, dB, dC_stu, M, N, K, tile);
    CUDA_OK(cudaMemcpy(hC_ref.data(), dC_ref, cN*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(hC_stu.data(), dC_stu, cN*sizeof(float), cudaMemcpyDeviceToHost));

    bool this_ok = nearly_equal(hC_ref.data(), hC_stu.data(), cN);
    ok = ok && this_ok;

    // Timing: student vs reference tiled (and optional naive)
    auto f_ref = [&](){ matmul_ref_tiled(dA, dB, dC_ref, M, N, K, tile); };
    auto f_stu = [&](){ matmul_student  (dA, dB, dC_stu, M, N, K, tile); };
    auto f_nv  = [&](){ matmul_ref_naive(dA, dB, dC_naive, M, N, K); };

    float t_ref = time_ms(f_ref, 3, 10);
    float t_stu = time_ms(f_stu, 3, 10);
    float t_nv  = time_ms(f_nv , 1,  3); // optional baseline, fewer runs

    printf("M=%4d N=%4d K=%4d | REF %7.3f ms | STU %7.3f ms | NAIVE %7.3f ms | %s\n",
           M, N, K, t_ref, t_stu, t_nv, this_ok ? "OK" : "FAIL");

    // Performance policy: student should not be >25% slower than reference for large cases
    // We only enforce perf for "large-ish" problems where timing noise is lower.
    if (M>=512 && N>=512 && K>=512) {
        bool this_perf_ok = (t_stu <= 1.25f * t_ref);
        perf_ok = perf_ok && this_perf_ok;
        if (!this_perf_ok) {
            fprintf(stderr, "PERF FAIL on %dx%dx%d: student %.3f ms vs ref %.3f ms\n",
                    M,N,K,t_stu,t_ref);
        }
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC_ref); cudaFree(dC_stu); cudaFree(dC_naive);
}

int main() {
    // Smoke check a 0-dimension (no-op) to ensure kernels handle empty gracefully (should be OK)
    // Then a set of sizes including non-multiples of TILE to test guards.
    std::vector<int> Ms = {0, 1, 63, 128, 257, 512, 1024};
    std::vector<int> Ns = {0, 1, 63, 128, 257, 512, 1024};
    std::vector<int> Ks = {0, 1, 61, 128, 259, 512, 1024};

    bool ok = true;
    bool perf_ok = true;

    // Keep total cases reasonable: pair same-index entries for variety
    for (size_t i=0; i<std::min({Ms.size(), Ns.size(), Ks.size()}); ++i) {
        run_case(Ms[i], Ns[i], Ks[i], TILE, ok, perf_ok);
    }

    printf("\nCorrectness: %s\n", ok ? "PASS" : "FAIL");
    printf("Performance (enforced only for >=512 dims): %s\n", perf_ok ? "PASS" : "FAIL");

    // Exit code semantics:
    // - Must pass correctness.
    // - Performance is enforced for big cases; if you prefer "advisory-only", change this.
    return (ok && perf_ok) ? 0 : 1;
}