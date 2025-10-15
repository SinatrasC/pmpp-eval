#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <cstring>

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

#ifndef COARSE_FACTOR
#define COARSE_FACTOR 4
#endif

#ifdef USE_STUDENT
extern "C" void launch_student(const float* A_d, const float* B_d, float* C_d,
                               int M, int N, int K);
#else
extern "C" void launch_reference(const float* A_d, const float* B_d, float* C_d,
                                 int M, int N, int K);
#endif

#define CUDA_CHECK(ans) do { \
  cudaError_t err = (ans); \
  if (err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(1); \
  } \
} while(0)

static void cpu_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K)
{
    // C = A[M×N] * B[N×K]
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += (double)A[i*N + k] * (double)B[k*K + j];
            }
            C[i*K + j] = (float)sum;
        }
    }
}

static void fill_adversarial(std::vector<float>& v, int pattern)
{
    // 4 patterns to catch indexing issues
    const int n = (int)v.size();
    for (int i = 0; i < n; ++i) {
        switch (pattern % 4) {
            case 0: v[i] = (float)(i % 97) * 0.01f; break;
            case 1: v[i] = (float)((i * 1315423911u) & 0xFF) * 0.02f; break;
            case 2: v[i] = std::sin(0.001f * (float)i); break;
            default:v[i] = (i & 1) ? -0.5f : 0.5f; break;
        }
    }
}

static bool almost_equal(const std::vector<float>& a,
                         const std::vector<float>& b,
                         float atol=1e-4f, float rtol=1e-4f)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::fabs(a[i] - b[i]);
        float tol  = atol + rtol * std::fabs(b[i]);
        if (diff > tol) {
            // Uncomment for debugging:
            // std::cerr << "Mismatch at " << i << ": " << a[i] << " vs " << b[i] << "\n";
            return false;
        }
    }
    return true;
}

static int run_one(int M, int N, int K, int pattern)
{
    // Host buffers
    std::vector<float> A(M*N), B(N*K), C(M*K), C_ref(M*K);
    std::vector<float> A_copy, B_copy;

    fill_adversarial(A, pattern);
    fill_adversarial(B, pattern+1);
    A_copy = A; B_copy = B;

    // Sentinels in C to detect unwritten outputs
    std::fill(C.begin(), C.end(), 1337.0f);

    // Device buffers
    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    CUDA_CHECK(cudaMalloc(&A_d, A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B_d, B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C_d, C.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(A_d, A.data(), A.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B.data(), B.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C.data(), C.size()*sizeof(float), cudaMemcpyHostToDevice));

    // Launch
#ifdef USE_STUDENT
    launch_student(A_d, B_d, C_d, M, N, K);
#else
    launch_reference(A_d, B_d, C_d, M, N, K);
#endif
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    CUDA_CHECK(cudaMemcpy(C.data(), C_d, C.size()*sizeof(float), cudaMemcpyDeviceToHost));
    // Check inputs were not modified
    std::vector<float> A_after(M*N), B_after(N*K);
    CUDA_CHECK(cudaMemcpy(A_after.data(), A_d, A_after.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(B_after.data(), B_d, B_after.size()*sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    if (A_after != A_copy) {
        std::cerr << "[FAIL] Input A modified.\n";
        return 1;
    }
    if (B_after != B_copy) {
        std::cerr << "[FAIL] Input B modified.\n";
        return 1;
    }

    // CPU oracle
    cpu_gemm(A.data(), B.data(), C_ref.data(), M, N, K);

    // Check all outputs written (not left as sentinel), unless M*K==0
    if (M*K > 0) {
        for (int i = 0; i < M*K; ++i) {
            if (C[i] == 1337.0f) {
                std::cerr << "[FAIL] Unwritten output at index " << i << "\n";
                return 1;
            }
        }
    }

    if (!almost_equal(C, C_ref)) {
        std::cerr << "[FAIL] Numerical mismatch.\n";
        return 1;
    }

    return 0; // PASS
}

int main()
{
    struct Case { int M,N,K; };
    // Include zeros, tiny, odd sizes, non-multiples of tile/coarse, and larger
    std::vector<Case> cases = {
        {0,0,0},
        {1,1,1},
        {1,3,2},
        {3,2,1},
        {7,5,9},
        {16,16,16},
        {31,17,33},
        {32,32,64},
        {64,48,32},
        {96,64,128},
        {127, 129, 63},   // odd + prime-ish
        {128, 96, 72}
    };

    int fails = 0;
    int pattern = 0;
    for (size_t t = 0; t < cases.size(); ++t) {
        const auto& c = cases[t];
        const int err = run_one(c.M, c.N, c.K, pattern++);
        std::printf("Test M=%d N=%d K=%d ... %s\n",
                    c.M, c.N, c.K, err ? "FAIL" : "OK");
        fails += (err != 0);
    }
    if (fails == 0) {
        std::puts("All tests passed.");
        return 0;
    }
    std::printf("%d tests failed.\n", fails);
    return 1;
}