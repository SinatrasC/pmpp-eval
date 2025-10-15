// test_matmul.cu
// Self-contained test harness with CPU oracle, multiple sizes, adversarial
// patterns, input immutability checks, and strict comparison.
// Build modes:
//   -DBUILD_STUDENT     -> links launch_student()
//   -DBUILD_REFERENCE   -> links launch_reference()

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include <algorithm>
#include <cassert>

#if !defined(BUILD_STUDENT) && !defined(BUILD_REFERENCE)
#error "Define BUILD_STUDENT or BUILD_REFERENCE when building."
#endif

#ifdef BUILD_STUDENT
extern "C" void launch_student(const float* A, const float* B, float* C,
                               int M, int N, int K, int blockSize);
static inline void launch_impl(const float* A, const float* B, float* C,
                               int M, int N, int K, int blk) {
    launch_student(A,B,C,M,N,K,blk);
}
#elif defined(BUILD_REFERENCE)
extern "C" void launch_reference(const float* A, const float* B, float* C,
                                 int M, int N, int K, int blockSize);
static inline void launch_impl(const float* A, const float* B, float* C,
                               int M, int N, int K, int blk) {
    launch_reference(A,B,C,M,N,K,blk);
}
#endif

static void checkCuda(cudaError_t st, const char* what) {
    if (st != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(st));
        std::exit(2);
    }
}

static void cpu_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K)
{
    // C = A[M x N] * B[N x K]
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            double acc = 0.0;
            for (int k = 0; k < N; ++k) {
                acc += static_cast<double>(A[i * N + k]) *
                       static_cast<double>(B[k * K + j]);
            }
            C[i * K + j] = static_cast<float>(acc);
        }
    }
}

enum Pattern { P_ZERO, P_SEQ, P_ALT, P_SINLIKE };

static void fill_pattern(std::vector<float>& v, int rows, int cols, Pattern p) {
    v.resize(rows * cols);
    if (rows * cols == 0) return;
    switch (p) {
        case P_ZERO:
            std::fill(v.begin(), v.end(), 0.0f);
            break;
        case P_SEQ:
            for (int r=0; r<rows; ++r)
                for (int c=0; c<cols; ++c)
                    v[r*cols + c] = static_cast<float>((r+1) * 0.1f + (c+1) * 0.01f);
            break;
        case P_ALT:
            for (int r=0; r<rows; ++r)
                for (int c=0; c<cols; ++c)
                    v[r*cols + c] = ((r+c)&1) ? -2.0f : 3.0f;
            break;
        case P_SINLIKE:
            for (int r=0; r<rows; ++r)
                for (int c=0; c<cols; ++c) {
                    float x = (r*cols + c) * 0.001f;
                    v[r*cols + c] = std::sin(x) * 0.5f + 0.25f;
                }
            break;
    }
}

static bool nearly_equal(const std::vector<float>& a,
                         const std::vector<float>& b,
                         float atol=1e-3f, float rtol=1e-3f)
{
    if (a.size() != b.size()) return false;
    for (size_t i=0;i<a.size();++i) {
        float aa=a[i], bb=b[i];
        float diff = std::fabs(aa - bb);
        if (diff > atol + rtol * std::fabs(bb)) {
            std::fprintf(stderr, "Mismatch at %zu: got %.6f, exp %.6f (|diff|=%.6g)\n",
                         i, aa, bb, diff);
            return false;
        }
    }
    return true;
}

static int run_case(int M, int N, int K, Pattern pA, Pattern pB, int blockSize)
{
    std::printf("Test M=%d N=%d K=%d (patternA=%d, patternB=%d) ... ",
                M, N, K, int(pA), int(pB));
    std::fflush(stdout);

    std::vector<float> hA, hB;
    fill_pattern(hA, M, N, pA);
    fill_pattern(hB, N, K, pB);

    std::vector<float> hA_copy = hA; // immutability check
    std::vector<float> hB_copy = hB;

    std::vector<float> hC(M * K, 1337.0f);    // sentinel
    std::vector<float> hC_ref(M * K, 0.0f);

    // CPU oracle
    cpu_gemm(hA.data(), hB.data(), hC_ref.data(), M, N, K);

    // Device buffers
    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    if (M*N > 0) checkCuda(cudaMalloc(&dA, sizeof(float)*M*N), "cudaMalloc dA");
    if (N*K > 0) checkCuda(cudaMalloc(&dB, sizeof(float)*N*K), "cudaMalloc dB");
    if (M*K > 0) checkCuda(cudaMalloc(&dC, sizeof(float)*M*K), "cudaMalloc dC");

    if (M*N > 0) checkCuda(cudaMemcpy(dA, hA.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice), "cpy A");
    if (N*K > 0) checkCuda(cudaMemcpy(dB, hB.data(), sizeof(float)*N*K, cudaMemcpyHostToDevice), "cpy B");
    if (M*K > 0) checkCuda(cudaMemcpy(dC, hC.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice), "init C");

    // Launch (skip if any dimension is 0)
    if (M > 0 && N > 0 && K > 0) {
        launch_impl(dA, dB, dC, M, N, K, blockSize);
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(), "sync");
    }

    if (M*K > 0) checkCuda(cudaMemcpy(hC.data(), dC, sizeof(float)*M*K, cudaMemcpyDeviceToHost), "cpy C back");

    // Check outputs against oracle
    bool ok = nearly_equal(hC, hC_ref);
    if (!ok) {
        std::fprintf(stderr, "Result mismatch\n");
    }

    // Check inputs immutability (device -> host)
    if (M*N > 0) {
        std::vector<float> hA_after(M*N);
        checkCuda(cudaMemcpy(hA_after.data(), dA, sizeof(float)*M*N, cudaMemcpyDeviceToHost), "cpy A back");
        if (!nearly_equal(hA_after, hA_copy, 1e-6f, 1e-6f)) {
            std::fprintf(stderr, "Input A was modified on device\n");
            ok = false;
        }
    }
    if (N*K > 0) {
        std::vector<float> hB_after(N*K);
        checkCuda(cudaMemcpy(hB_after.data(), dB, sizeof(float)*N*K, cudaMemcpyDeviceToHost), "cpy B back");
        if (!nearly_equal(hB_after, hB_copy, 1e-6f, 1e-6f)) {
            std::fprintf(stderr, "Input B was modified on device\n");
            ok = false;
        }
    }

    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);

    std::puts(ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int main()
{
    // A matrix of tests: include edge cases and non-multiples of 16
    struct Case { int M,N,K; Pattern pA,pB; };
    const Case cases[] = {
        {0,0,0,      P_ZERO, P_ZERO},
        {1,1,1,      P_SEQ,  P_SEQ },
        {3,5,7,      P_ALT,  P_SEQ },
        {16,16,16,   P_SEQ,  P_SEQ },
        {17,17,17,   P_SEQ,  P_ALT },
        {31,37,19,   P_ALT,  P_SINLIKE},
        {64,32,80,   P_SINLIKE, P_SEQ},
        {128,64,33,  P_SEQ,  P_ALT },
        {96,200,48,  P_ALT,  P_ALT },
        {256,256,256,P_SINLIKE, P_SINLIKE}
    };

    std::puts("Matrix Multiplication Tiled Tests:");
    int failures = 0;
    const int blockSize = 16; // fixed for this task

    for (const auto& cs : cases) {
        failures += run_case(cs.M, cs.N, cs.K, cs.pA, cs.pB, blockSize);
    }

    std::printf("\nResult: %d/%d tests passed\n", 
                (int)(sizeof(cases)/sizeof(cases[0])) - failures,
                (int)(sizeof(cases)/sizeof(cases[0])));
    
    return failures == 0 ? 0 : 1;
}