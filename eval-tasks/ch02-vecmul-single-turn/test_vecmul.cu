#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>
#include <string>

// Declaration from student's file
__global__ void vecMulKernel(const float* A, const float* B, float* C, int n);

// Helper to check CUDA calls
#define CHECK_CUDA(call) do {                                   \
    cudaError_t err__ = (call);                                  \
    if (err__ != cudaSuccess) {                                  \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n",         \
                     cudaGetErrorString(err__), __FILE__, __LINE__); \
        std::exit(1);                                            \
    }                                                            \
} while(0)

static void cpu_vecmul(const std::vector<float>& A,
                       const std::vector<float>& B,
                       std::vector<float>& C) {
    const int n = static_cast<int>(A.size());
    for (int i = 0; i < n; ++i) C[i] = A[i] * B[i];
}

// Adversarial input patterns to catch off-by-one and other indexing errors
static void fill_inputs(std::vector<float>& A, std::vector<float>& B, int n, int pattern_id) {
    for (int i = 0; i < n; ++i) {
        switch (pattern_id % 4) {
            case 0: // mixed signs, different moduli
                A[i] = float((i * 7) % 101) - 50.0f;    // [-50, 50)
                B[i] = float((i * 13 + 5) % 97) - 30.0f;
                break;
            case 1: // linear + quadratic (guarantees unique products)
                A[i] = 0.1f * float(i) + 1.0f;
                B[i] = 0.001f * float(i * i) + 0.5f;
                break;
            case 2: // primes to avoid short cycles
                A[i] = float((i * 31 + 17) % 257) / 3.0f - 20.0f;
                B[i] = float((i * 37 + 11) % 263) / 7.0f - 15.0f;
                break;
            case 3: // alternating sign with varying magnitude
                A[i] = (i % 2 == 0 ? +1.0f : -1.0f) * (0.3f * float(i) + 2.0f);
                B[i] = (i % 3 == 0 ? -1.0f : +1.0f) * (0.2f * float(i) + 1.0f);
                break;
        }
    }
}

static bool almost_equal(const std::vector<float>& x,
                         const std::vector<float>& y,
                         float tol = 1e-6f) {
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) {
        if (std::fabs(x[i] - y[i]) > tol) {
            std::fprintf(stderr, "Mismatch at %zu: got %.8f, want %.8f\n",
                         i, x[i], y[i]);
            return false;
        }
    }
    return true;
}

static constexpr int PAD = 32;              // canary padding
static constexpr float CANARY = -12345.0f;  // sentinel
static constexpr float SENTINEL = +987654.0f; // output init value

static int run_case(int n, int block, int pattern_id) {
    std::printf("Test n=%d, block=%d, pattern=%d ... ", n, block, pattern_id);
    std::fflush(stdout);

    // Host data
    std::vector<float> hA(n), hB(n), hC(n), hRef(n);
    fill_inputs(hA, hB, n, pattern_id);
    cpu_vecmul(hA, hB, hRef);

    // Device buffers WITH padding so OOB writes touch canaries
    size_t bytes_main = n * sizeof(float);
    size_t bytes_pad  = PAD * sizeof(float);
    size_t bytes_all  = bytes_main + 2 * bytes_pad;

    float *dA_full=nullptr, *dB_full=nullptr, *dC_full=nullptr;
    if (bytes_all > 0) {
        CHECK_CUDA(cudaMalloc((void**)&dA_full, bytes_all));
        CHECK_CUDA(cudaMalloc((void**)&dB_full, bytes_all));
        CHECK_CUDA(cudaMalloc((void**)&dC_full, bytes_all));
    }

    // carve pointers so student sees only the main segment
    float* dA = reinterpret_cast<float*>(reinterpret_cast<char*>(dA_full) + bytes_pad);
    float* dB = reinterpret_cast<float*>(reinterpret_cast<char*>(dB_full) + bytes_pad);
    float* dC = reinterpret_cast<float*>(reinterpret_cast<char*>(dC_full) + bytes_pad);

    // prepare canary-filled host buffers
    std::vector<float> hA_padded(PAD + n + PAD, CANARY);
    std::vector<float> hB_padded(PAD + n + PAD, CANARY);
    std::vector<float> hC_padded(PAD + n + PAD, CANARY);
    
    // copy main data into center, init output with sentinel
    std::copy(hA.begin(), hA.end(), hA_padded.begin() + PAD);
    std::copy(hB.begin(), hB.end(), hB_padded.begin() + PAD);
    std::fill(hC_padded.begin() + PAD, hC_padded.begin() + PAD + n, SENTINEL);

    // upload A/B/C + canaries
    if (bytes_all > 0) {
        CHECK_CUDA(cudaMemcpy(dA_full, hA_padded.data(), bytes_all, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB_full, hB_padded.data(), bytes_all, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dC_full, hC_padded.data(), bytes_all, cudaMemcpyHostToDevice));
    }

    // Launch
    const int grid = (n + block - 1) / block;
    if (n > 0) {
        vecMulKernel<<<grid, block>>>(dA, dB, dC, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // download result w/ canaries
    std::vector<float> hC_padded_out(PAD + n + PAD, 0.0f);
    if (bytes_all > 0) {
        CHECK_CUDA(cudaMemcpy(hC_padded_out.data(), dC_full, bytes_all, cudaMemcpyDeviceToHost));
    }

    // check main results
    hC.assign(n, 0.0f);
    std::copy(hC_padded_out.begin() + PAD, hC_padded_out.begin() + PAD + n, hC.begin());

    bool ok = almost_equal(hC, hRef);

    // check output canaries (no OOB writes)
    for (int k = 0; k < PAD; ++k) {
        if (hC_padded_out[k] != CANARY) ok = false;
        if (hC_padded_out[PAD + n + k] != CANARY) ok = false;
    }

    // verify outputs fully written (no sentinel values remain)
    for (int i = 0; i < n; ++i) {
        if (hC[i] == SENTINEL) ok = false;
    }

    // **NEW: verify A/B unchanged (both canaries and main data)**
    if (bytes_all > 0) {
        std::vector<float> chk_A(bytes_all/sizeof(float));
        std::vector<float> chk_B(bytes_all/sizeof(float));
        CHECK_CUDA(cudaMemcpy(chk_A.data(), dA_full, bytes_all, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(chk_B.data(), dB_full, bytes_all, cudaMemcpyDeviceToHost));
        
        // Check canaries
        for (int k = 0; k < PAD; ++k) {
            if (chk_A[k] != CANARY || chk_A[PAD+n+k] != CANARY) ok = false;
            if (chk_B[k] != CANARY || chk_B[PAD+n+k] != CANARY) ok = false;
        }
        
        // Check main data unchanged
        for (int i = 0; i < n; ++i) {
            if (chk_A[PAD+i] != hA[i] || chk_B[PAD+i] != hB[i]) ok = false;
        }

        CHECK_CUDA(cudaFree(dA_full));
        CHECK_CUDA(cudaFree(dB_full));
        CHECK_CUDA(cudaFree(dC_full));
    }

    std::printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    // Enhanced adversarial sizes: comprehensive bounds check coverage
    const int sizes[]  = {0, 1, 2, 17, 31, 63, 64, 65, 127, 128, 129, 255, 256, 257, 384, 511, 512, 513, 3000};
    const int blocks[] = {32, 63, 64, 128, 192, 256, 512}; // mix of powers-of-2 and odd sizes

    std::puts("VecMul tests:");
    int failures = 0;
    int pat = 0;
    for (int n : sizes) {
        for (int b : blocks) {
            failures += run_case(n, b, pat++);
        }
    }
    return failures ? 1 : 0;
}