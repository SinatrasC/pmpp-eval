#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cassert>

#define CUDA_CHECK(ans) do { cudaError_t err = (ans); if (err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(2); } } while(0)

static constexpr int PAD = 32;              // canary padding
static constexpr float CANARY = -12345.0f;  // sentinel

// CPU oracle
static void cpu_vec_add(const std::vector<float>& A, const std::vector<float>& B,
                        std::vector<float>& C, int n) {
    for (int i = 0; i < n; ++i) C[i] = A[i] + B[i];
}

// Enhanced deterministic inputs with adversarial numerical patterns
static void fill_inputs(std::vector<float>& A, std::vector<float>& B, int n, int pattern = 0) {
    for (int i = 0; i < n; ++i) {
        switch (pattern % 4) {
            case 0: // Original pattern
                A[i] = float((i * 7) % 101) / 10.0f - 3.0f;
                B[i] = float((i * 13) % 97) / 9.0f - 2.0f;
                break;
            case 1: // Large numbers (test overflow/precision)
                A[i] = float(i * 123456) + 1e6f;
                B[i] = float(i * 654321) - 1e6f;
                break;
            case 2: // Small numbers (test underflow/denormals)
                A[i] = float(i + 1) * 1e-7f;
                B[i] = float(i + 1) * 1e-8f;
                break;
            case 3: // Mixed signs and special values
                A[i] = (i % 2 == 0) ? float(i) : -float(i);
                B[i] = (i % 3 == 0) ? 0.0f : float(i * 0.01f);
                break;
        }
    }
}

// Decl from student OR reference
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n);

// One test instance
static bool run_case(int n, int block, int pattern = 0) {
    int grid = (n + block - 1) / block;

    // host buffers (with canaries)
    std::vector<float> hA(n), hB(n), hC(n), hC_ref(n);
    fill_inputs(hA, hB, n, pattern);
    cpu_vec_add(hA, hB, hC_ref, n);

    // device buffers WITH padding so OOB writes touch canaries
    size_t bytes_main = n * sizeof(float);
    size_t bytes_pad  = PAD * sizeof(float);
    size_t bytes_all  = bytes_main + 2 * bytes_pad;

    float *dA_full=nullptr, *dB_full=nullptr, *dC_full=nullptr;
    if (bytes_all > 0) {
        CUDA_CHECK(cudaMalloc(&dA_full, bytes_all));
        CUDA_CHECK(cudaMalloc(&dB_full, bytes_all));
        CUDA_CHECK(cudaMalloc(&dC_full, bytes_all));
    }

    // carve pointers so student sees only the main segment
    float* dA = reinterpret_cast<float*>(reinterpret_cast<char*>(dA_full) + bytes_pad);
    float* dB = reinterpret_cast<float*>(reinterpret_cast<char*>(dB_full) + bytes_pad);
    float* dC = reinterpret_cast<float*>(reinterpret_cast<char*>(dC_full) + bytes_pad);

    // prepare canary-filled host buffers
    std::vector<float> h_pad(bytes_pad/sizeof(float), CANARY);
    std::vector<float> hA_padded(PAD + n + PAD, CANARY);
    std::vector<float> hB_padded(PAD + n + PAD, CANARY);
    std::vector<float> hC_padded(PAD + n + PAD, CANARY);
    // copy main into center
    std::copy(hA.begin(), hA.end(), hA_padded.begin() + PAD);
    std::copy(hB.begin(), hB.end(), hB_padded.begin() + PAD);

    // upload A/B, init C + canaries
    if (bytes_all > 0) {
        CUDA_CHECK(cudaMemcpy(dA_full, hA_padded.data(), bytes_all, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB_full, hB_padded.data(), bytes_all, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dC_full, hC_padded.data(), bytes_all, cudaMemcpyHostToDevice));
    }

    // launch (skip if n=0)
    if (n > 0) {
        vecAddKernel<<<grid, block>>>(dA, dB, dC, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
    }

    // download result w/ canaries
    std::vector<float> hC_padded_out(PAD + n + PAD, 0.0f);
    if (bytes_all > 0) {
        CUDA_CHECK(cudaMemcpy(hC_padded_out.data(), dC_full, bytes_all, cudaMemcpyDeviceToHost));
    }

    // check main results
    hC.assign(n, 0.0f);
    std::copy(hC_padded_out.begin() + PAD, hC_padded_out.begin() + PAD + n, hC.begin());

    auto nearly_equal = [](float a, float b) {
        float diff = std::fabs(a - b);
        float tol  = 1e-6f * std::max(1.0f, std::fabs(a) + std::fabs(b));
        return diff <= tol;
    };

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (!nearly_equal(hC[i], hC_ref[i])) { ok = false; break; }
    }

    // check canaries (no OOB writes)
    for (int k = 0; k < PAD; ++k) {
        if (hC_padded_out[k] != CANARY) ok = false;
        if (hC_padded_out[PAD + n + k] != CANARY) ok = false;
    }

    // also verify A/B canaries unchanged (no accidental writes to inputs)
    if (bytes_all > 0) {
        std::vector<float> chk(bytes_all/sizeof(float));
        CUDA_CHECK(cudaMemcpy(chk.data(), dA_full, bytes_all, cudaMemcpyDeviceToHost));
        for (int k = 0; k < PAD; ++k) if (chk[k] != CANARY || chk[PAD+n+k] != CANARY) ok = false;
        CUDA_CHECK(cudaMemcpy(chk.data(), dB_full, bytes_all, cudaMemcpyDeviceToHost));
        for (int k = 0; k < PAD; ++k) if (chk[k] != CANARY || chk[PAD+n+k] != CANARY) ok = false;

        CUDA_CHECK(cudaFree(dA_full));
        CUDA_CHECK(cudaFree(dB_full));
        CUDA_CHECK(cudaFree(dC_full));
    }

    std::printf("  Test n=%d, block=%d, pattern=%d ... %s\n", n, block, pattern, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    bool all = true;
    // Enhanced coverage: added adversarial sizes that expose bounds check bugs
    // Focus on cases where grid*block != n to catch missing bounds checks
    const int sizes[]  = {0, 1, 17, 63, 65, 127, 129, 191, 193, 255, 256, 257, 383, 385, 511, 512, 513, 3000};
    const int blocks[] = {32, 64, 128, 192, 256, 512};

    std::puts("VecAdd tests:");
    for (int n : sizes) {
        for (int b : blocks) {
            for (int p = 0; p < 4; ++p) { // Test all 4 numerical patterns
                all = run_case(n, b, p) && all;
            }
        }
    }
    return all ? 0 : 1;
}