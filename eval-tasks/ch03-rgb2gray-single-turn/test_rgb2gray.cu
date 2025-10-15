#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cassert>

#define CUDA_CHECK(ans) do { cudaError_t err = (ans); if (err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(2); } } while(0)

static constexpr int PAD = 64;            // bigger padding; arrays are bytes
static constexpr unsigned char CANARY = 0xA5;

// CPU oracle (match device rounding)
static unsigned char clamp_u8(int v) {
    return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}
static void cpu_rgb2gray(const std::vector<unsigned char>& R,
                         const std::vector<unsigned char>& G,
                         const std::vector<unsigned char>& B,
                         std::vector<unsigned char>& Y,
                         int n) {
    for (int i = 0; i < n; ++i) {
        float y = 0.299f * (float)R[i] + 0.587f * (float)G[i] + 0.114f * (float)B[i];
        int yi = (int)std::floor(y + 0.5f);
        Y[i] = clamp_u8(yi);
    }
}

// Deterministic inputs
static void fill_rgb(std::vector<unsigned char>& R,
                     std::vector<unsigned char>& G,
                     std::vector<unsigned char>& B, int n) {
    for (int i = 0; i < n; ++i) {
        R[i] = (unsigned char)((i * 37) % 256);
        G[i] = (unsigned char)((i * 83 + 17) % 256);
        B[i] = (unsigned char)((i * 19 + 251) % 256);
    }
}

// Decl from student OR reference
__global__ void rgb2grayKernel(const unsigned char* R,
                               const unsigned char* G,
                               const unsigned char* B,
                               unsigned char* gray,
                               int n);

static bool run_case(int n, int block) {
    int grid = (n + block - 1) / block;

    std::vector<unsigned char> hR(n), hG(n), hB(n), hY_ref(n), hY(n);
    fill_rgb(hR, hG, hB, n);
    cpu_rgb2gray(hR, hG, hB, hY_ref, n);

    // padded device buffers
    size_t bytes_main = n * sizeof(unsigned char);
    size_t bytes_pad  = PAD * sizeof(unsigned char);
    size_t bytes_all  = bytes_main + 2 * bytes_pad;

    unsigned char *dR_full=nullptr, *dG_full=nullptr, *dB_full=nullptr, *dY_full=nullptr;
    if (bytes_all > 0) {
        CUDA_CHECK(cudaMalloc(&dR_full, bytes_all));
        CUDA_CHECK(cudaMalloc(&dG_full, bytes_all));
        CUDA_CHECK(cudaMalloc(&dB_full, bytes_all));
        CUDA_CHECK(cudaMalloc(&dY_full, bytes_all));
    }

    unsigned char* dR = dR_full + PAD;
    unsigned char* dG = dG_full + PAD;
    unsigned char* dB = dB_full + PAD;
    unsigned char* dY = dY_full + PAD;

    // host padded buffers with canaries
    std::vector<unsigned char> pad(bytes_pad, CANARY);
    std::vector<unsigned char> hR_pad(PAD + n + PAD, CANARY);
    std::vector<unsigned char> hG_pad(PAD + n + PAD, CANARY);
    std::vector<unsigned char> hB_pad(PAD + n + PAD, CANARY);
    std::vector<unsigned char> hY_pad(PAD + n + PAD, CANARY);

    std::copy(hR.begin(), hR.end(), hR_pad.begin() + PAD);
    std::copy(hG.begin(), hG.end(), hG_pad.begin() + PAD);
    std::copy(hB.begin(), hB.end(), hB_pad.begin() + PAD);

    if (bytes_all > 0) {
        CUDA_CHECK(cudaMemcpy(dR_full, hR_pad.data(), bytes_all, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dG_full, hG_pad.data(), bytes_all, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB_full, hB_pad.data(), bytes_all, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dY_full, hY_pad.data(), bytes_all, cudaMemcpyHostToDevice));
    }

    if (n > 0) {
        rgb2grayKernel<<<grid, block>>>(dR, dG, dB, dY, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<unsigned char> hY_pad_out(PAD + n + PAD, 0);
    if (bytes_all > 0) {
        CUDA_CHECK(cudaMemcpy(hY_pad_out.data(), dY_full, bytes_all, cudaMemcpyDeviceToHost));
    }
    hY.assign(n, 0);
    std::copy(hY_pad_out.begin() + PAD, hY_pad_out.begin() + PAD + n, hY.begin());

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (hY[i] != hY_ref[i]) { ok = false; break; }
    }

    // canaries intact?
    for (int k = 0; k < PAD; ++k) {
        if (hY_pad_out[k] != CANARY) ok = false;
        if (hY_pad_out[PAD + n + k] != CANARY) ok = false;
    }

    // also check inputs not overwritten
    if (bytes_all > 0) {
        std::vector<unsigned char> chk(bytes_all);
        CUDA_CHECK(cudaMemcpy(chk.data(), dR_full, bytes_all, cudaMemcpyDeviceToHost));
        for (int k = 0; k < PAD; ++k) if (chk[k] != CANARY || chk[PAD+n+k] != CANARY) ok = false;
        CUDA_CHECK(cudaMemcpy(chk.data(), dG_full, bytes_all, cudaMemcpyDeviceToHost));
        for (int k = 0; k < PAD; ++k) if (chk[k] != CANARY || chk[PAD+n+k] != CANARY) ok = false;
        CUDA_CHECK(cudaMemcpy(chk.data(), dB_full, bytes_all, cudaMemcpyDeviceToHost));
        for (int k = 0; k < PAD; ++k) if (chk[k] != CANARY || chk[PAD+n+k] != CANARY) ok = false;

        CUDA_CHECK(cudaFree(dR_full));
        CUDA_CHECK(cudaFree(dG_full));
        CUDA_CHECK(cudaFree(dB_full));
        CUDA_CHECK(cudaFree(dY_full));
    }

    std::printf("  Test n=%d, block=%d ... %s\n", n, block, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    bool all = true;
    // Enhanced coverage: adversarial sizes for bounds checking
    const int sizes[]  = {0, 1, 17, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1024, 4093};
    const int blocks[] = {32, 64, 128, 192, 256, 512};

    std::puts("RGB2Gray tests:");
    for (int n : sizes) {
        for (int b : blocks) all = run_case(n, b) && all;
    }
    return all ? 0 : 1;
}