#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <algorithm>

#ifndef TILE
#define TILE 16
#endif
#ifndef MAX_RADIUS
#define MAX_RADIUS 8
#endif

// Declarations from student/reference build:
extern "C" void setFilterConstant(const float* h_filter, int r);
__global__ void conv2d_tiled_constant_kernel(const float* in, float* out, int H, int W, int r);

// ---------------- CPU oracle ----------------
static void conv2d_cpu(const std::vector<float>& in, std::vector<float>& out,
                       int H, int W, const std::vector<float>& filt, int r)
{
    const int K = 2*r + 1;
    auto F = [&](int dy, int dx){ return filt[(dy+r)*K + (dx+r)]; };

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    int ny = y + dy, nx = x + dx;
                    float v = 0.0f;
                    if (0 <= nx && nx < W && 0 <= ny && ny < H) v = in[ny*W + nx];
                    acc += F(dy,dx) * v;
                }
            }
            out[y*W + x] = acc;
        }
    }
}

// ---------------- helpers ----------------
static void cudaCheck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(2);
    }
}

static bool almostEqual(const std::vector<float>& a, const std::vector<float>& b,
                        float atol=1e-5f, float rtol=1e-5f)
{
    if (a.size() != b.size()) return false;
    for (size_t i=0;i<a.size();++i) {
        float diff = std::fabs(a[i]-b[i]);
        float tol  = atol + rtol*std::fabs(b[i]);
        if (diff > tol) return false;
    }
    return true;
}

static void fill_adversarial(std::vector<float>& v, int mode) {
    // 0: zeros, 1: ramp x+y, 2: sin pattern, 3: random
    const int n = (int)v.size();
    switch (mode % 4) {
        case 0: std::fill(v.begin(), v.end(), 0.0f); break;
        case 1: for (int i=0;i<n;++i) v[i] = (float)i * 0.001f; break;
        case 2: for (int i=0;i<n;++i) v[i] = std::sin(0.01f * (float)i); break;
        case 3: {
            std::mt19937 rng(1234u);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (int i=0;i<n;++i) v[i] = dist(rng);
        } break;
    }
}

static void make_filter(std::vector<float>& f, int r, int kind) {
    const int K = 2*r + 1;
    f.assign(K*K, 0.0f);
    if (kind % 3 == 0) {
        // Box blur
        const float w = 1.0f / (K*K);
        std::fill(f.begin(), f.end(), w);
    } else if (kind % 3 == 1) {
        // Delta (should reproduce input)
        f[(r)*K + (r)] = 1.0f;
    } else {
        // Simple separable-ish center-heavy kernel
        for (int dy=-r; dy<=r; ++dy)
            for (int dx=-r; dx<=r; ++dx)
                f[(dy+r)*K + (dx+r)] = 1.0f / (1.0f + (float)(dx*dx+dy*dy));
    }
}

// ---------------- one test ----------------
static bool run_one(int H, int W, int r, int img_mode, int filt_kind, int blk=0)
{
    if (r > MAX_RADIUS) return false;
    const int K = 2*r + 1;

    std::vector<float> h_in(H*W), h_out_cpu(H*W), h_out_gpu(H*W);
    std::vector<float> h_in_copy(H*W);
    std::vector<float> filt;

    fill_adversarial(h_in, img_mode);
    h_in_copy = h_in;
    make_filter(filt, r, filt_kind);

    // CPU oracle
    conv2d_cpu(h_in, h_out_cpu, H, W, filt, r);

    // Device buffers
    float *d_in=nullptr, *d_out=nullptr;
    cudaCheck(cudaMalloc(&d_in,  H*W*sizeof(float)), "malloc d_in");
    cudaCheck(cudaMalloc(&d_out, H*W*sizeof(float)), "malloc d_out");
    // Sentinels
    cudaCheck(cudaMemset(d_out, 0xAD, H*W*sizeof(float)), "memset d_out");

    cudaCheck(cudaMemcpy(d_in, h_in.data(), H*W*sizeof(float), cudaMemcpyHostToDevice),
              "cpy in->d");
    setFilterConstant(filt.data(), r);

    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1)/TILE, (H + TILE - 1)/TILE);
    const size_t smemBytes = (TILE + 2*r) * (TILE + 2*r) * sizeof(float);

    conv2d_tiled_constant_kernel<<<grid, block, smemBytes>>>(d_in, d_out, H, W, r);
    cudaError_t ke = cudaGetLastError();
    if (ke != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(ke));
        cudaFree(d_in); cudaFree(d_out);
        return false;
    }
    cudaCheck(cudaDeviceSynchronize(), "sync");

    cudaCheck(cudaMemcpy(h_out_gpu.data(), d_out, H*W*sizeof(float), cudaMemcpyDeviceToHost),
              "cpy d->out");

    // Check input immutability
    std::vector<float> h_in_after(H*W);
    cudaCheck(cudaMemcpy(h_in_after.data(), d_in, H*W*sizeof(float), cudaMemcpyDeviceToHost),
              "cpy check in");

    cudaFree(d_in); cudaFree(d_out);

    if (!almostEqual(h_in, h_in_after)) {
        std::fprintf(stderr, "Input array was modified!\n");
        return false;
    }

    // Exactness (within tolerance)
    if (!almostEqual(h_out_cpu, h_out_gpu)) {
        std::fprintf(stderr, "Mismatch H=%d W=%d r=%d mode=%d kind=%d\n",
                     H,W,r,img_mode,filt_kind);
        return false;
    }
    return true;
}

int main() {
    struct Case { int H,W,r; };
    const Case cases[] = {
        {1,1,1}, {7,7,1}, {15,15,1}, {16,16,1}, {17,17,1},     // TILE boundary tests
        {19,13,2}, {31,31,2}, {32,32,2}, {33,33,2},            // 2*TILE boundary tests
        {32,32,3}, {45,67,2}, {48,48,3},                       // 3*TILE tests
        {63,64,1}, {64,63,1}, {64,64,2},                       // 4*TILE tests
        {96,128,3}, {128,96,1}, {127,129,2}                    // Large asymmetric tests
    };

    bool all_ok = true;
    int total = 0, ok = 0;

    for (const auto& cs : cases) {
        for (int img_mode = 0; img_mode < 4; ++img_mode) {
            for (int kind = 0; kind < 3; ++kind) {
                ++total;
                bool pass = run_one(cs.H, cs.W, cs.r, img_mode, kind);
                ok += pass ? 1 : 0;
                std::printf("Test H=%d W=%d r=%d img=%d filt=%d ... %s\n",
                            cs.H, cs.W, cs.r, img_mode, kind, pass?"OK":"FAIL");
                if (!pass) all_ok = false;
            }
        }
    }

    std::printf("Summary: %d/%d passed\n", ok, total);
    return all_ok ? 0 : 1;
}