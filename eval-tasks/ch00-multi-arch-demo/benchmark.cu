#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <numeric>

__global__ void vec_add(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ c,
                        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

static void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

int main() {
    constexpr int N = 1 << 26; // ~67M elements
    constexpr int BYTES = N * sizeof(float);
    constexpr int RUNS = 200;

    std::vector<float> host_a(N), host_b(N);
    std::iota(host_a.begin(), host_a.end(), 0.0f);
    std::iota(host_b.begin(), host_b.end(), 1.0f);

    float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_a, BYTES));
    CUDA_CHECK(cudaMalloc(&dev_b, BYTES));
    CUDA_CHECK(cudaMalloc(&dev_c, BYTES));

    CUDA_CHECK(cudaMemcpy(dev_a, host_a.data(), BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, host_b.data(), BYTES, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        vec_add<<<grid, block>>>(dev_a, dev_b, dev_c, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < RUNS; ++i) {
        vec_add<<<grid, block>>>(dev_a, dev_b, dev_c, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / RUNS;

    double bytes_moved = 3.0 * BYTES; // two reads, one write
    double bandwidth = (bytes_moved / avg_ms) * 1e-6; // GB/s

    std::printf("Average kernel time: %.3f ms\n", avg_ms);
    std::printf("Approximate bandwidth: %.2f GB/s\n", bandwidth);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_c));
    return 0;
}
