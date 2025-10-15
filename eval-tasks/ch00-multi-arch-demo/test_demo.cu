#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>

#ifndef REFERENCE_BUILD
extern "C" void launch_reference(const float* a, const float* b, float* c, int n);
#endif
extern "C" void launch_student(const float* a, const float* b, float* c, int n);

#define CUDA_CHECK(expr)                                                   \
    do {                                                                   \
        cudaError_t _err = (expr);                                          \
        if (_err != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s:%d -> %s\n",              \
                        __FILE__, __LINE__, cudaGetErrorString(_err));     \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

static bool vectors_close(const std::vector<float>& lhs,
                          const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (std::fabs(lhs[i] - rhs[i]) > 1e-5f) {
            std::fprintf(stderr, "mismatch at %zu: %.6f vs %.6f\n", i, lhs[i], rhs[i]);
            return false;
        }
    }
    return true;
}

int main() {
    constexpr int N = 1 << 12;
    std::vector<float> host_a(N), host_b(N), host_ref(N), host_stu(N);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        host_a[i] = dist(rng);
        host_b[i] = dist(rng);
    }

    float *dev_a = nullptr, *dev_b = nullptr, *dev_ref = nullptr, *dev_stu = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_ref, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_stu, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dev_a, host_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, host_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

#ifdef REFERENCE_BUILD
    launch_student(dev_a, dev_b, dev_ref, N);
#else
    launch_reference(dev_a, dev_b, dev_ref, N);
#endif
    launch_student(dev_a, dev_b, dev_stu, N);

    CUDA_CHECK(cudaMemcpy(host_ref.data(), dev_ref, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_stu.data(), dev_stu, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_ref));
    CUDA_CHECK(cudaFree(dev_stu));

    bool ok = vectors_close(host_ref, host_stu);
    std::puts(ok ? "Student matches reference." : "Student diverges from reference.");
    return ok ? 0 : 1;
}
