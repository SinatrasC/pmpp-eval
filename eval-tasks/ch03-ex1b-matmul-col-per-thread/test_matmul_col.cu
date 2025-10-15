#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(2);} } while(0)
#endif

// student or reference impl is linked; declare it here
extern "C" __global__
void matrixMulColKernel(const float*, const float*, float*, int);

static void cpu_mm(const std::vector<float>& A,
                   const std::vector<float>& B,
                   std::vector<float>& C,
                   int n) {
    for (int r=0; r<n; ++r){
        for (int c=0; c<n; ++c){
            float s=0.f;
            for (int j=0;j<n;++j) s += A[r*n+j]*B[j*n+c];
            C[r*n+c]=s;
        }
    }
}

static void fill_pattern(std::vector<float>& v, int n, int which){
    for (int i=0;i<n;i++){
        switch(which%4){
            case 0: v[i] = (float)((i*131) % 257) * 0.01f; break;
            case 1: v[i] = std::sin(0.1f*i); break;
            case 2: v[i] = (i%7==0)? -3.25f : 2.0f; break;
            default: v[i] = (float)(i%11) - 5.0f; break;
        }
    }
}

static bool almost_equal(const std::vector<float>& a,
                         const std::vector<float>& b,
                         float tol=1e-3f){
    if (a.size()!=b.size()) return false;
    for (size_t i=0;i<a.size();++i){
        float da = std::fabs(a[i]-b[i]);
        float ma = std::max(1.0f, std::fabs(a[i])+std::fabs(b[i]));
        if (da > tol*ma){
            // show first mismatch
            fprintf(stderr,"Mismatch at %zu: got %.6f expect %.6f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

static bool run_case(int n, int blockSize){
    const float CANARY = 1337.0f;

    std::vector<float> hM(n*n), hN(n*n), hC(n*n, CANARY), hRef(n*n);
    fill_pattern(hM, n*n, 0);
    fill_pattern(hN, n*n, 1);

    // build oracle
    cpu_mm(hM, hN, hRef, n);

    // device buffers with guard space
    size_t bytes = n*n*sizeof(float);

    // Inputs immutable check (copy back after run)
    std::vector<float> hM_orig = hM, hN_orig = hN;

    float *dM=nullptr, *dN=nullptr, *dC=nullptr;
    if (bytes > 0) {
        CHECK_CUDA(cudaMalloc(&dM, bytes));
        CHECK_CUDA(cudaMalloc(&dN, bytes));
        CHECK_CUDA(cudaMalloc(&dC, bytes));

        CHECK_CUDA(cudaMemcpy(dM, hM.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dN, hN.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dC, hC.data(), bytes, cudaMemcpyHostToDevice));
    }

    if (n > 0) {
        int grid = (n + blockSize - 1) / blockSize;
        dim3 gridDim(grid), blockDim(blockSize);

        matrixMulColKernel<<<gridDim, blockDim>>>(dM, dN, dC, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    if (bytes > 0) {
        CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        std::vector<float> hM_after(n*n), hN_after(n*n);
        CHECK_CUDA(cudaMemcpy(hM_after.data(), dM, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hN_after.data(), dN, bytes, cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(dM));
        CHECK_CUDA(cudaFree(dN));
        CHECK_CUDA(cudaFree(dC));

        // immutability
        if (!almost_equal(hM_after, hM_orig) || !almost_equal(hN_after, hN_orig)){
            fprintf(stderr,"Input arrays modified\n");
            return false;
        }
    }

    // correctness (CANARY ensures unwritten elements are caught)
    if (!almost_equal(hC, hRef)){
        fprintf(stderr,"Wrong result (n=%d, block=%d)\n", n, blockSize);
        return false;
    }
    return true;
}

int main(){
    int sizes[] = {0,1,5,17,31,64,65,96};
    int blocks[] = {64,128,256};
    bool ok=true;
    for (int n: sizes){
        for (int b: blocks){
            bool pass = run_case(n,b);
            printf("ColKernel n=%d block=%d ... %s\n", n, b, pass?"OK":"FAIL");
            ok &= pass;
        }
    }
    return ok?0:1;
}