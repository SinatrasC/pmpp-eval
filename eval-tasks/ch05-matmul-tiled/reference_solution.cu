#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <cstdio>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true){
  if(code != cudaSuccess){
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

#ifndef TILE
#define TILE 16
#endif

__global__ void TiledMatMulKernelRef(const float* __restrict__ M,
                                     const float* __restrict__ N,
                                     float* __restrict__ P,
                                     int m, int n, int o)
{
    __shared__ float Ms[TILE][TILE];
    __shared__ float Ns[TILE][TILE];

    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0..m)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0..o)

    float acc = 0.f;
    int phases = (n + TILE - 1) / TILE;

    for(int ph=0; ph<phases; ++ph){
        int kM = ph*TILE + threadIdx.x; // column into M
        int kN = ph*TILE + threadIdx.y; // row    into N

        // Load M tile (row, kM)
        if(row < m && kM < n)  Ms[threadIdx.y][threadIdx.x] = M[row*n + kM];
        else                   Ms[threadIdx.y][threadIdx.x] = 0.f;

        // Load N tile (kN, col)
        if(kN < n && col < o)  Ns[threadIdx.y][threadIdx.x] = N[kN*o + col];
        else                   Ns[threadIdx.y][threadIdx.x] = 0.f;

        __syncthreads();

        #pragma unroll
        for(int k=0;k<TILE;k++){
            acc += Ms[threadIdx.y][k] * Ns[k][threadIdx.x];
        }

        __syncthreads();
    }

    if(row < m && col < o){
        P[row*o + col] = acc;
    }
}

extern "C"
void launch_tiled_matmul(const float* M_h, const float* N_h, float* P_h,
                         int m, int n, int o)
{
    if(m==0 || n==0 || o==0) return;

    size_t bytesM = size_t(m)*n*sizeof(float);
    size_t bytesN = size_t(n)*o*sizeof(float);
    size_t bytesP = size_t(m)*o*sizeof(float);

    const int GUARD = 128;

    float *M_d=nullptr, *N_d=nullptr, *Pguard_d=nullptr;
    gpuErrchk(cudaMalloc(&M_d, bytesM));
    gpuErrchk(cudaMalloc(&N_d, bytesN));
    gpuErrchk(cudaMalloc(&Pguard_d, bytesP + 2*GUARD*sizeof(float)));

    gpuErrchk(cudaMemset(Pguard_d, 0x7B, GUARD*sizeof(float)));
    gpuErrchk(cudaMemset(Pguard_d + GUARD + m*o, 0x7B, GUARD*sizeof(float)));

    gpuErrchk(cudaMemcpy(M_d, M_h, bytesM, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(N_d, N_h, bytesN, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid( (o + TILE - 1)/TILE, (m + TILE - 1)/TILE );

    float* P_d = Pguard_d + GUARD;

    TiledMatMulKernelRef<<<grid, block>>>(M_d, N_d, P_d, m, n, o);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    std::vector<float> P_with_guard(m*o + 2*GUARD);
    gpuErrchk(cudaMemcpy(P_with_guard.data(), Pguard_d,
                         P_with_guard.size()*sizeof(float),
                         cudaMemcpyDeviceToHost));

    std::memcpy(P_h, P_with_guard.data()+GUARD, bytesP);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(Pguard_d);
}