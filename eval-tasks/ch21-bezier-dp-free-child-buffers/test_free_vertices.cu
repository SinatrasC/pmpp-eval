// ch21-bezier-dp-free-child-buffers / test_free_vertices.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cassert>

#include <vector_types.h>

struct BezierLine {
  float2 CP[3];
  float2* vertexPos; // device-heap pointer
  int     nVertices;
};

__global__ void freeVertexMem(BezierLine* bLines, int nLines);

// Allocate device-heap buffers for each line; write dummy data
__global__ void alloc_vertices_kernel(BezierLine* bLines, int nLines, int nVerts) {
  int lidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (lidx >= nLines) return;
  bLines[lidx].nVertices = nVerts;
  float2* buf = (float2*)malloc((size_t)nVerts * sizeof(float2));
  bLines[lidx].vertexPos = buf;
  if (!buf) { bLines[lidx].nVertices = 0; return; }
  for (int i=0;i<nVerts;i++){
    buf[i].x = (float)lidx;
    buf[i].y = (float)i;
  }
}

static void ck(cudaError_t e, const char* m){
  if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}

int main(){
  std::printf("ch21-bezier-dp-free-child-buffers tests\n");

  // Enable heap for device malloc/free
  ck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64*1024*1024), "set heap");

  const int nLines = 1024;
  const int nVerts = 256;

  // Init host
  std::vector<BezierLine> h(nLines);
  for (int i=0;i<nLines;i++){ h[i].CP[0]={0,0}; h[i].CP[1]={0,0}; h[i].CP[2]={0,0}; h[i].vertexPos=nullptr; h[i].nVertices=0; }

  BezierLine* d=nullptr;
  ck(cudaMalloc(&d, nLines*sizeof(BezierLine)), "malloc d");
  ck(cudaMemcpy(d, h.data(), nLines*sizeof(BezierLine), cudaMemcpyHostToDevice), "H2D");

  dim3 block(256), grid((nLines+block.x-1)/block.x);

  // 1) Allocate
  alloc_vertices_kernel<<<grid, block>>>(d, nLines, nVerts);
  ck(cudaGetLastError(), "alloc launch");
  ck(cudaDeviceSynchronize(), "alloc sync");

  // Check that pointers are non-null for most lines
  ck(cudaMemcpy(h.data(), d, nLines*sizeof(BezierLine), cudaMemcpyDeviceToHost), "D2H check alloc");
  int nonnull_before=0;
  for (int i=0;i<nLines;i++) if (h[i].vertexPos) nonnull_before++;
  std::printf("Non-null before free: %d / %d\n", nonnull_before, nLines);
  if (nonnull_before == 0){ std::printf("Allocation failed for all lines. FAIL\n"); return 1; }

  // 2) Free once
  freeVertexMem<<<grid, block>>>(d, nLines);
  ck(cudaGetLastError(), "free1 launch");
  ck(cudaDeviceSynchronize(), "free1 sync");

  // Validate freed
  ck(cudaMemcpy(h.data(), d, nLines*sizeof(BezierLine), cudaMemcpyDeviceToHost), "D2H after free1");
  int null_after1=0;
  for (int i=0;i<nLines;i++) if (h[i].vertexPos == nullptr) null_after1++;
  std::printf("Null after first free: %d / %d -> %s\n", null_after1, nLines, (null_after1==nLines?"OK":"FAIL"));
  if (null_after1 != nLines) return 1;

  // 3) Free again (idempotence)
  freeVertexMem<<<grid, block>>>(d, nLines);
  ck(cudaGetLastError(), "free2 launch");
  ck(cudaDeviceSynchronize(), "free2 sync");

  // 4) Re-allocate to confirm heap was reclaimed
  alloc_vertices_kernel<<<grid, block>>>(d, nLines, nVerts);
  ck(cudaGetLastError(), "realloc launch");
  ck(cudaDeviceSynchronize(), "realloc sync");

  ck(cudaMemcpy(h.data(), d, nLines*sizeof(BezierLine), cudaMemcpyDeviceToHost), "D2H check realloc");
  int nonnull_after_realloc=0;
  for (int i=0;i<nLines;i++) if (h[i].vertexPos) nonnull_after_realloc++;
  std::printf("Non-null after realloc: %d / %d -> %s\n", nonnull_after_realloc, nLines,
              (nonnull_after_realloc>0 ? "OK" : "FAIL"));

  cudaFree(d);
  bool ok = (null_after1==nLines) && (nonnull_before>0) && (nonnull_after_realloc>0);
  std::printf("Summary: %s\n", ok?"OK":"FAIL");
  return ok?0:1;
}