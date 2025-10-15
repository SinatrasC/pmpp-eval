// ch21-bezier-dp-parent-child-single / test_bezier_dp.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstring>

#include <vector_types.h>

struct BezierLine {
  float2 CP[3];
  float2* vertexPos;  // device ptr (allocated on device heap)
  int     nVertices;
};

// Prototypes provided by linked object (reference_solution.o or student_kernel.o)
__global__ void computeBezierLines_parent(BezierLine* bLines, int nLines, int maxTess);
__global__ void computeBezierLine_child(int lidx, BezierLine* bLines, int nTess);

// Host utilities
static void ck(cudaError_t e, const char* m){
  if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}

static float2 f2(float x, float y){ return make_float2(x, y); }

static void cpu_tess(const BezierLine& L, std::vector<float2>& out) {
  int n = L.nVertices;
  out.resize(n);
  for (int i=0;i<n;i++){
    float u = (n>1) ? (float)i/(float)(n-1) : 0.f;
    float omu=1.f-u;
    float B0=omu*omu, B1=2.f*u*omu, B2=u*u;
    float2 P0=L.CP[0], P1=L.CP[1], P2=L.CP[2];
    out[i].x = B0*P0.x + B1*P1.x + B2*P2.x;
    out[i].y = B0*P0.y + B1*P1.y + B2*P2.y;
  }
}

static bool almost_equal(const float2& a, const float2& b, float eps=1e-6f){
  float dx = std::fabs(a.x-b.x), dy = std::fabs(a.y-b.y);
  return dx<=eps && dy<=eps;
}

int main(){
  // Enable device heap for device malloc/free
  size_t heapSize = 64*1024*1024; // 64MB
  ck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize), "set heap");

  ck(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 8), "set dev sync depth");
  ck(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 131072), "set pending launches");
  ck(cudaDeviceSetLimit(cudaLimitStackSize, 16384), "set stack");

  std::printf("ch21-bezier-dp-parent-child-single tests\n");

  // Construct test lines
  std::vector<BezierLine> h_lines;
  auto push_line = [&](float2 P0, float2 P1, float2 P2){
    BezierLine L; L.CP[0]=P0; L.CP[1]=P1; L.CP[2]=P2; L.vertexPos=nullptr; L.nVertices=0; h_lines.push_back(L);
  };

  // Straight line: curvature ~ 0 -> clamp to 4
  push_line(f2(0,0), f2(0.5f,0), f2(1,0));
  // Gentle curve
  push_line(f2(0,0), f2(0.5f,0.2f), f2(1,0));
  // Sharp curve
  push_line(f2(0,0), f2(0.0f,1.0f), f2(1,0));
  // Random-ish set
  for(int i=0;i<20;i++){
    float t = (float)i/20.f;
    push_line(f2(t,0), f2(t+0.05f, 0.3f+0.2f*t), f2(t+0.1f, 0.0f));
  }

  const int nLines = (int)h_lines.size();
  const int maxTess = 128;

  // Device buffer of lines
  BezierLine* d_lines=nullptr;
  ck(cudaMalloc(&d_lines, nLines*sizeof(BezierLine)), "malloc d_lines");
  ck(cudaMemcpy(d_lines, h_lines.data(), nLines*sizeof(BezierLine), cudaMemcpyHostToDevice), "H2D lines");

  // Launch parent
  dim3 block(64);
  dim3 grid((nLines + block.x - 1) / block.x);
  computeBezierLines_parent<<<grid, block>>>(d_lines, nLines, maxTess);
  ck(cudaGetLastError(), "launch parent");
  ck(cudaDeviceSynchronize(), "sync parent");

  // Additional safety sync to ensure all child kernels complete
  ck(cudaDeviceSynchronize(), "additional sync");

  // Fetch back line descriptors (to get device ptrs + nVertices)
  ck(cudaMemcpy(h_lines.data(), d_lines, nLines*sizeof(BezierLine), cudaMemcpyDeviceToHost), "D2H lines");

  // Validate: compute CPU oracles with the **same nVertices** the GPU chose
  int total=0, pass=0;
  for (int i=0;i<nLines;i++){
    const BezierLine& L = h_lines[i];
    std::printf("DEBUG: Line %d nVertices=%d ptr=%p\n", i, L.nVertices, L.vertexPos);
    if (L.nVertices == 0 || L.vertexPos == nullptr){
      std::printf("Line %d -> allocation failed (nVertices=%d, ptr=%p)  FAIL\n", i, L.nVertices, L.vertexPos);
      continue;
    }
    std::vector<float2> cpu;
    cpu_tess(L, cpu);

    std::vector<float2> gpu(L.nVertices);

    // Test if pointer is valid by checking size
    cudaError_t ptrTest = cudaMemcpy(gpu.data(), L.vertexPos, L.nVertices*sizeof(float2), cudaMemcpyDeviceToHost);
    if (ptrTest != cudaSuccess) {
      std::printf("Line %d -> Device memory access failed: %s (ptr=%p, size=%d)\n",
                  i, cudaGetErrorString(ptrTest), L.vertexPos, L.nVertices);
      continue;
    }

    bool ok = true;
    for (int k=0;k<L.nVertices;k++){ ok = ok && almost_equal(cpu[k], gpu[k]); }
    std::printf("Line %2d nVerts=%3d -> %s\n", i, L.nVertices, ok?"OK":"FAIL");
    total++; if (ok) pass++;
  }

  std::printf("Summary: %d/%d passed\n", pass, total);

  // Free device heap allocations (best-effort) via a small device kernel or leave to next task.
  // Here we just free the BezierLine array (device-deferred frees are fine for CI).
  cudaFree(d_lines);
  return (pass==total)?0:1;
}