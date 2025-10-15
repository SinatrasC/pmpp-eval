// ch21-bezier-dp-free-child-buffers / reference_solution.cu
#include <cuda_runtime.h>
#include <stdint.h>

#include <vector_types.h>

struct BezierLine {
  float2 CP[3];
  float2* vertexPos;
  int     nVertices;
};

__global__ void freeVertexMem(BezierLine* bLines, int nLines) {
  int lidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (lidx >= nLines) return;
  float2* p = bLines[lidx].vertexPos;
  if (p) {
    free(p);
    bLines[lidx].vertexPos = nullptr;
    bLines[lidx].nVertices = 0;
  }
}