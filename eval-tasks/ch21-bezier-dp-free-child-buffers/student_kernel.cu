// ch21-bezier-dp-free-child-buffers / student_kernel.cu
#include <cuda_runtime.h>
#include <stdint.h>

#include <vector_types.h>

struct BezierLine {
  float2 CP[3];
  float2* vertexPos; // device-heap pointer
  int     nVertices;
};

// TODO: Implement idempotent free:
//  - lidx = blockIdx.x*blockDim.x + threadIdx.x; if (lidx>=nLines) return;
//  - if (bLines[lidx].vertexPos != nullptr) { free(ptr); bLines[lidx].vertexPos = nullptr; }
//  - (optional) bLines[lidx].nVertices = 0;
__global__ void freeVertexMem(BezierLine* bLines, int nLines) {
  // TODO
}