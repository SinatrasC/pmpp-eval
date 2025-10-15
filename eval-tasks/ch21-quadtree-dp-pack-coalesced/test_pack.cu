// ch21-quadtree-dp-pack-coalesced / test_pack.cu
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <random>
#include <cmath>
#include <cassert>

extern "C" __global__
void pack_quadrants_singleblock(const float* x, const float* y,
                                const int* idx_in, int* idx_out,
                                int segBegin, int segCount,
                                float minx, float miny, float maxx, float maxy);

static void ck(cudaError_t e, const char* m){
  if (e != cudaSuccess){ std::fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2); }
}

static inline bool in_NW(float px,float py,float minx,float miny,float maxx,float maxy){
  float mx=0.5f*(minx+maxx), my=0.5f*(miny+maxy);
  return (px < mx) && (py >= my);
}
static inline bool in_NE(float px,float py,float minx,float miny,float maxx,float maxy){
  float mx=0.5f*(minx+maxx), my=0.5f*(miny+maxy);
  return (px >= mx) && (py >= my);
}
static inline bool in_SW(float px,float py,float minx,float miny,float maxx,float maxy){
  float mx=0.5f*(minx+maxx), my=0.5f*(miny+maxy);
  return (px < mx) && (py <  my);
}
static inline bool in_SE(float px,float py,float minx,float miny,float maxx,float maxy){
  float mx=0.5f*(minx+maxx), my=0.5f*(miny+maxy);
  return (px >= mx) && (py <  my);
}

static void cpu_pack(const std::vector<float>& x, const std::vector<float>& y,
                     const std::vector<int>& idx_in,
                     int segBegin, int segCount,
                     float minx, float miny, float maxx, float maxy,
                     std::vector<int>& idx_out)
{
  idx_out = idx_in; // copy full, then rewrite subrange
  std::vector<int> NW, NE, SW, SE;
  NW.reserve(segCount); NE.reserve(segCount); SW.reserve(segCount); SE.reserve(segCount);
  for (int t=0;t<segCount;t++){
    int id = idx_in[segBegin+t];
    float px=x[id], py=y[id];
    if      (in_NW(px,py,minx,miny,maxx,maxy)) NW.push_back(id);
    else if (in_NE(px,py,minx,miny,maxx,maxy)) NE.push_back(id);
    else if (in_SW(px,py,minx,miny,maxx,maxy)) SW.push_back(id);
    else                                       SE.push_back(id);
  }
  int o=segBegin;
  for (int v: NW) idx_out[o++]=v;
  for (int v: NE) idx_out[o++]=v;
  for (int v: SW) idx_out[o++]=v;
  for (int v: SE) idx_out[o++]=v;
}

int main(){
  std::printf("ch21-quadtree-dp-pack-coalesced tests\n");

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> U(0.f,1.f);

  struct Case { int n; int segBegin; int segCount; const char* name; };
  std::vector<Case> cases = {
    {0, 0, 0, "n=0"},
    {8, 0, 8, "small-full"},
    {64, 5, 32, "mid-subrange"},
    {257,40,128, "odd-subrange"},
    {1024,100,256, "checkerboard"},
    {1024,0,1024, "all-one-quadrant"},
  };

  int total=0, pass=0;
  for (auto cs : cases){
    int n = cs.n;
    std::vector<float> hx(n), hy(n);
    for (int i=0;i<n;i++){ hx[i]=U(rng); hy[i]=U(rng); }

    // adversarial patterns
    if (std::string(cs.name)=="checkerboard"){
      for (int i=0;i<n;i++){
        float px = (i&1)? 0.75f : 0.25f;
        float py = (i&2)? 0.75f : 0.25f;
        hx[i]=px; hy[i]=py;
      }
    }
    if (std::string(cs.name)=="all-one-quadrant"){
      for (int i=0;i<n;i++){ hx[i]=0.1f+0.1f*U(rng); hy[i]=0.9f-0.05f*U(rng); } // NW
    }

    std::vector<int> hidx_in(n), cpu_out(n);
    for (int i=0;i<n;i++) hidx_in[i]=i;

    float minx=0.f,miny=0.f,maxx=1.f,maxy=1.f;

    // CPU oracle
    cpu_pack(hx,hy,hidx_in, cs.segBegin, cs.segCount, minx,miny,maxx,maxy, cpu_out);

    // Device
    float *dx=nullptr,*dy=nullptr;
    int *d_in=nullptr,*d_out=nullptr;
    ck(cudaMalloc(&dx,n*sizeof(float)),"dx");
    ck(cudaMalloc(&dy,n*sizeof(float)),"dy");
    ck(cudaMalloc(&d_in,n*sizeof(int)),"d_in");
    ck(cudaMalloc(&d_out,n*sizeof(int)),"d_out");

    ck(cudaMemcpy(dx,hx.data(),n*sizeof(float),cudaMemcpyHostToDevice),"H2D x");
    ck(cudaMemcpy(dy,hy.data(),n*sizeof(float),cudaMemcpyHostToDevice),"H2D y");
    ck(cudaMemcpy(d_in,hidx_in.data(),n*sizeof(int),cudaMemcpyHostToDevice),"H2D idx_in");
    // Initialize idx_out like the CPU oracle does (idx_out = idx_in; then rewrite subrange)
    ck(cudaMemcpy(d_out,hidx_in.data(),n*sizeof(int),cudaMemcpyHostToDevice),"prime out with idx_in");

    dim3 block(256), grid(1);
    size_t shmem = 12 * sizeof(int);
    pack_quadrants_singleblock<<<grid,block,shmem>>>(dx,dy,d_in,d_out,
                                                     cs.segBegin, cs.segCount,
                                                     minx,miny,maxx,maxy);
    ck(cudaGetLastError(),"launch");
    ck(cudaDeviceSynchronize(),"sync");

    std::vector<int> gpu_out(n);
    ck(cudaMemcpy(gpu_out.data(), d_out, n*sizeof(int), cudaMemcpyDeviceToHost),"D2H out");

    bool ok = (gpu_out == cpu_out);
    std::printf("Case %-15s -> %s\n", cs.name, ok?"OK":"FAIL");
    total++; if (ok) pass++;

    cudaFree(dx); cudaFree(dy); cudaFree(d_in); cudaFree(d_out);
  }

  std::printf("Summary: %d/%d passed\n", pass,total);
  return (pass==total)?0:1;
}