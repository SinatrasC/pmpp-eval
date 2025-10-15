// ch21-quadtree-dp-build-single / test_quadtree_dp.cu
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <random>
#include <cassert>
#include <cmath>

struct Bounds { float minx, miny, maxx, maxy; };
__host__ __device__ inline Bounds make_bounds(float a,float b,float c,float d){ return Bounds{a,b,c,d}; }

__global__ void quadtree_build_parent(const float* x, const float* y, int n,
                                      Bounds root, int max_depth, int min_points,
                                      int* perm, int* leafOffset, int* leafCount,
                                      int* leafCounter, int* permCursor,
                                      const int* idx_root);

static void ck(cudaError_t e, const char* m){
  if (e != cudaSuccess){ std::fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2); }
}

static bool is_perm(const std::vector<int>& p){
  int n = (int)p.size();
  std::vector<int> v=p;
  std::sort(v.begin(), v.end());
  for (int i=0;i<n;i++) if (v[i]!=i) return false;
  return true;
}

// CPU reference with same traversal rules
static inline bool in_NW(float px,float py,const Bounds& b){
  float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy);
  return (px < mx) && (py >= my);
}
static inline bool in_NE(float px,float py,const Bounds& b){
  float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy);
  return (px >= mx) && (py >= my);
}
static inline bool in_SW(float px,float py,const Bounds& b){
  float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy);
  return (px < mx) && (py <  my);
}
static inline bool in_SE(float px,float py,const Bounds& b){
  float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy);
  return (px >= mx) && (py <  my);
}
static inline Bounds bNW(const Bounds& b){ float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy); return make_bounds(b.minx, my, mx, b.maxy); }
static inline Bounds bNE(const Bounds& b){ float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy); return make_bounds(mx, my, b.maxx, b.maxy); }
static inline Bounds bSW(const Bounds& b){ float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy); return make_bounds(b.minx, b.miny, mx, my); }
static inline Bounds bSE(const Bounds& b){ float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy); return make_bounds(mx, b.miny, b.maxx, my); }

static void cpu_build(const std::vector<float>& x, const std::vector<float>& y,
                      const std::vector<int>& idx, int begin, int count,
                      const Bounds& b, int depth, int max_depth, int min_points,
                      std::vector<int>& out_perm)
{
  if (depth>=max_depth || count<=min_points){
    for (int i=0;i<count;i++) out_perm.push_back(idx[begin+i]);
    return;
  }
  // Count
  int cNW=0,cNE=0,cSW=0,cSE=0;
  for (int i=0;i<count;i++){
    int id=idx[begin+i]; float px=x[id], py=y[id];
    if      (in_NW(px,py,b)) cNW++;
    else if (in_NE(px,py,b)) cNE++;
    else if (in_SW(px,py,b)) cSW++;
    else                     cSE++;
  }
  int sNW=0, sNE=cNW, sSW=cNW+cNE, sSE=cNW+cNE+cSW;
  std::vector<int> local(count);
  int pNW=0,pNE=0,pSW=0,pSE=0;
  for (int i=0;i<count;i++){
    int id=idx[begin+i]; float px=x[id], py=y[id];
    if      (in_NW(px,py,b)) local[sNW+(pNW++)]=id;
    else if (in_NE(px,py,b)) local[sNE+(pNE++)]=id;
    else if (in_SW(px,py,b)) local[sSW+(pSW++)]=id;
    else                     local[sSE+(pSE++)]=id;
  }
  int off=0;
  if (cNW) cpu_build(x,y,local,0,cNW,bNW(b), depth+1, max_depth, min_points, out_perm);
  off += cNW;
  if (cNE) cpu_build(x,y,local,off,cNE,bNE(b), depth+1, max_depth, min_points, out_perm);
  off += cNE;
  if (cSW) cpu_build(x,y,local,off,cSW,bSW(b), depth+1, max_depth, min_points, out_perm);
  off += cSW;
  if (cSE) cpu_build(x,y,local,off,cSE,bSE(b), depth+1, max_depth, min_points, out_perm);
}

int main(){
  std::printf("ch21-quadtree-dp-build-single tests\n");
  ck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024), "heap");

  ck(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 16), "set dev sync depth");
  ck(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 262144), "set pending launches");
  ck(cudaDeviceSetLimit(cudaLimitStackSize, 16384), "set stack");

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> U(0.0f, 1.0f);

  struct Case{ int n; int max_depth; int min_points; const char* name; };
  std::vector<Case> cases = {
    { 0,  8, 8, "n=0" },
    { 1,  8, 8, "n=1" },
    { 32, 8, 4, "uniform-32" },
    { 128,8, 8, "random-128" },
    { 512,8, 8, "two-clusters-512" },
  };

  int total=0, pass=0;
  for (const auto& cs : cases){
    int n=cs.n;
    std::vector<float> hx(n), hy(n);
    if (std::string(cs.name).find("two-clusters")==0){
      for (int i=0;i<n;i++){
        float cx = (i%2==0)? 0.25f : 0.75f;
        float cy = (i%2==0)? 0.25f : 0.75f;
        hx[i] = cx + 0.02f*U(rng);
        hy[i] = cy + 0.02f*U(rng);
      }
    } else {
      for (int i=0;i<n;i++){ hx[i]=U(rng); hy[i]=U(rng); }
    }
    // uniform grid case
    if (std::string(cs.name).find("uniform")==0){
      int s=(int)std::ceil(std::sqrt((double)std::max(1, n)));
      int k=0;
      for(int j=0;j<s && k<n;j++) for(int i=0;i<s && k<n;i++){
        hx[k]=(i+0.5f)/s; hy[k]=(j+0.5f)/s; k++;
      }
    }

    std::vector<int> idx(n);
    for (int i=0;i<n;i++) idx[i]=i;

    // CPU ref
    std::vector<int> cpu_perm; cpu_perm.reserve(n);
    Bounds root = make_bounds(0.f,0.f,1.f,1.f);
    cpu_build(hx,hy,idx,0,n,root,0,cs.max_depth, cs.min_points, cpu_perm);
    if ((int)cpu_perm.size()!=n){ std::printf("%s CPU perm size mismatch\n", cs.name); return 1; }

    // Device
    float *dx=nullptr,*dy=nullptr; int *dperm=nullptr,*didx=nullptr;
    int *leafOffset=nullptr,*leafCount=nullptr,*leafCounter=nullptr,*permCursor=nullptr;
    ck(cudaMalloc(&dx,n*sizeof(float)), "dx");
    ck(cudaMalloc(&dy,n*sizeof(float)), "dy");
    ck(cudaMalloc(&didx,n*sizeof(int)), "didx");
    ck(cudaMalloc(&dperm,n*sizeof(int)), "dperm");
    ck(cudaMalloc(&leafOffset,(std::max(1,n)+1)*sizeof(int)), "leafOffset");
    ck(cudaMalloc(&leafCount,std::max(1,n)*sizeof(int)), "leafCount");
    ck(cudaMalloc(&leafCounter,sizeof(int)), "leafCounter");
    ck(cudaMalloc(&permCursor,sizeof(int)), "permCursor");

    ck(cudaMemcpy(dx,hx.data(),n*sizeof(float),cudaMemcpyHostToDevice),"H2D x");
    ck(cudaMemcpy(dy,hy.data(),n*sizeof(float),cudaMemcpyHostToDevice),"H2D y");
    ck(cudaMemcpy(didx,idx.data(),n*sizeof(int),cudaMemcpyHostToDevice),"H2D idx");
    ck(cudaMemset(dperm,0,n*sizeof(int)),"clr perm");
    ck(cudaMemset(leafOffset,0,(std::max(1,n)+1)*sizeof(int)),"clr off");
    ck(cudaMemset(leafCount,0,std::max(1,n)*sizeof(int)),"clr cnt");
    ck(cudaMemset(leafCounter,0,sizeof(int)),"clr lc");
    ck(cudaMemset(permCursor,0,sizeof(int)),"clr pc");

    quadtree_build_parent<<<1,1>>>(dx,dy,n,root,cs.max_depth,cs.min_points,
                                   dperm,leafOffset,leafCount,leafCounter,permCursor,didx);
    ck(cudaGetLastError(),"launch parent");
    ck(cudaDeviceSynchronize(),"sync parent");

    // Copy back
    std::vector<int> gpu_perm(n);
    ck(cudaMemcpy(gpu_perm.data(), dperm, n*sizeof(int), cudaMemcpyDeviceToHost), "D2H perm");
    // exact match with CPU ref
    bool ok = (gpu_perm == cpu_perm) && is_perm(gpu_perm);
    std::printf("Case %-18s -> %s\n", cs.name, ok?"OK":"FAIL");
    total++; if (ok) pass++;

    cudaFree(dx); cudaFree(dy); cudaFree(didx); cudaFree(dperm);
    cudaFree(leafOffset); cudaFree(leafCount); cudaFree(leafCounter); cudaFree(permCursor);
  }

  std::printf("Summary: %d/%d passed\n", pass,total);
  return (pass==total)?0:1;
}