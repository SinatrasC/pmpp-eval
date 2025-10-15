// ch14-spmv-ell-single / test_spmv_ell.cu
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cassert>
#include <cmath>

extern "C" __global__
void spmv_ell_kernel(const int*, const float*, const float*, float*, int, int);

static void ck(cudaError_t e, const char* m){
    if(e!=cudaSuccess){ std::fprintf(stderr,"CUDA %s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}
static inline bool feq(float a,float b,float eps=1e-6f){
    float d=std::fabs(a-b); if(d<=eps) return true;
    float ma=std::max(1.f,std::max(std::fabs(a),std::fabs(b))); return d<=eps*ma;
}
static bool vec_close(const std::vector<float>& a,const std::vector<float>& b,float eps=1e-6f){
    if(a.size()!=b.size()) return false; for(size_t i=0;i<a.size();++i) if(!feq(a[i],b[i],eps)) return false; return true;
}

// Build ELL from a synthetic COO-like generator
struct Ell { int m,n,K; std::vector<int> col; std::vector<float> val; };

static Ell gen_ell(int m,int n,int targetK,int pat){
    std::mt19937 rng(2025 + pat*911 + m*7 + n*13 + targetK*17);
    std::uniform_int_distribution<int> cdist(0, (n? n-1:0));
    std::uniform_real_distribution<float> fdist(-1.f,1.f);

    Ell A{m,n,targetK};
    A.col.assign((size_t)m*targetK, -1);
    A.val.assign((size_t)m*targetK, 0.f);

    for(int i=0;i<m;i++){
        int nnzInRow = 0;
        switch (pat%6){
            case 0: nnzInRow = std::min(targetK, (i*3 + 5) % (targetK+1)); break;
            case 1: nnzInRow = std::min(targetK, targetK); break;    // full row
            case 2: nnzInRow = std::min(targetK, (i%2? targetK: 0)); break; // alternating empty/full
            case 3: nnzInRow = std::min(targetK, std::max(0, targetK- (i%3))); break;
            case 4: nnzInRow = std::min(targetK, (i%5)); break;
            default: nnzInRow = std::min(targetK, (int)(targetK/2)); break;
        }
        for(int t=0;t<nnzInRow;t++){
            int idx = i*targetK + t;
            int c = (n? ( (i*37 + t*17) % n ) : 0);
            if (pat%6==5) c = cdist(rng);
            A.col[idx] = c;
            A.val[idx] = fdist(rng);
        }
    }
    if(m==0 || n==0){ std::fill(A.col.begin(), A.col.end(), -1); std::fill(A.val.begin(), A.val.end(), 0.f); }
    return A;
}

static void spmv_ell_cpu(const Ell& A,const std::vector<float>& x,std::vector<float>& y){
    y.assign(A.m,0.f);
    for(int i=0;i<A.m;i++){
        float s=0.f; int base=i*A.K;
        for(int t=0;t<A.K;t++){
            int c=A.col[base+t];
            if(c>=0) s += A.val[base+t]*x[c];
        }
        y[i]=s;
    }
}

int main(){
    printf("ch14-spmv-ell-single tests\n");
    struct Case{int m,n,K; const char* name;};
    const Case cases[] = {
        {0,0,0,"empty"},
        {1,1,0,"1x1 K=0"},
        {1,1,3,"1x1 K=3 pad"},
        {4,4,3,"4x4 K=3"},
        {64,64,8,"64x64 K=8"},
        {77,41,5,"rect K=5"},
        {256,256,12,"dense-ish K=12"},
        {513,257,7,"prime K=7"},
    };
    const dim3 blocks[] = { dim3(128), dim3(256), dim3(512) };

    const size_t GUARD_I=1024; const int SENT_I=0x7f7f7f7f;
    const size_t GUARD_F=1024; const float SENT_F=1337.f;

    int total=0, pass=0;
    for(const auto& cs: cases){
      for(int pat=0; pat<6; ++pat){
        for(auto bdim: blocks){
          ++total;

          Ell A = gen_ell(cs.m, cs.n, cs.K, pat);
          std::vector<float> x(cs.n,0.f); for(int j=0;j<cs.n;j++) x[j]=0.01f*(j+1);
          std::vector<float> y_ref; spmv_ell_cpu(A,x,y_ref);

          int *d_col_all=nullptr; float *d_val_all=nullptr,*d_x_all=nullptr,*d_y_all=nullptr;
          ck(cudaMalloc(&d_col_all, (A.m*A.K + 2*GUARD_I)*sizeof(int)), "malloc col");
          ck(cudaMalloc(&d_val_all, (A.m*A.K + 2*GUARD_F)*sizeof(float)), "malloc val");
          ck(cudaMalloc(&d_x_all,   (A.n      + 2*GUARD_F)*sizeof(float)), "malloc x");
          ck(cudaMalloc(&d_y_all,   (A.m      + 2*GUARD_F)*sizeof(float)), "malloc y");

          std::vector<int>   h_col_guard(A.m*A.K + 2*GUARD_I, SENT_I);
          std::vector<float> h_val_guard(A.m*A.K + 2*GUARD_F, SENT_F);
          std::vector<float> h_x_guard(  A.n      + 2*GUARD_F, SENT_F);
          std::vector<float> h_y_guard(  A.m      + 2*GUARD_F, SENT_F);

          if(A.m*A.K>0){ std::copy(A.col.begin(),A.col.end(), h_col_guard.begin()+GUARD_I);
                         std::copy(A.val.begin(),A.val.end(), h_val_guard.begin()+GUARD_F); }
          if(A.n>0) std::copy(x.begin(),x.end(), h_x_guard.begin()+GUARD_F);
          if(A.m>0) std::fill(h_y_guard.begin()+GUARD_F, h_y_guard.begin()+GUARD_F+A.m, -999.f);

          ck(cudaMemcpy(d_col_all,h_col_guard.data(),h_col_guard.size()*sizeof(int),cudaMemcpyHostToDevice),"H2D col");
          ck(cudaMemcpy(d_val_all,h_val_guard.data(),h_val_guard.size()*sizeof(float),cudaMemcpyHostToDevice),"H2D val");
          ck(cudaMemcpy(d_x_all,  h_x_guard.data(),  h_x_guard.size()*sizeof(float),  cudaMemcpyHostToDevice),"H2D x");
          ck(cudaMemcpy(d_y_all,  h_y_guard.data(),  h_y_guard.size()*sizeof(float),  cudaMemcpyHostToDevice),"H2D y");

          int* d_col = d_col_all + GUARD_I;
          float* d_val = d_val_all + GUARD_F;
          float* d_x = d_x_all + GUARD_F;
          float* d_y = d_y_all + GUARD_F;

          auto cdiv=[](int a,int b){return (a+b-1)/b;};
          int grid_x = std::max(1, cdiv(A.m, (int)bdim.x));
          grid_x = std::min(grid_x, 65535);

          spmv_ell_kernel<<<dim3(grid_x), bdim>>>(d_col, d_val, d_x, d_y, A.m, A.K);
          ck(cudaGetLastError(),"launch"); ck(cudaDeviceSynchronize(),"sync");

          ck(cudaMemcpy(h_y_guard.data(), d_y_all, h_y_guard.size()*sizeof(float), cudaMemcpyDeviceToHost), "D2H y");
          ck(cudaMemcpy(h_col_guard.data(), d_col_all, h_col_guard.size()*sizeof(int), cudaMemcpyDeviceToHost), "D2H col");
          ck(cudaMemcpy(h_val_guard.data(), d_val_all, h_val_guard.size()*sizeof(float), cudaMemcpyDeviceToHost), "D2H val");
          ck(cudaMemcpy(h_x_guard.data(),   d_x_all,   h_x_guard.size()*sizeof(float),   cudaMemcpyDeviceToHost), "D2H x");

          bool ok=true;
          // y equals oracle
          std::vector<float> y_got(A.m,0.f);
          if(A.m>0) std::copy(h_y_guard.begin()+GUARD_F, h_y_guard.begin()+GUARD_F+A.m, y_got.begin());
          ok = ok && vec_close(y_got, y_ref);

          auto guard_ok_f=[&](const std::vector<float>& g){ for(size_t i=0;i<GUARD_F;i++) if(g[i]!=SENT_F || g[g.size()-1-i]!=SENT_F) return false; return true; };
          auto guard_ok_i=[&](const std::vector<int>& g){ for(size_t i=0;i<GUARD_I;i++) if(g[i]!=SENT_I || g[g.size()-1-i]!=SENT_I) return false; return true; };
          ok = ok && guard_ok_f(h_y_guard) && guard_ok_f(h_x_guard) && guard_ok_f(h_val_guard);
          ok = ok && guard_ok_i(h_col_guard);

          // Inputs unchanged (interior)
          if(A.m*A.K>0){
              std::vector<int> col_in(A.m*A.K); std::vector<float> val_in(A.m*A.K);
              std::copy(h_col_guard.begin()+GUARD_I, h_col_guard.begin()+GUARD_I + A.m*A.K, col_in.begin());
              std::copy(h_val_guard.begin()+GUARD_F, h_val_guard.begin()+GUARD_F + A.m*A.K, val_in.begin());
              ok = ok && std::equal(col_in.begin(), col_in.end(), A.col.begin());
              ok = ok && vec_close(val_in, A.val);
          }
          if(A.n>0){
              std::vector<float> x_in(A.n);
              std::copy(h_x_guard.begin()+GUARD_F, h_x_guard.begin()+GUARD_F+A.n, x_in.begin());
              ok = ok && vec_close(x_in, x);
          }

          std::printf("ELL  m=%4d n=%4d K=%2d pat=%d block=%3u grid=%4d -> %s\n",
                      A.m,A.n,A.K,pat,bdim.x,grid_x, ok?"OK":"FAIL");
          if(ok) ++pass;

          cudaFree(d_col_all); cudaFree(d_val_all); cudaFree(d_x_all); cudaFree(d_y_all);
        }
      }
    }
    std::printf("Summary: %d / %d passed\n", pass, total);
    return (pass==total)?0:1;
}