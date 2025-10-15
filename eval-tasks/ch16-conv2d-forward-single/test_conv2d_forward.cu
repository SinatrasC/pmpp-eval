// ch16-conv2d-forward-single / test_conv2d_forward.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <cassert>

extern "C" __global__
void conv2d_forward_kernel(const float* input, const float* weight, const float* bias,
                           float* output,
                           int N, int C, int H, int W,
                           int OC, int KH, int KW,
                           int SH, int SW, int OH, int OW);

static void ck(cudaError_t e, const char* m){ if(e){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2);} }

static inline int out_dim(int I, int K, int S){ return (I - K)/S + 1; }

static void cpu_conv2d(const std::vector<float>& x,  // [N,C,H,W]
                       const std::vector<float>& w,  // [OC,C,KH,KW]
                       const std::vector<float>* b,  // [OC] or nullptr
                       std::vector<float>& y,        // [N,OC,OH,OW]
                       int N,int C,int H,int W,int OC,int KH,int KW,int SH,int SW)
{
    int OH = out_dim(H,KH,SH), OW = out_dim(W,KW,SW);
    y.assign(N*OC*OH*OW, 0.0f);
    auto X = [&](int n,int c,int h,int ww)->float{
        return x[ ((n*C + c)*H + h)*W + ww ];
    };
    auto Wt = [&](int oc,int c,int kh,int kw)->float{
        return w[ (((oc*C + c)*KH + kh)*KW + kw) ];
    };
    auto Y = [&](int n,int oc,int oh,int ow)->float&{
        return y[ ((n*OC + oc)*OH + oh)*OW + ow ];
    };
    for(int n=0;n<N;++n){
      for(int oc=0;oc<OC;++oc){
        for(int oh=0;oh<OH;++oh){
          for(int ow=0;ow<OW;++ow){
            float acc = b ? (*b)[oc] : 0.0f;
            int ih0 = oh*SH, iw0 = ow*SW;
            for(int c=0;c<C;++c){
              for(int kh=0;kh<KH;++kh){
                int ih=ih0+kh; if(ih>=H) continue;
                for(int kw=0;kw<KW;++kw){
                  int iw=iw0+kw; if(iw>=W) continue;
                  acc += X(n,c,ih,iw) * Wt(oc,c,kh,kw);
                }
              }
            }
            Y(n,oc,oh,ow)=acc;
          }
        }
      }
    }
}

static bool almost_equal(const std::vector<float>& a,const std::vector<float>& b,float eps=1e-4f){
    if(a.size()!=b.size()) return false;
    for(size_t i=0;i<a.size();++i){
        float da = std::fabs(a[i]-b[i]);
        float ma = std::max(1.f, std::max(std::fabs(a[i]), std::fabs(b[i])));
        if(!(da<=eps || da<=eps*ma)) return false;
    }
    return true;
}

int main(){
    printf("ch16-conv2d-forward-single tests\n");
    struct Case{int N,C,H,W,OC,KH,KW,SH,SW; const char* name; bool with_bias;};
    const Case cases[] = {
        {1,1,3,3,1,3,3,1,1, "1x1x3x3_K3_S1 no-bias", false},
        {1,1,5,5,1,3,3,2,2, "1x1x5x5_K3_S2 no-bias", false},
        {2,3,7,9,4,3,3,1,1, "N=2 C=3 H=7 W=9 OC=4 K3 S1 bias", true},
        {1,4,8,8,5,5,1,2,2, "C=4 H=8 W=8 OC=5 K5 S2 bias", true},
        {2,2,9,7,3,3,2,1,2, "stride mismatched dims bias", true},
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f,1.f);

    const size_t GUARD = 2048;
    const float SENT = 1337.0f;

    int total=0,pass=0;
    for(const auto& cs : cases){
        ++total;
        int OH = out_dim(cs.H,cs.KH,cs.SH);
        int OW = out_dim(cs.W,cs.KW,cs.SW);
        const int inN  = cs.N*cs.C*cs.H*cs.W;
        const int wN   = cs.OC*cs.C*cs.KH*cs.KW;
        const int outN = cs.N*cs.OC*OH*OW;

        std::vector<float> hX(inN), hW(wN), hB(cs.OC), hY(outN), hYref;
        for(auto& v:hX) v=dist(rng);
        for(auto& v:hW) v=dist(rng);
        if(cs.with_bias){ for(auto& v:hB) v=dist(rng); }

        // CPU oracle
        cpu_conv2d(hX, hW, cs.with_bias? &hB : nullptr, hYref,
                   cs.N, cs.C, cs.H, cs.W, cs.OC, cs.KH, cs.KW, cs.SH, cs.SW);

        // Guarded device buffers
        float *dXall=nullptr, *dWall=nullptr, *dBall=nullptr, *dYall=nullptr;
        ck(cudaMalloc(&dXall, (inN+2*GUARD)*sizeof(float)), "malloc X");
        ck(cudaMalloc(&dWall, (wN +2*GUARD)*sizeof(float)), "malloc W");
        ck(cudaMalloc(&dYall, (outN+2*GUARD)*sizeof(float)), "malloc Y");
        if(cs.with_bias) ck(cudaMalloc(&dBall, (cs.OC+2*GUARD)*sizeof(float)), "malloc B");

        std::vector<float> gX(inN+2*GUARD, SENT), gW(wN+2*GUARD, SENT), gY(outN+2*GUARD, SENT), gB(cs.OC+2*GUARD, SENT);
        std::copy(hX.begin(), hX.end(), gX.begin()+GUARD);
        std::copy(hW.begin(), hW.end(), gW.begin()+GUARD);
        if(cs.with_bias) std::copy(hB.begin(), hB.end(), gB.begin()+GUARD);

        ck(cudaMemcpy(dXall, gX.data(), gX.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D X");
        ck(cudaMemcpy(dWall, gW.data(), gW.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D W");
        ck(cudaMemcpy(dYall, gY.data(), gY.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D Y");
        if(cs.with_bias) ck(cudaMemcpy(dBall, gB.data(), gB.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D B");

        float* dX = dXall + GUARD;
        float* dW = dWall + GUARD;
        float* dY = dYall + GUARD;
        const float* dB = cs.with_bias ? (dBall + GUARD) : nullptr;

        long long totalOut = 1LL*cs.N*cs.OC*OH*OW;
        dim3 block(256);
        dim3 grid( (unsigned)((totalOut + block.x - 1)/block.x) );

        conv2d_forward_kernel<<<grid, block>>>(dX, dW, dB, dY,
            cs.N, cs.C, cs.H, cs.W,
            cs.OC, cs.KH, cs.KW,
            cs.SH, cs.SW, OH, OW);
        ck(cudaGetLastError(),"launch");
        ck(cudaDeviceSynchronize(),"sync");

        ck(cudaMemcpy(gY.data(), dYall, gY.size()*sizeof(float), cudaMemcpyDeviceToHost), "D2H Y");
        ck(cudaMemcpy(gX.data(), dXall, gX.size()*sizeof(float), cudaMemcpyDeviceToHost), "D2H X");
        ck(cudaMemcpy(gW.data(), dWall, gW.size()*sizeof(float), cudaMemcpyDeviceToHost), "D2H W");
        if(cs.with_bias) ck(cudaMemcpy(gB.data(), dBall, gB.size()*sizeof(float), cudaMemcpyDeviceToHost), "D2H B");

        // Extract interior Y
        std::vector<float> hYgot(outN);
        std::copy(gY.begin()+GUARD, gY.begin()+GUARD+outN, hYgot.begin());

        // Validate
        auto guard_ok=[&](const std::vector<float>& v){
            for(size_t i=0;i<GUARD;i++) if(v[i]!=1337.f || v[v.size()-1-i]!=1337.f) return false;
            return true;
        };
        bool ok = almost_equal(hYgot, hYref, 1e-4f)
               && guard_ok(gY)
               && guard_ok(gX) && guard_ok(gW)
               && (!cs.with_bias || guard_ok(gB));

        printf("%-28s -> %s\n", cs.name, ok? "OK":"FAIL");
        if(ok) ++pass;

        cudaFree(dXall); cudaFree(dWall); cudaFree(dYall);
        if(cs.with_bias) cudaFree(dBall);
    }
    printf("Summary: %d / %d passed\n", pass, total);
    return (pass==total)?0:1;
}