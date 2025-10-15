// ch16-maxpool2d-forward-single / test_maxpool_forward.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <cassert>
#include <float.h>

extern "C" __global__
void maxpool2d_forward_kernel(const float* input, float* output, int* indices,
                              int N, int C, int H, int W,
                              int KH, int KW, int SH, int SW,
                              int OH, int OW);

static void ck(cudaError_t e, const char* m){ if(e){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2);} }
static inline int out_dim(int I, int K, int S){ return (I - K)/S + 1; }

static void cpu_maxpool(const std::vector<float>& x, std::vector<float>& y,
                        std::vector<int>& idx,
                        int N,int C,int H,int W,int KH,int KW,int SH,int SW){
    int OH=out_dim(H,KH,SH), OW=out_dim(W,KW,SW);
    y.assign(N*C*OH*OW, -INFINITY);
    idx.assign(N*C*OH*OW, -1);
    auto X=[&](int n,int c,int h,int w){return x[ ((n*C + c)*H + h)*W + w ];};
    auto Y=[&](int n,int c,int oh,int ow)->float&{return y[((n*C + c)*OH + oh)*OW + ow];};
    auto I=[&](int n,int c,int oh,int ow)->int&{return idx[((n*C + c)*OH + oh)*OW + ow];};

    for(int n=0;n<N;++n)for(int c=0;c<C;++c)
    for(int oh=0;oh<OH;++oh)for(int ow=0;ow<OW;++ow){
        float best=-INFINITY; int besti=-1;
        int ih0=oh*SH, iw0=ow*SW;
        for(int kh=0;kh<KH;++kh){
            int ih=ih0+kh; if(ih>=H) continue;
            for(int kw=0;kw<KW;++kw){
                int iw=iw0+kw; if(iw>=W) continue;
                float v=X(n,c,ih,iw); int lid=kh*KW+kw;
                if(v>best){best=v;besti=lid;}
            }
        }
        Y(n,c,oh,ow)=best; I(n,c,oh,ow)=besti;
    }
}

static bool almost_equal(const std::vector<float>& a,const std::vector<float>& b,float eps=1e-6f){
    if(a.size()!=b.size()) return false;
    for(size_t i=0;i<a.size();++i){
        float da = std::fabs(a[i]-b[i]);
        float ma = std::max(1.f, std::max(std::fabs(a[i]), std::fabs(b[i])));
        if(!(da<=eps || da<=eps*ma)) return false;
    }
    return true;
}

int main(){
    printf("ch16-maxpool2d-forward-single tests\n");
    struct Case{int N,C,H,W,KH,KW,SH,SW; const char* name;};
    const Case cases[] = {
        {1,1,4,4,2,2,2,2, "1x1x4x4_K2_S2"},
        {2,3,7,9,2,3,2,3, "N=2 C=3 mixed strides"},
        {1,4,8,8,3,3,1,1, "3x3 S1 over 8x8"},
        {2,2,9,7,2,2,2,2, "even K2 S2 mismatched dims"},
    };

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-5.f,5.f);

    const size_t GUARD=2048; const float SENT=1337.0f;

    int total=0, pass=0;
    for(const auto& cs: cases){
        ++total;
        const int OH=out_dim(cs.H,cs.KH,cs.SH), OW=out_dim(cs.W,cs.KW,cs.SW);
        const int inN = cs.N*cs.C*cs.H*cs.W;
        const int outN= cs.N*cs.C*OH*OW;

        std::vector<float> hX(inN), hYref, hY(outN);
        std::vector<int>   hIref, hI(outN);
        for(auto& v:hX) v=dist(rng);
        cpu_maxpool(hX,hYref,hIref, cs.N,cs.C,cs.H,cs.W, cs.KH,cs.KW,cs.SH,cs.SW);

        float *dXall=nullptr, *dYall=nullptr; int *dIall=nullptr;
        ck(cudaMalloc(&dXall,(inN+2*GUARD)*sizeof(float)),"malloc X");
        ck(cudaMalloc(&dYall,(outN+2*GUARD)*sizeof(float)),"malloc Y");
        ck(cudaMalloc(&dIall,(outN+2*GUARD)*sizeof(int)),"malloc I");

        std::vector<float> gX(inN+2*GUARD,SENT), gY(outN+2*GUARD,SENT);
        std::vector<int>   gI(outN+2*GUARD, (int)0x7f7f7f7f);
        std::copy(hX.begin(), hX.end(), gX.begin()+GUARD);

        ck(cudaMemcpy(dXall,gX.data(),gX.size()*sizeof(float),cudaMemcpyHostToDevice),"H2D X");
        ck(cudaMemcpy(dYall,gY.data(),gY.size()*sizeof(float),cudaMemcpyHostToDevice),"H2D Y");
        ck(cudaMemcpy(dIall,gI.data(),gI.size()*sizeof(int),  cudaMemcpyHostToDevice),"H2D I");

        float* dX=dXall+GUARD; float* dY=dYall+GUARD; int* dI=dIall+GUARD;

        long long totalOut=1LL*cs.N*cs.C*OH*OW;
        dim3 block(256), grid((unsigned)((totalOut+block.x-1)/block.x));
        maxpool2d_forward_kernel<<<grid,block>>>(
            dX,dY,dI, cs.N,cs.C,cs.H,cs.W, cs.KH,cs.KW,cs.SH,cs.SW, OH,OW);
        ck(cudaGetLastError(),"launch");
        ck(cudaDeviceSynchronize(),"sync");

        ck(cudaMemcpy(gY.data(), dYall, gY.size()*sizeof(float), cudaMemcpyDeviceToHost),"D2H Y");
        ck(cudaMemcpy(gI.data(), dIall, gI.size()*sizeof(int),   cudaMemcpyDeviceToHost),"D2H I");
        ck(cudaMemcpy(gX.data(), dXall, gX.size()*sizeof(float), cudaMemcpyDeviceToHost),"D2H X");

        std::copy(gY.begin()+GUARD, gY.begin()+GUARD+outN, hY.begin());
        std::copy(gI.begin()+GUARD, gI.begin()+GUARD+outN, hI.begin());

        auto guard_ok=[&](const auto& vec){
            for(size_t i=0;i<GUARD;i++){
                if constexpr (std::is_same_v<typename std::decay<decltype(vec[0])>::type, float>){
                    if(vec[i]!=SENT || vec[vec.size()-1-i]!=SENT) return false;
                }else{
                    if(vec[i]!=(int)0x7f7f7f7f || vec[vec.size()-1-i]!=(int)0x7f7f7f7f) return false;
                }
            } return true;
        };

        bool ok = almost_equal(hY,hYref,1e-6f)
               && (hI==hIref)
               && guard_ok(gY) && guard_ok(gI) && guard_ok(gX);

        printf("%-28s -> %s\n", cs.name, ok? "OK":"FAIL");
        if(ok) ++pass;

        cudaFree(dXall); cudaFree(dYall); cudaFree(dIall);
    }
    printf("Summary: %d / %d passed\n", pass, total);
    return (pass==total)?0:1;
}