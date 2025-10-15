#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

extern "C" void stencil25_single_gpu(const float* d_in, float* d_out,
                                     int dimx, int dimy, int dimz);

static inline size_t idx3(int i,int j,int k,int dx,int dy){
    return (size_t(k)*dy + j)*dx + i;
}

static void ck(cudaError_t e, const char* m){
    if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}

static void fill_pattern(std::vector<float>& a){
    for (size_t i=0;i<a.size();++i){
        // mild variability but deterministic
        a[i] = 0.3f*float(i%97) - 1.7f*float((i*13)%31);
    }
}

static void cpu_oracle(const std::vector<float>& in, std::vector<float>& out,
                       int dimx,int dimy,int dimz)
{
    const int R=4;
    const float w0=0.5f, w1=0.10f, w2=0.05f, w3=0.025f, w4=0.0125f;
    const float w[5]={w0,w1,w2,w3,w4};

    out.resize(in.size());
    for(int k=0;k<dimz;++k){
        for(int j=0;j<dimy;++j){
            for(int i=0;i<dimx;++i){
                size_t p = idx3(i,j,k,dimx,dimy);
                bool interior = (i>=R && i<dimx-R &&
                                 j>=R && j<dimy-R &&
                                 k>=R && k<dimz-R);
                if(!interior){ out[p]=in[p]; continue; }
                float acc = w[0]*in[p];
                for(int d=1; d<=4; ++d){
                    acc += w[d] * ( in[idx3(i-d,j,k,dimx,dimy)]+in[idx3(i+d,j,k,dimx,dimy)]
                                  + in[idx3(i,j-d,k,dimx,dimy)]+in[idx3(i,j+d,k,dimx,dimy)]
                                  + in[idx3(i,j,k-d,dimx,dimy)]+in[idx3(i,j,k+d,dimx,dimy)] );
                }
                out[p]=acc;
            }
        }
    }
}

static bool almost_equal(const std::vector<float>& a, const std::vector<float>& b){
    if(a.size()!=b.size()) return false;
    for(size_t i=0;i<a.size();++i){
        float A=a[i], B=b[i];
        float diff = fabsf(A-B);
        float tol = 1e-5f + 1e-5f*std::max(fabsf(A), fabsf(B));
        if(diff>tol) return false;
    }
    return true;
}

int main(){
    printf("ch20-stencil-25pt-single-gpu-single tests\n");
    struct C{int x,y,z;};
    const C cases[] = {{8,8,8},{17,13,9},{32,16,10},{48,48,40}};

    int total=0, pass=0;

    for(const auto& cs : cases){
        ++total;
        const int dimx=cs.x, dimy=cs.y, dimz=cs.z;
        const size_t N = size_t(dimx)*dimy*dimz;

        // Guarded buffers
        const size_t GUARD=4096;
        const float SENT=1337.0f;

        std::vector<float> hin(N), href, hout(N);
        fill_pattern(hin);
        cpu_oracle(hin, href, dimx,dimy,dimz);

        // host guard
        std::vector<float> h_in_guard(N+2*GUARD, SENT);
        std::copy(hin.begin(),hin.end(),h_in_guard.begin()+GUARD);
        std::vector<float> h_out_guard(N+2*GUARD, SENT);

        float *d_in_all=nullptr, *d_out_all=nullptr;
        ck(cudaMalloc(&d_in_all,(N+2*GUARD)*sizeof(float)),"malloc in");
        ck(cudaMalloc(&d_out_all,(N+2*GUARD)*sizeof(float)),"malloc out");
        ck(cudaMemcpy(d_in_all, h_in_guard.data(), (N+2*GUARD)*sizeof(float), cudaMemcpyHostToDevice), "H2D in");
        ck(cudaMemcpy(d_out_all,h_out_guard.data(),(N+2*GUARD)*sizeof(float), cudaMemcpyHostToDevice), "H2D out");

        float* d_in = d_in_all + GUARD;
        float* d_out= d_out_all + GUARD;

        stencil25_single_gpu(d_in, d_out, dimx,dimy,dimz);

        ck(cudaMemcpy(h_out_guard.data(), d_out_all,(N+2*GUARD)*sizeof(float), cudaMemcpyDeviceToHost), "D2H out");

        std::copy(h_out_guard.begin()+GUARD, h_out_guard.begin()+GUARD+N, hout.begin());

        auto guard_ok=[&](const std::vector<float>& g){
            for(size_t t=0;t<GUARD;t++){
                if(g[t]!=SENT || g[g.size()-1-t]!=SENT) return false;
            } return true;
        };

        bool ok = almost_equal(hout, href) && guard_ok(h_out_guard);
        printf("Case %3dx%3dx%3d -> %s\n", dimx,dimy,dimz, ok?"OK":"FAIL");
        if(ok) ++pass;

        cudaFree(d_in_all); cudaFree(d_out_all);
    }
    printf("Summary: %d/%d passed\n", pass,total);
    return (pass==total)?0:1;
}