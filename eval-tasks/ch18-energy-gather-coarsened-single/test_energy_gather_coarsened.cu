// ch18-energy-gather-coarsened-single / test_energy_gather_coarsened.cu
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 256
#endif
#ifndef COARSEN_FACTOR
#define COARSEN_FACTOR 8
#endif

extern "C" __global__
void cenergyCoarsenKernel(float* energygrid, dim3 grid, float gridspacing, float z,
                          int atoms_in_chunk, int start_atom);
extern __constant__ float atoms[CHUNK_SIZE * 4];

static void ck(cudaError_t e, const char* m){
    if(e != cudaSuccess){ std::fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}

static void cpu_oracle_slice(std::vector<float>& out, dim3 grid, float h, float z,
                             const std::vector<float>& a) {
    const int N = (int)a.size()/4;
    out.assign((size_t)grid.x*grid.y*grid.z, 0.0f);
    int k = int(z/h);
    for (int j=0;j<(int)grid.y;++j){
        double y = h*(double)j;
        for (int i=0;i<(int)grid.x;++i){
            double x = h*(double)i, sum=0.0;
            for (int t=0;t<N;++t){
                double dx = x - (double)a[4*t+0];
                double dy = y - (double)a[4*t+1];
                double dz = z - (double)a[4*t+2];
                double den= std::sqrt(dx*dx+dy*dy+dz*dz);
                sum += (double)a[4*t+3] / std::max(den,1e-18);
            }
            size_t idx=(size_t)grid.x*grid.y*k + (size_t)grid.x*j + (size_t)i;
            out[idx]=(float)sum;
        }
    }
}

static bool almost_equal(const std::vector<float>& a,const std::vector<float>& b,
                         float abs_eps=1e-5f, float rel_eps=1e-5f){
    if (a.size()!=b.size()) return false;
    for (size_t i=0;i<a.size();++i){
        float x=a[i], y=b[i], d=std::fabs(x-y);
        if (!(d<=abs_eps || d <= rel_eps*std::max(1.0f,std::max(std::fabs(x),std::fabs(y))))) return false;
    }
    return true;
}

int main(){
    std::printf("ch18-energy-gather-coarsened-single tests\n");

    struct G { dim3 g; const char* name; };
    const G grids[] = {
        {{7,5,3}, "7x5x3"},       // adversarial width (not divisible by COARSEN_FACTOR*blockDim.x)
        {{31,17,4}, "31x17x4"},
        {{64,33,4}, "64x33x4"},
    };
    const int atom_counts[] = {0, 1, 13, CHUNK_SIZE-5, CHUNK_SIZE, CHUNK_SIZE+11};

    const size_t GUARD=1024;
    const float SENT=7.25f;
    float h=1.0f; float z=1.0f;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> pos01(0.0f,1.0f), charge(-1.5f,2.5f);

    int total=0, passed=0;

    for (auto cfg: grids){
        for (int NA: atom_counts){
            // atoms
            std::vector<float> a(4*NA);
            for (int t=0;t<NA;++t){
                a[4*t+0]=pos01(rng)*cfg.g.x*h;
                a[4*t+1]=pos01(rng)*cfg.g.y*h;
                a[4*t+2]=(pos01(rng)*2.0f-0.5f)*cfg.g.z*h;
                a[4*t+3]=charge(rng);
            }

            // reference
            std::vector<float> ref;
            cpu_oracle_slice(ref, cfg.g, h, z, a);

            size_t Nout=(size_t)cfg.g.x*cfg.g.y*cfg.g.z;
            std::vector<float> out_guard(Nout+2*GUARD, SENT);
            float* d_all=nullptr; ck(cudaMalloc(&d_all,(Nout+2*GUARD)*sizeof(float)),"malloc");
            ck(cudaMemcpy(d_all,out_guard.data(),(Nout+2*GUARD)*sizeof(float),cudaMemcpyHostToDevice),"H2D");
            float* d_out=d_all+GUARD;
            ck(cudaMemset(d_out,0,Nout*sizeof(float)),"zero");

            // chunked launches (sequential then shuffled)
            int chunks=(NA+CHUNK_SIZE-1)/CHUNK_SIZE;

            auto run_mode = [&](bool shuffle){
                // reset interior part
                ck(cudaMemset(d_out,0,Nout*sizeof(float)),"zero2");

                std::vector<int> ord(chunks);
                for(int i=0;i<chunks;i++) ord[i]=i;
                if (shuffle) std::shuffle(ord.begin(), ord.end(), rng);

                dim3 block(128, 2);
                int tiles_x = (cfg.g.x + (int)block.x*COARSEN_FACTOR - 1) / ((int)block.x*COARSEN_FACTOR);
                dim3 grid(tiles_x, (cfg.g.y + block.y - 1)/block.y);

                for (int idx=0; idx<chunks; ++idx){
                    int c=(chunks==0?0:ord[idx]);
                    int start=c*CHUNK_SIZE;
                    int count=std::min(CHUNK_SIZE, NA-start);
                    if (count<=0) continue;

                    ck(cudaMemcpyToSymbol(atoms, a.data()+4*start, count*4*sizeof(float), 0, cudaMemcpyHostToDevice), "H2C atoms");
                    cenergyCoarsenKernel<<<grid,block>>>(d_out, cfg.g, h, z, count, start);
                    ck(cudaGetLastError(),"launch");
                    ck(cudaDeviceSynchronize(),"sync");
                }

                std::vector<float> got(Nout);
                ck(cudaMemcpy(out_guard.data(), d_all, (Nout+2*GUARD)*sizeof(float), cudaMemcpyDeviceToHost),"D2H");
                std::copy(out_guard.begin()+GUARD, out_guard.begin()+GUARD+Nout, got.begin());

                bool ok = almost_equal(got, ref);
                std::printf("Grid %-8s NA=%5d order=%s -> %s\n",
                            cfg.name, NA, (shuffle?"shuf":"seq "), ok?"OK":"FAIL");
                ++total; if (ok) ++passed;
            };

            run_mode(false); // sequential chunk order
            run_mode(true);  // shuffled chunk order

            cudaFree(d_all);
        }
    }

    std::printf("Summary: %d / %d passed\n", passed, total);
    return (passed==total)?0:1;
}