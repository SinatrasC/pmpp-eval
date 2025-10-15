// ch18-energy-gather-single / test_energy_gather.cu
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

extern "C" __global__
void cenergyGatherKernel(float* energygrid, dim3 grid, float gridspacing, float z,
                         int atoms_in_chunk, int start_atom);
// Forward declare the constant memory symbol
extern __constant__ float atoms[CHUNK_SIZE * 4];

static void ck(cudaError_t e, const char* m){
    if(e != cudaSuccess){ std::fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}

static void cpu_oracle_slice(std::vector<float>& out, dim3 grid, float gridspacing, float z,
                             const std::vector<float>& h_atoms /* len = 4*N */) {
    const int N = (int)h_atoms.size() / 4;
    out.assign((size_t)grid.x * grid.y * grid.z, 0.0f);

    const int k = int(z / gridspacing);
    for (int j = 0; j < (int)grid.y; ++j) {
        double y = gridspacing * (double)j;
        for (int i = 0; i < (int)grid.x; ++i) {
            double x = gridspacing * (double)i;
            double sum = 0.0;
            for (int a = 0; a < N; ++a) {
                double ax = (double)h_atoms[4*a + 0];
                double ay = (double)h_atoms[4*a + 1];
                double az = (double)h_atoms[4*a + 2];
                double q  = (double)h_atoms[4*a + 3];
                double dx = x - ax, dy = y - ay, dz = z - az;
                double denom = std::sqrt(dx*dx + dy*dy + dz*dz);
                sum += q / std::max(denom, 1e-18);
            }
            size_t idx = (size_t)grid.x * grid.y * k + (size_t)grid.x * j + (size_t)i;
            out[idx] = (float)sum;
        }
    }
}

static bool almost_equal(const std::vector<float>& a, const std::vector<float>& b,
                         float abs_eps = 1e-5f, float rel_eps = 1e-5f){
    if (a.size() != b.size()) return false;
    for (size_t i=0;i<a.size();++i){
        float x=a[i], y=b[i];
        float diff = std::fabs(x-y);
        if (!(diff <= abs_eps || diff <= rel_eps * std::max(1.0f, std::max(std::fabs(x), std::fabs(y))))) {
            return false;
        }
    }
    return true;
}

int main(){
    std::printf("ch18-energy-gather-single tests\n");

    struct GridCfg { dim3 g; float dz; const char* name; };
    const GridCfg grids[] = {
        {{8,8,3}, 1.0f,  "8x8x3"},
        {{31,17,4},1.0f, "31x17x4"},
        {{64,33,4},1.0f, "64x33x4"},
    };
    const int atom_counts[] = {0, 1, 17, CHUNK_SIZE-3, CHUNK_SIZE, CHUNK_SIZE+5};

    const size_t GUARD = 1024;
    const float  SENT  = 2333.0f;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> pos01(0.0f, 1.0f);
    std::uniform_real_distribution<float> charge(-1.0f, 2.0f);

    int total=0, passed=0;

    for (auto cfg : grids){
        float gridspacing = 1.0f;
        float z = gridspacing * cfg.dz;

        for (int NA : atom_counts){
            std::vector<float> h_atoms(4 * NA);
            for (int a=0; a<NA; ++a){
                h_atoms[4*a+0] = pos01(rng) * (cfg.g.x * gridspacing);
                h_atoms[4*a+1] = pos01(rng) * (cfg.g.y * gridspacing);
                h_atoms[4*a+2] = (pos01(rng) * 2.0f - 0.5f) * (cfg.g.z * gridspacing);
                h_atoms[4*a+3] = charge(rng);
            }

            std::vector<float> h_ref;
            cpu_oracle_slice(h_ref, cfg.g, gridspacing, z, h_atoms);

            const size_t Nout = (size_t)cfg.g.x * cfg.g.y * cfg.g.z;
            std::vector<float> h_out_guard(Nout + 2*GUARD, SENT);
            float* d_out_all = nullptr;
            ck(cudaMalloc(&d_out_all, (Nout + 2*GUARD) * sizeof(float)), "malloc out");
            ck(cudaMemcpy(d_out_all, h_out_guard.data(),
                          (Nout + 2*GUARD)*sizeof(float), cudaMemcpyHostToDevice),
               "H2D canary");
            float* d_out = d_out_all + GUARD;

            auto guard_ok = [&](const std::vector<float>& g){
                for (size_t i=0;i<GUARD;i++){
                    if (g[i] != SENT) return false;
                    if (g[g.size()-1-i] != SENT) return false;
                }
                return true;
            };

            // zero interior and accumulate chunks (order shouldn't matter here either)
            ck(cudaMemset(d_out, 0, Nout * sizeof(float)), "memset out");

            int chunks = (NA + CHUNK_SIZE - 1) / CHUNK_SIZE;
            // test both sequential and shuffled orders
            for (int mode=0; mode<2; ++mode){
                // reset interior
                ck(cudaMemset(d_out, 0, Nout * sizeof(float)), "memset out-2");

                std::vector<int> ord(chunks);
                for (int i=0;i<chunks;i++) ord[i]=i;
                if (mode==1) std::shuffle(ord.begin(), ord.end(), rng);

                for (int idx=0; idx<chunks; ++idx){
                    int c = (chunks==0?0:ord[idx]);
                    int start = c * CHUNK_SIZE;
                    int count = std::min(CHUNK_SIZE, NA - start);
                    if (count <= 0) continue;

                    ck(cudaMemcpyToSymbol(atoms, h_atoms.data() + 4*start,
                                          count * 4 * sizeof(float), 0, cudaMemcpyHostToDevice),
                       "H2C atoms");

                    dim3 block(16,16);
                    dim3 grid((cfg.g.x + block.x - 1)/block.x,
                              (cfg.g.y + block.y - 1)/block.y);
                    cenergyGatherKernel<<<grid,block>>>(d_out, cfg.g, gridspacing, z, count, start);
                    ck(cudaGetLastError(), "launch");
                    ck(cudaDeviceSynchronize(), "sync");
                }

                ck(cudaMemcpy(h_out_guard.data(), d_out_all, (Nout + 2*GUARD)*sizeof(float),
                              cudaMemcpyDeviceToHost), "D2H");
                std::vector<float> h_out(Nout);
                std::copy(h_out_guard.begin()+GUARD, h_out_guard.begin()+GUARD+Nout, h_out.begin());

                bool ok = almost_equal(h_out, h_ref) && guard_ok(h_out_guard);
                std::printf("Grid %-8s NA=%5d order=%s -> %s\n",
                            cfg.name, NA, (mode==0?"seq":"shuf"), ok?"OK":"FAIL");
                ++total; if (ok) ++passed;
            }

            cudaFree(d_out_all);
        }
    }
    std::printf("Summary: %d / %d passed\n", passed, total);
    return (passed==total)?0:1;
}