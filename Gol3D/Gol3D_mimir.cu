#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>
#include <curand_kernel.h>

using namespace std;

#include <mimir/mimir.hpp>
#include <mimir/validation.hpp> // checkCuda
using namespace mimir;
using namespace mimir::validation; // checkCuda

__global__ void fillMatrix(int *matrix, int n, int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < n + 2 && y < n + 2 && z < n + 2) {
        int index = z * (n + 2) * (n + 2) + y * (n + 2) + x;
        
        if (x == 0 || y == 0 || z == 0 || x == n + 1 || y == n + 1 || z == n + 1) {
            matrix[index] = 0;
        } else {
            curandState state;
            curand_init(seed, index, 0, &state);
            matrix[index] = (curand_uniform(&state) < 0.93) ? 0 : 1;
        }
    }
}

__device__ int index_d(int x, int y, int z, int n) {
    return (z + 1) * (n + 2) * (n + 2) + (y + 1) * (n + 2) + (x + 1);
}

__global__ void GOL(int *a, int *b, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < n && y < n && z < n) {
        int index = index_d(x, y, z, n);
        int cont = 0;

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx != 0 || dy != 0 || dz != 0) {
                        int neighborIdx = index_d(x + dx, y + dy, z + dz, n);
                        if (a[neighborIdx] == 1) {
                            cont++;
                        }
                    }
                }
            }
        }

        if (cont == 3) {
            b[index] = 1;
        } else if (cont > 3 || cont < 2) {
            b[index] = 0;
        } else {
            b[index] = a[index];
        }
    }
}

__global__ void separateCells(int *matrix, int *live_cells, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < n && y < n && z < n) {
        int index = x + y * n + z * n * n;
        live_cells[index] = matrix[index_d(x, y, z, n)];
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Uso: " << argv[0] << " <tamaño de la matriz (n)> <cantidad de iteraciones (k)>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    int seed = rand();
    int size = (n + 2) * (n + 2) * (n + 2) * sizeof(int);

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((n + 7) / 8, (n + 7) / 8, (n + 7) / 8);

    int *d_a, *d_b;
    checkCuda(cudaMalloc((void **)&d_a, size));
    checkCuda(cudaMalloc((void **)&d_b, size));

    fillMatrix<<<gridSize, blockSize>>>(d_a, n, seed);

    int *d_live_cells;
    checkCuda(cudaMalloc((void**)&d_live_cells, n * n * n * sizeof(int)));

    MimirEngine engine;
    engine.init(1920, 1080);

    MemoryParams live_params;
    live_params.layout = DataLayout::Layout3D;
    live_params.element_count = {(unsigned)n, (unsigned)n, (unsigned)n};
    live_params.component_type = ComponentType::Int;
    live_params.channel_count = 1;
    live_params.resource_type = ResourceType::Buffer;

    auto live_buffer = engine.createBuffer((void**)&d_live_cells, live_params);

    ViewParams live_view_params;
    live_view_params.element_count = n * n * n;
    live_view_params.extent = {(unsigned)n, (unsigned)n, (unsigned)n};
    live_view_params.data_domain = DataDomain::Domain3D;
    live_view_params.domain_type = DomainType::Structured;
    live_view_params.view_type = ViewType::Voxels;
    live_view_params.attributes[AttributeType::Color] = *live_buffer;
    live_view_params.options.default_color = {0.0f, 1.0f, 0.0f, 1.0f};
    live_view_params.options.default_size = 1;

    engine.createView(live_view_params);

    engine.displayAsync();

    for (int i = 0; i < k; ++i) {
        GOL<<<gridSize, blockSize>>>(d_a, d_b, n);
        checkCuda(cudaMemcpy(d_a, d_b, size, cudaMemcpyDeviceToDevice));

        separateCells<<<gridSize, blockSize>>>(d_a, d_live_cells, n);
        checkCuda(cudaDeviceSynchronize());

        engine.updateViews();
        this_thread::sleep_for(chrono::milliseconds(500));
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_live_cells);

    return 0;
}
