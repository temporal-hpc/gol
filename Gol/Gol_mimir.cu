/**
 * @file Gol_mimir.cu
 * @brief CUDA implementation of the Game of Life (GOL) using Mimir for visualization.
 * @author IsaiasCabreraG
 * @date 2024-11
 */

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>
#include <curand_kernel.h>
using namespace std;

// mimir
#include <mimir/mimir.hpp>
#include <mimir/validation.hpp> // checkCuda
using namespace mimir;
using namespace mimir::validation; // checkCuda


/**
 * @brief Kernel to fill a matrix with random values.
 * @param matrix The matrix to fill.
 * @param n The size of the matrix.
 * @param seed The seed for the random number generator.
 */
__global__ void fillMatrix(int *matrix, int n, int seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < (n+2) * (n+2)) {
        int x = index / n;  // Row
        int y = index % n; // Column
        if(x == 0 || y == 0 ||x == n+1 || y == n+1 )
                matrix[index] = 0 ;
            else {
                curandState state;
        
                // Inicializamos el estado del generador con una semilla única por hilo
                curand_init(seed, index, 0, &state);
                matrix[index] = (int)(curand_uniform(&state) * 2.0f);
            }
    }
}

/**
 * @brief Get the index of a matrix element in the host.
 * @param index The index of the element.
 * @param n The size of the matrix.
 * @return The index of the element in the host.
 */
__host__
int index_h(int index, int n) {
    return ((index / n) + 1) * (n + 2) + (index % n + 1);
}

/**
 * @brief Get the index of a matrix element in the device.
 * @param index The index of the element.
 * @param n The size of the matrix.
 * @return The index of the element in the device.
 */
__device__
int index_d(int index, int n) {
    return ((index / n) + 1) * (n + 2) + (index % n + 1);
}

/**
 * @brief Kernel to perform one iteration of the Game of Life.
 * @param a The current state of the matrix.
 * @param b The next state of the matrix.
 * @param n The size of the matrix.
 */
__global__
void GOL(int *a, int *b, int n) {
    int index = index_d(blockIdx.x * blockDim.x + threadIdx.x, n);

    if (index < (n + 1) * (n + 2)) {
        int cont = 0;
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                int index_n = index + (x) + (n + 2) * y;
                if ((index_n != index) && (a[index_n] == 1)) {
                    cont += 1;
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

/**
 * @brief Kernel to separate alive and dead cells.
 * @param matrix The matrix to separate.
 * @param live_cells The buffer to store alive cells.
 * @param n The size of the matrix.
 */
__global__
void separateCells(int *matrix, int *live_cells, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {   
        live_cells[index] = matrix[index_d(index, n)]; 
    }
}

int main(int argc, char *argv[]) {
    int n = 50;
    int k = 100;
    if (argc >= 2) n = std::atoi(argv[1]);
    if (argc >= 3) k = std::atoi(argv[2]);
    int size = (n + 2) * (n + 2) * sizeof(int);
    unsigned block_size = 256;
    unsigned grid_size = (n * n + block_size - 1) / block_size;
    int seed = rand();

    int *d_a, *d_b;
    checkCuda(cudaMalloc((void**)&d_a, size));
    checkCuda(cudaMalloc((void**)&d_b, size));

    fillMatrix<<<grid_size, block_size>>>(d_a, n, seed);
    checkCuda(cudaDeviceSynchronize());

    // Buffers para células vivas y muertas
    int *d_live_cells;
    checkCuda(cudaMalloc((void**)&d_live_cells, n * n * sizeof(int)));

    // Configuración de Mimir para visualizar como "voxels"
    MimirEngine engine;
    engine.init(1920, 1080);

    // Configuración del buffer para células vivas
    MemoryParams live_params;
    live_params.layout = DataLayout::Layout2D;
    live_params.element_count = {(unsigned)(n), (unsigned)(n)};
    live_params.component_type = ComponentType::Int;
    live_params.channel_count = 1;
    live_params.resource_type = ResourceType::Buffer;

    auto live_buffer = engine.createBuffer((void**)&d_live_cells, live_params);

    // Configuración de la vista para células vivas
    ViewParams live_view_params;
    live_view_params.element_count = (n) * (n);
    live_view_params.extent = {(unsigned)(n), (unsigned)(n),1};
    live_view_params.data_domain = DataDomain::Domain2D;
    live_view_params.domain_type = DomainType::Structured;
    live_view_params.view_type = ViewType::Voxels;
    live_view_params.attributes[AttributeType::Color] = *live_buffer;
    live_view_params.options.default_color = {0.0f, 1.0f, 0.0f, 1.0f}; // Verde para células vivas
    live_view_params.options.default_size = 1;

    engine.createView(live_view_params);

    engine.displayAsync();

    checkCuda(cudaDeviceSynchronize());

    for (size_t iter = 0; iter < k; ++iter) {
        // Ejecutar el kernel de Juego de la Vida
        GOL<<<grid_size, block_size>>>(d_a, d_b, n);
        checkCuda(cudaMemcpy(d_a, d_b, size, cudaMemcpyDeviceToDevice));

        // Separar células vivas y muertas
        separateCells<<<grid_size, block_size>>>(d_a, d_live_cells, n);
        checkCuda(cudaDeviceSynchronize());

        // Actualizar la visualización de Mimir
        engine.updateViews();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    engine.showMetrics();
    engine.exit();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_live_cells);

    return EXIT_SUCCESS;
}

