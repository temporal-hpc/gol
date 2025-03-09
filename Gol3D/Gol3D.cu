#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>
#include <curand_kernel.h>

using namespace std;

#include <mimir/mimir.hpp>
#include <mimir/validation.hpp>
using namespace mimir;
using namespace mimir::validation;

#define LIVE make_float4(0.41f, 0.145f, 0.8f, 1.0f)
#define DEAD make_float4(0.0f, 0.0f, 0.0f, 0.0f)

/**
 * @brief Rellena la matriz con 1 y 0 de forma aleatoria dejando un capa de 0 en cada eje.
 * @param matrix matriz a rellenar.
 * @param n dimension de la matriz cubica (n*n*n).
 * @param seed seed para la aleatoriedad.
 * @return La matriz inicializada.
 */
__global__ void fillMatrix(float4 *matrix, int n, int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < n + 2 && y < n + 2 && z < n + 2) {
        int index = z * (n + 2) * (n + 2) + y * (n + 2) + x;
        
        if (x == 0 || y == 0 || z == 0 || x == n + 1 || y == n + 1 || z == n + 1) {
            matrix[index] = DEAD;
        } else {
            curandState state;
            curand_init(seed, index, 0, &state);
            matrix[index] = (curand_uniform(&state) < 0.50) ? DEAD : LIVE;
        }
    }
}
/**
 * @brief Determina si una cadena de texto contiene solo digitos.
 * @param str cadena a analizar.
 * @return true si la cadena contiene solo digitos, false en caso contrario.
 */
bool esNumerico(const string& str) {
    if (str.empty()) return false;
    for (char c : str) {
        if (!std::isdigit(c)) return false;
    }
    return true;
}

/**
 * @brief Calcula el índice de una posición en la matriz para mayor facilidad de operacion.
 * @param index indice a calcular.
 * @param n dimension de la matriz cubica (n*n*n).
 * @return el indice en la matriz.
 */
__device__ int index_d(int x, int y, int z, int n) {
    return (z + 1) * (n + 2) * (n + 2) + (y + 1) * (n + 2) + (x + 1);
}

/**
 * @brief funcion que calcula el siguiente estado de la matriz segun las reglas de Game Of Life y lo deja en la matriz b.
 * @param a estado actual de la matriz.
 * @param b estado siguiente de la matriz.
 * @param n dimension de la matriz cuadrada (n*n*n).
 */
__global__ void GOL(float4 *a, float4 *b, int n) {
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
                        if ((a[neighborIdx].w == LIVE.w)&&( neighborIdx != index )) {
                            cont++;
                        }
                    }
                }
            }
        }

        if (cont == 3) {
            b[index] = LIVE;
        } else if (cont > 3 || cont < 2) {
            b[index] = DEAD;
        } else {
            b[index] = a[index];
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Uso: " << argv[0] << " <tamaño de la matriz (n)> <cantidad de iteraciones (k)>" << endl;
        return 1;
    }
    if (!esNumerico(argv[1]) ) {
        cout << "El tamaño de la matriz debe ser un número entero." << endl;
        return EXIT_FAILURE;
    }
    if (!esNumerico(argv[2]) ) {
        cout << "El número de iteraciones debe ser un número entero." << endl;
        return EXIT_FAILURE;
    }
    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    int seed = rand();
    int size = (n + 2) * (n + 2) * (n + 2) * sizeof(float4);

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((n + 7) / 8, (n + 7) / 8, (n + 7) / 8);

    float4 *d_a, *d_b;
    checkCuda(cudaMalloc((void **)&d_a, size));
    checkCuda(cudaMalloc((void **)&d_b, size));



    MimirEngine engine;
    printf("Game of Life 3D\n");
    engine.init(1920, 1080);
    
    MemoryParams live_params;
    live_params.layout = DataLayout::Layout3D;
    live_params.element_count = {(unsigned)(n+2), (unsigned)(n+2), (unsigned)(n+2)};
    live_params.component_type = ComponentType::Float;
    live_params.channel_count = 4;
    live_params.resource_type = ResourceType::Buffer;

    auto live_buffer = engine.createBuffer((void**)&d_a, live_params);
    fillMatrix<<<gridSize, blockSize>>>(d_a, n, seed);

    ViewParams live_view_params;
    live_view_params.element_count = (n+2) * (n+2) * (n+2);
    live_view_params.extent = {(unsigned)(n+2), (unsigned)(n+2), (unsigned)(n+2)};
    live_view_params.data_domain = DataDomain::Domain3D;
    live_view_params.domain_type = DomainType::Structured;
    live_view_params.view_type = ViewType::Voxels;
    live_view_params.attributes[AttributeType::Color] = *live_buffer;
    live_view_params.options.default_color = LIVE;
    live_view_params.options.default_size = 50;

    engine.createView(live_view_params);

    engine.displayAsync();

    for (int i = 0; i < k; ++i) {
        GOL<<<gridSize, blockSize>>>(d_a, d_b, n);
        checkCuda(cudaMemcpy(d_a, d_b, size, cudaMemcpyDeviceToDevice));

        //separateCells<<<gridSize, blockSize>>>(d_a, d_live_cells, n);
        checkCuda(cudaDeviceSynchronize());

        engine.updateViews();
        this_thread::sleep_for(chrono::milliseconds(100));
    }

    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}

