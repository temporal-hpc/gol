#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>
#include <curand_kernel.h>

#include "samplesReader.cpp"
using namespace std;

#include <mimir/mimir.hpp>
#include <mimir/validation.hpp> // checkCuda
using namespace mimir;
using namespace mimir::validation; // checkCuda

#define LIVE make_float4(1.0f, 1.0f, 1.0f, 1.0f)
#define DEAD make_float4(0.0f, 0.0f, 0.0f, 0.01f)

/**
 * @brief Rellena la matriz con LIVE y DEAD de forma aleatoria dejando un capa de DEAD en cada eje.
 * @param matrix matriz a rellenar.
 * @param n dimension de la matriz cuadrada (n*n).
 * @param seed seed para la aleatoriedad.
 */
__global__ void fillMatrix(float4 *matrix, int n, int seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < (n + 2) * (n + 2)) {
        int x = index / (n + 2);
        int y = index % (n + 2);
        if (x == 0 || y == 0 || x == n + 1 || y == n + 1)
            matrix[index] = DEAD;
        else {
            curandState state;
            curand_init(seed, index, 0, &state);
            matrix[index] = (curand_uniform(&state) < 0.5f) ? DEAD : LIVE;
        }
    }
}

/**
 * @brief Convierte una matriz de int (1 y 0) a float4 (LIVE y DEAD).
 * @param int_matrix matriz de enteros (1 y 0) en el host.
 * @param float4_matrix matriz de float4 (LIVE y DEAD) en el device.
 * @param n dimension de la matriz cuadrada (n*n).
 */
__global__ void convertMatrix(int *int_matrix, float4 *float4_matrix, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < (n + 2) * (n + 2)) {
        float4_matrix[index] = (int_matrix[index] == 1) ? LIVE : DEAD;
    }
}

/**
 * @brief Calcula el índice de una posición en la matriz para mayor facilidad de operacion.
 * @param index indice a calcular.
 * @param n dimension de la matriz cuadrada (n*n).
 * @return el indice en la matriz.
 */
__device__ int index_d(int index, int n) {
    return ((index / n) + 1) * (n + 2) + (index % n + 1);
}

/**
 * @brief funcion que calcula el siguiente estado de la matriz segun las reglas de Game Of Life y lo deja en la matriz b.
 * @param a estado actual de la matriz.
 * @param b estado siguiente de la matriz.
 * @param n dimension de la matriz cuadrada (n*n).
 */
__global__ void GOL(float4 *a, float4 *b, int n) {
    int index = index_d(blockIdx.x * blockDim.x + threadIdx.x, n);
    if (index < (n + 2) * (n + 2)) {
        int cont = 0;
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                int index_n = index + x + (n + 2) * y;
                if ((index_n != index) && (a[index_n].w == LIVE.w)) {
                    cont += 1;
                }
            }
        }
        if (cont == 3)
            b[index] = LIVE;
        else if (cont > 3 || cont < 2)
            b[index] = DEAD;
        else
            b[index] = a[index];
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "Uso: " << argv[0] << " <modo> [parametros...]" << endl;
        cout << "Modo 'r': " << argv[0] << " r <tamano_matriz> <num_iteraciones>" << endl;
        cout << "Modo 's': " << argv[0] << " s <archivo.txt>" << endl;
        return EXIT_FAILURE;
    }

    // Inicializar variables
    char mode = argv[1][0];
    int n;
    int k;
    int size;
    int *a = nullptr;

    if (mode == 'r') {
        if (argc < 4) {
            cout << "Uso: " << argv[0] << " r <tamano_matriz> <num_iteraciones>" << endl;
            return EXIT_FAILURE;
        }
        if (!esNumerico(argv[2])) {
            cout << "El tamaño de la matriz debe ser un número entero." << endl;
            return EXIT_FAILURE;
        }
        if (!esNumerico(argv[3])) {
            cout << "El número de iteraciones debe ser un número entero." << endl;
            return EXIT_FAILURE;
        }
        n = atoi(argv[2]);
        k = atoi(argv[3]);
        size = (n + 2) * (n + 2) * sizeof(int);
    } 
    else if (mode == 's') {
        if (argc < 3) {
            cout << "Uso: " << argv[0] << " s <archivo.txt>" << endl;
            return EXIT_FAILURE;
        }
        string filename = argv[2];
        filename = "../Gol2D_mimir/samples/" + filename;

        if (!fileExists(filename)) {
            cout << "El archivo " << filename << " no existe." << endl;
            return EXIT_FAILURE;
        }
        n = nInput(filename);
        if (n == -1) {
            cout << "El archivo " << filename << " no contiene el tamaño de la matriz." << endl;
            return EXIT_FAILURE;
        }
        cout << "Matriz de tamaño " << n << "x" << n << endl;
        size = (n + 2) * (n + 2) * sizeof(int);

        k = kInput(filename);
        if (k == -1) {
            cout << "El archivo " << filename << " no contiene el número de iteraciones." << endl;
            return EXIT_FAILURE;
        }

        // Lee la matriz desde un archivo
        a = matrix2DInput(filename, n);
        if (a == nullptr) {
            cout << "El archivo " << filename << " no contiene la matriz." << endl;
            return EXIT_FAILURE;
        }

        cout << "Archivo seleccionado: " << filename << endl;
    } 
    else {
        cout << "Modo desconocido. Usa 'r' o 's'." << endl;
        return EXIT_FAILURE;
    }

    // Inicializar matrices en GPU
    unsigned block_size = 256;
    unsigned grid_size = ((n + 2) * (n + 2) + block_size - 1) / block_size;

    float4 *d_a, *d_b;
    checkCuda(cudaMalloc(&d_a, (n + 2) * (n + 2) * sizeof(float4)));
    checkCuda(cudaMalloc(&d_b, (n + 2) * (n + 2) * sizeof(float4)));


    checkCuda(cudaDeviceSynchronize());

    // Inicializar variables para la representación gráfica a través de Mimir
    MimirEngine engine;
    engine.init(1920, 1080);

    MemoryParams live_params;
    live_params.layout = DataLayout::Layout2D;
    live_params.element_count = {(unsigned)(n + 2), (unsigned)(n + 2)};
    live_params.component_type = ComponentType::Float;
    live_params.channel_count = 4;
    live_params.resource_type = ResourceType::Buffer;

    // Crear el buffer con d_a
    auto live_buffer = engine.createBuffer((void**)&d_a, live_params);
    if (mode == 'r') {
        int seed = rand();
        // Inicializa la matriz con valores aleatorios en GPU
        fillMatrix<<<grid_size, block_size>>>(d_a, n, seed);
    } 
    else if (mode == 's') {
        // Copia la matriz de enteros al device
        int *d_int_matrix;
        checkCuda(cudaMalloc(&d_int_matrix, size));
        checkCuda(cudaMemcpy(d_int_matrix, a, size, cudaMemcpyHostToDevice));

        // Convierte la matriz de enteros a float4 en el device
        convertMatrix<<<grid_size, block_size>>>(d_int_matrix, d_a, n);

        // Libera la memoria de la matriz de enteros en la GPU
        cudaFree(d_int_matrix);
    }

    ViewParams live_view_params;
    live_view_params.element_count = (n + 2) * (n + 2);
    live_view_params.extent = {(unsigned)(n + 2), (unsigned)(n + 2), 1};
    live_view_params.data_domain = DataDomain::Domain2D;
    live_view_params.domain_type = DomainType::Structured;
    live_view_params.view_type = ViewType::Voxels;
    live_view_params.attributes[AttributeType::Color] = *live_buffer;
    live_view_params.options.default_color = {0.0f, 1.0f, 0.0f, 1.0f};
    live_view_params.options.default_size = 1;

    // Crear la vista con Mimir
    engine.createView(live_view_params);
    engine.displayAsync();
    checkCuda(cudaDeviceSynchronize());
    engine.updateViews();

    // Iterar la matriz
    for (size_t iter = 0; iter < k; ++iter) {
        // Dormir el sistema para comprensión
        this_thread::sleep_for(chrono::milliseconds(100));
        GOL<<<grid_size, block_size>>>(d_a, d_b, n);
        // Copiar la memoria de d_b a d_a
        checkCuda(cudaMemcpy(d_a, d_b, (n + 2) * (n + 2) * sizeof(float4), cudaMemcpyDeviceToDevice));

        // Actualizar las vistas
        engine.updateViews();
    }

    engine.exit();

    cudaFree(d_a);
    cudaFree(d_b);

    if (a != nullptr) {
        delete[] a;
    }

    return EXIT_SUCCESS;
}
