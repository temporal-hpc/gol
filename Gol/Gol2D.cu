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

#include "samplesReader.cpp"
using namespace std;

#include <mimir/mimir.hpp>
#include <mimir/validation.hpp> // checkCuda
using namespace mimir;
using namespace mimir::validation; // checkCuda

#define LIVE make_float4(1.0f, 1.0f, 1.0f, 1.0f)
#define DEAD make_float4(0.0f, 0.0f, 0.0f, 0.01f)

/**
 * @brief Rellena la matriz con 1 y 0 de forma aleatoria dejando un capa de 0 en cada eje.
 * @param matrix matriz a rellenar.
 * @param n dimension de la matriz cuadrada (n*n).
 * @param seed seed para la aleatoriedad.
 * @return La matriz inicializada.
 */
__global__ void fillMatrix(int *matrix, int n, int seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < (n+2) * (n+2)) {
        int x = index / n;
        int y = index % n;
        if (x == 0 || y == 0 || x == n+1 || y == n+1)
            matrix[index] = 0;
        else {
            curandState state;
            curand_init(seed, index, 0, &state);
            matrix[index] = (int)(curand_uniform(&state) * 2.0f);
        }
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
__global__ void GOL(int *a, int *b, int n) {
    int index = index_d(blockIdx.x * blockDim.x + threadIdx.x, n);
    if (index < (n + 1) * (n + 2)) {
        int cont = 0;
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                int index_n = index + x + (n + 2) * y;
                if ((index_n != index) && (a[index_n] == 1)) {
                    cont += 1;
                }
            }
        }
        if (cont == 3)
            b[index] = 1;
        else if (cont > 3 || cont < 2)
            b[index] = 0;
        else
            b[index] = a[index];
    }
}

/**
 * @brief funcion aplica los colores a las celdas vivas y muertas para su representacion con mimir.
 * @param matrix estado actual de la matriz.
 * @param live_cells matriz con colores y transaparencias aplicadas.
 * @param n dimension de la matriz cuadrada (n*n).
 */
__global__ void separateCells(int *matrix, float4 *live_cells, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        live_cells[index] = (matrix[index_d(index, n)] == 1) ? LIVE : DEAD;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "Uso: " << argv[0] << " <modo> [parametros...]" << endl;
        cout << "Modo 'r': " << argv[0] << " r <tamano_matriz> <num_iteraciones>" << endl;
        cout << "Modo 's': " << argv[0] << " s <archivo.txt>" << endl;
        return EXIT_FAILURE;
    }

    //inicializar variables
    char mode = argv[1][0];
    int n;
    int k;
    int size ;
    int *a;

    if (mode == 'r') {
        if (argc < 4) {
            cout << "Uso: " << argv[0] << " r <tamano_matriz> <num_iteraciones>" << endl;
            return EXIT_FAILURE;
        }
        if (!esNumerico(argv[2]) ) {
            cout << "El tamaño de la matriz debe ser un número entero." << endl;
            return EXIT_FAILURE;
        }
        if (!esNumerico(argv[3]) ) {
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
        filename = "../Gol2D_mimir/samples/"+filename;

        if (!fileExists(filename)) {
            cout << "El archivo " << filename << " no existe." << endl;
            return EXIT_FAILURE;}
        n = nInput(filename);
        if (n == -1) {
            cout << "El archivo " << filename << " no contiene el tamaño de la matriz." << endl;
            return EXIT_FAILURE;}
        cout << "Matriz de tamaño " << n << "x" << n << endl;
        size = (n + 2) * (n + 2) * sizeof(int);

        k = kInput(filename);
        if (k == -1) {
            cout << "El archivo " << filename << " no contiene el número de iteraciones." << endl;
            return EXIT_FAILURE;}

        //lee la matriz desde un archivo
        a = matrix2DInput(filename, n);
        if (a == NULL) {
            cout << "El archivo " << filename << " no contiene la matriz." << endl;
            return EXIT_FAILURE;}
        
        cout << "Archivo seleccionado: " << filename << endl;
    } 
    else {
        cout << "Modo desconocido. Usa 'r' o 's'." << endl;
        return EXIT_FAILURE;
    }

    //inicializar matrices en GPU
    unsigned block_size = 256;
    unsigned grid_size = (n * n + block_size - 1) / block_size;

    int *d_a, *d_b;
    checkCuda(cudaMalloc(&d_a, size));
    checkCuda(cudaMalloc(&d_b, size));
    if (mode == 'r'){
        int seed = rand(); 
        //inicializa la matriz con los valores en gpu
        fillMatrix<<<grid_size, block_size>>>(d_a, n, seed);
    }
    else if (mode == 's'){
        checkCuda(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    }

    checkCuda(cudaDeviceSynchronize());
    // se inicializa la matriz para el buffer
    float4 *d_live_cells;
    checkCuda(cudaMalloc(&d_live_cells, n * n * sizeof(float4)));

    //se inicializan las variables para la representacion grafica a traves de mimir
    MimirEngine engine;
    engine.init(1920, 1080);

    MemoryParams live_params;
    live_params.layout = DataLayout::Layout2D;
    live_params.element_count = {(unsigned)n, (unsigned)n};
    live_params.component_type = ComponentType::Float;
    live_params.channel_count = 4;
    live_params.resource_type = ResourceType::Buffer;

    // se crea el buffer con d_live_cells 
    auto live_buffer = engine.createBuffer((void**)&d_live_cells, live_params);

    ViewParams live_view_params;
    live_view_params.element_count = n * n;
    live_view_params.extent = {(unsigned)n, (unsigned)n, 1};
    live_view_params.data_domain = DataDomain::Domain2D;
    live_view_params.domain_type = DomainType::Structured;
    live_view_params.view_type = ViewType::Voxels;
    live_view_params.attributes[AttributeType::Color] = *live_buffer;
    live_view_params.options.default_color = {0.0f, 1.0f, 0.0f, 1.0f};
    live_view_params.options.default_size = 1;

    // se crea la vista con mimir
    engine.createView(live_view_params);
    engine.displayAsync();
    checkCuda(cudaDeviceSynchronize());
    separateCells<<<grid_size, block_size>>>(d_a, d_live_cells, n);
    engine.updateViews();

    // se itera la la matriz
    for (size_t iter = 0; iter < k; ++iter) {
        // se duerme el sistema para comprension
        this_thread::sleep_for(chrono::milliseconds(100));
        GOL<<<grid_size, block_size>>>(d_a, d_b, n);
        // se copia la matriz con el siguiente estado a la actual
        checkCuda(cudaMemcpy(d_a, d_b, size, cudaMemcpyDeviceToDevice));
        //se colocan lo elementos en el buffer
        separateCells<<<grid_size, block_size>>>(d_a, d_live_cells, n);
        checkCuda(cudaDeviceSynchronize());
        //se actualizan las vistas
        engine.updateViews();
    }

    engine.exit();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_live_cells);

    return EXIT_SUCCESS;
}

