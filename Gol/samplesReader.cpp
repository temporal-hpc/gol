#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cctype>
#include <string>

using namespace std;




/**
 * @brief Comprueba si un archivo existe.
 * @param filename nombre del archivo a analizar.
 * @return true si el archivo existe y puede abrirse, false en caso contrario.
 */
bool fileExists(const string& filename) {
    ifstream file(filename);
    return file.good(); // Retorna true si el archivo puede abrirse, false en caso contrario
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
 * @brief Lee un archivo de texto en busca de n.
 * @param filename nombre del archivo.
 * @return retorna n o -1 si se encontro un error.
 */
int nInput(string filename){
    ifstream file(filename);  

    if (!file) {
        cerr << "Error: No se pudo abrir el archivo " << filename << endl;
        return 1;
    }

    int n;                         // Variable para almacenar el tamaño de la matriz
    string line;
    getline(file, line);
    if (line.find("n=") != string::npos) {
        if (esNumerico(line.substr(2))) {
            // Extraer el valor de `n`
            return stoi(line.substr(2));
        }
    }
    cout<<"El formato del tamaño de la matriz no es el correcto."<<endl;
    return -1;
}

/**
 * @brief Lee un archivo de texto en busca de k.
 * @param filename nombre del archivo.
 * @return retorna k o -1 si se encontro un error.
 */
int kInput(string filename){
    ifstream file(filename);  
    if (!file) {
        cerr << "Error: No se pudo abrir el archivo " << filename << endl;
        return 1;
    }

    int k;
    string line;
    getline(file, line);
    getline(file, line);
    if (line.find("k=") != string::npos) {
        if (esNumerico(line.substr(2))) {
            // Extraer el valor de `k`
            return stoi(line.substr(2));
        }
    }
    cout<<"El formato de las iteraciones no es el correcto."<<endl;
    return -1;
}

/**
 * @brief Calcula el índice de una posición en la matriz para mayor facilidad de operación.
 * @param index índice a calcular.
 * @param n dimension de la matriz cuadrada (n*n).
 * @return el índice en la matriz.
 */
int index(int index, int n) {
    return ((index / n) + 1) * (n + 2) + (index % n + 1);
}

/**
 * @brief Lee un archivo de texto en busca de la matriz.
 * @param filename nombre del archivo.
 * @param n dimension de la matriz cuadrada (n*n).
 * @return retorna la matriz o NULL si se encontro un error.
 *
 * El archivo debe tener el formato:
 * n=<tamanio_matriz>
 * k=<num_iteraciones>
 * <matriz>
 *
 * La matriz se representa como una cadena de 0 y 1, con cada fila separada por un salto de l \n ieja.
 * @throws runtime_error si no se puede abrir el archivo.
 */
int* matrix2DInput(string filename, int n){
    ifstream file(filename);  
    if (!file) {
        throw runtime_error("Error: No se pudo abrir el archivo.");
        return NULL;
    }

    int* matrix = new int[(n + 2) * (n + 2)]();  // Inicializa todo en 0

    string line;
    getline(file, line); // Leer "n=..."
    getline(file, line); // Leer "k=..."

    int i = 0;  
    for (int j = 0; j < n; j++) {
        (getline(file, line));
        if (line.length() != static_cast<size_t>(n)) {
            cerr << "Formato inválido: la línea no tiene el tamaño esperado." << endl;
            delete[] matrix;
            return NULL;
        }
        for (char c : line) {
            if (c != '0' && c != '1') {
                cerr << "Formato inválido: caracteres no permitidos en la matriz." << endl;
                delete[] matrix;
                return NULL;
            }

            // Insertar el valor en la nueva matriz desplazado en +1 en filas y columnas
            matrix[index(i, n)] = c - '0';
            i++;
        }
        if ((file.eof()) && (j!=n-1)) {
            cout << "Faltas " << n-j-1 << " filas en el archivo." << endl;
            delete[] matrix;
            return NULL;
        }
    }
    file.close();
    return matrix;
}
