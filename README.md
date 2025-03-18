# Game of Life con Mimir
Este proyecto es una implementación del **Juego de la Vida de Conway**, diseñada para ejecutarse en GPU y visualizarse mediante la biblioteca **Mimir**. Los cálculos se procesan directamente en la GPU, lo que permite una representación eficiente de los datos sin necesidad de transferirlos a la CPU. Para las visualizaciones en 2D y 3D, se emplean arreglos de tipo `float4`, que almacenan los estados de las celdas y permiten definir colores y niveles de transparencia, ofreciendo una experiencia visual más rica y dinámica.

## Características principales

- **Representación 2D**:
  - **Modo Random**: Inicializa el estado del juego de manera aleatoria.
  - **Modo Samples**: Permite cargar un estado inicial desde un archivo de texto.
  
- **Representación 3D**:
  - Solo admite inicialización aleatoria.

- **Uso de GPU**:
  - Los cálculos se realizan en la GPU, y los datos se visualizan directamente desde la memoria de la GPU utilizando Mimir.

## Requisitos

- **CMake**: Herramienta para la configuración y compilación del proyecto.
- **CUDA**: Asegúrate de tener CUDA instalado en tu sistema.
- **Mimir**: La biblioteca Mimir debe estar correctamente configurada en tu entorno.

## Instalación

Para compilar y ejecutar el proyecto, sigue estos pasos:

1. Crea un directorio `build` y accede a él:
    ```
   mkdir build
   cd build
    ```

2. Ejecuta `cmake` para configurar el proyecto:
    ``` 
   cmake ..
    ``` 

3. Compila el proyecto con `make`:
    ``` 
   make
    ``` 

   Esto generará un directorio llamado `executables` dentro de la carpeta `build`, donde encontrarás los archivos ejecutables listos para usar.

## Modo de uso

Desde la carpeta `executables`, ejecuta los siguientes comandos:

### Game of Life 2D

#### Modo Random
Para ejecutar el modo aleatorio, usa el siguiente comando:
 ``` 
./gol2d r <n> <k>
 ``` 
- **`n`**: Tamaño de la matriz (n x n).
- **`k`**: Número de iteraciones.

#### Modo Samples
Para ejecutar el modo con un archivo de texto, usa el siguiente comando:
 ``` 
./gol2d s <nombre de ejemplo>.txt
 ``` 
- **`nombre de ejemplo`**: Nombre del archivo de texto que contiene el estado inicial. Los archivos se encuentran en la carpeta `gol/samples/`.

### Game of Life 3D

Para ejecutar la versión 3D, usa el siguiente comando:
 ``` 
./gol3d <n> <k>
 ``` 
- **`n`**: Tamaño de la matriz (n x n x n).
- **`k`**: Número de iteraciones.

## Ejemplos de archivos de texto (Samples)

En la carpeta `samples/` se encuentran varios archivos de texto con configuraciones iniciales interesantes. A continuación, se describe cada uno de ellos:

1. **Diamond**:
   - Un patrón simétrico que genera formas de diamante en evolución.

2. **Glider**:
   - Un patrón pequeño que se mueve diagonalmente a través de la matriz.

3. **GliderGun**:
   - Un patrón que genera "gliders" de manera continua.

4. **Pulsar**:
   - Un patrón oscilante que cambia entre varios estados en ciclos.

5. **GrowingSpaceship**:
   - Un patrón que simula una "nave espacial" en crecimiento.

6. **Exploder**:
   - Un patrón que genera explosiones controladas.


Para ejecutar cualquiera de los ejemplos, usa el siguiente comando:
 ``` 
./gol2d s <nombre de ejemplo>.txt
 ``` 

## Creación de nuevos samples

Puedes crear tus propios archivos de texto para definir nuevos estados iniciales. Simplemente sigue el formato descrito a continuación y coloca el archivo en la carpeta `samples/`.

### Formato del archivo de texto

1. La primera línea indica el tamaño de la matriz (`n`).
2. La segunda línea indica el número de iteraciones (`k`).
3. Las siguientes líneas representan la matriz inicial, donde `1` indica una celda viva y `0` una celda muerta.

### Ejemplo de archivo (`mi_sample.txt`):
 ``` 
n=10
k=100
0000000000
0001110000
0001010000
0001110000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
 ``` 

Una vez creado el archivo, puedes ejecutarlo con:
 ``` 
./gol2d s mi_sample.txt
 ``` 
 ## Errores conocidos

- **Representación 3D y tamaño de los cubos**:
  En la representación 3D, el tamaño de los "cubos" que representan las celdas puede solaparse si el valor de `n` (tamaño de la matriz) es demasiado pequeño o demasiado grande. Esto puede afectar la claridad de la visualización. Para una mejor representación, se recomienda ajustar el valor de `n` según sea necesario. Si los cubos se ven demasiado juntos o solapados, prueba con un valor de `n` más grande o más pequeño hasta lograr una visualización óptima.
