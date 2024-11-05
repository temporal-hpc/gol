cmake_minimum_required(VERSION 3.17)
project (RTX-RMQ LANGUAGES CXX CUDA)
include(FetchContent)

# Set default build type to 'Release'
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# add path to our CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

find_package(OpenMP)


FetchContent_Declare(mimir
    GIT_REPOSITORY https://github.com/temporal-hpc/mimir.git
    GIT_TAG        279e2ccea5f76fc806f5d867e8fc873b83ce4875 # master
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(mimir)

add_subdirectory(Gol3D)
add_subdirectory(Gol)

find_path(SHADERS_SOURCE_DIR shaders
  PATHS
    /usr/local/mimir
    /usr/mimir
    /opt/mimir
    /usr/local
    /usr
  PATH_SUFFIXES mimir/shaders
  NO_DEFAULT_PATH
)

# Si no se encuentra en las rutas estándar, solicita al usuario un path o lanza error
if(NOT SHADERS_SOURCE_DIR)
    message(WARNING "No se encontró el directorio de shaders de Mimir en las rutas comunes. Intentando consultar al usuario...")

    # Solicita al usuario que ingrese la ruta de instalación de mimir
    if(NOT DEFINED CUSTOM_SHADERS_PATH)
        set(CUSTOM_SHADERS_PATH "" CACHE STRING "Ruta de shaders de Mimir no encontrada. Por favor, proporcione un path.")
    endif()

    if(NOT CUSTOM_SHADERS_PATH)
        message(FATAL_ERROR "No se encontró el directorio de shaders de Mimir y no se proporcionó un path. Abortando.")
    else()
        set(SHADERS_SOURCE_DIR ${CUSTOM_SHADERS_PATH})
    endif()
endif()

message(STATUS "Usando shaders de Mimir desde: ${SHADERS_SOURCE_DIR}")



# Copiar shaders al directorio de salida
add_custom_target(copy_shaders ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${SHADERS_SOURCE_DIR}/shaders
          ${PROJECT_BINARY_DIR}/samples/shaders/
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Copiando shaders de Mimir..."
  VERBATIM
)
