
add_executable(gol3d Gol3D.cu)
target_link_libraries(gol3d PRIVATE mimir OpenMP::OpenMP_CXX)
set_target_properties(gol3d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/executables
)
