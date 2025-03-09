add_executable(gol2d "Gol2D.cu" )
target_link_libraries(gol2d PRIVATE mimir OpenMP::OpenMP_CXX)
set_target_properties(gol2d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/executables
)
