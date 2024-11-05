# Gol

## Building

To build gol, you need a C++20 host compiler, CUDA SDK >= 10.0 with the included nvcc device compiler, and cmake >= 3.24. After cloning or downloading the repository located at `<gol_dir>`, to generate a build in folder `<build_dir>`, run:

```bash
cmake -B <build_dir> -S <gol_dir>
cmake --build <build_dir> --config <mode> --parallel <ncores>
```

## Running gol
A successful build with the above steps allows to run gol from the `build` directory, so running any version of gol should be executed as `./executable/<gol version>`. This Proyect currently provides the following version of gol:
* `gol`: CUDA implementation of the Game of Life (GOL) using Mimir for visualization.
* `gol3D`: CUDA implementation of the Game of Life (GOL) in three dimensions using Mimir for visualization.

