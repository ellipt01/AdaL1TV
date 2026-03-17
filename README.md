# AdaL1TV - Magnetic Data Inversion with Adaptive L1-TV & L1-L2 Regularizations

## Overview

**AdaL1TV** is an inversion analysis tool designed to estimate 3D subsurface magnetization (or magnetic susceptibility) models from observed magnetic anomaly data. By adopting **ADMM (Alternating Direction Method of Multipliers)** as the optimization method, it provides advanced regularization techniques that excel at recovering sharp geological boundaries.

This project includes two main programs tailored to specific use cases:

1. `l1l2inv`: Inversion using L1-L2 regularization
2. `l1tvinv`: Inversion using Adaptive L1-Total Variation (Adaptive L1-TV) regularization

## Key Features

* **Advanced Regularization Algorithms**: Supports L1-L2 norm penalties based on sparse modeling, as well as 3D L1-TV regularization capable of adaptive weighting based on a guide model.
* **Highly Efficient ADMM Solver**: Minimizes the computational cost at each ADMM iteration by utilizing the Sherman-Morrison-Woodbury (SMW) formula and Cholesky decomposition.
* **Fast Large-Scale Computation**:

  * Implements multi-threaded processing using **OpenMP** for tasks such as constructing 3D spatial differential operators.
  * Utilizes the **Intel MKL PARDISO** solver for highly efficient solutions of large sparse systems of linear equations (CSR format).
* **Geophysical Modeling**: Allows arbitrary specification of the inclination and declination of the external geomagnetic field and the magnetization vector. Supports forward modeling kernel computation using 3D prism grids that can incorporate surface topography (terrain) models.

## Dependencies

The following environments and libraries are required to compile and run this software.

* **C++ Compiler**: Supports C++11 or later (e.g., g++, clang++, icpc, icpx)
* **OpenMP**: For multi-threaded processing
* **Intel MKL (Math Kernel Library)**: For LAPACKE (e.g., dpotrs) and the PARDISO solver
* **External Libraries**: `mgcal` (for magnetic calculations) and `mmreal` (for CSC sparse/dense matrix and vector operations) are included in the source tree.

## Build

This project uses a `Makefile` for compilation. Simply run the `make` command in the project root directory.

```bash
# Build all programs (l1l2inv and l1tvinv)
make
```

> **Note**: Please ensure that compiler settings such as Intel MKL and OpenMP flags are correctly configured in your `Makefile` before building.

## Input Data Formats

The text files read by this tool should be formatted with columns separated by tabs or spaces.

### 1. Magnetic Anomaly Data (Observation Data)

This is the main observation data specified with the `-f` option. Each row must contain the following four values:

```
<Easting (km)>    <Northing (km)>    <Altitude (km)>    <Magnetic Anomaly (nT)>
```

### 2. Terrain Data

Provide this file using the `-t` option when incorporating surface topography. Each row must contain the following three values:

```
<Easting (km)>    <Northing (km)>    <Elevation (km)>
```

## Usage

### 1. L1-L2 Inversion (`l1l2inv`)

Executes an inversion using a mixed penalty of L1 and L2 norms.

```bash
USAGE: ./l1l2inv -f <input_file> -l <log10(lambda)> [optional arguments]

Required Arguments:
  -f <input_file>       Path to the input observation data (magnetic anomaly) file
  -l <log10(lambda)>    Common logarithm of the regularization parameter (λ)

Optional Arguments:
  -a <alpha>            Mixing ratio of L1 and L2 regularization (default: 0.9)
  -t <terrain_file>     Path to the terrain file (default: terrain.in)
  -s <settings_file>    Path to the settings file (default: settings.par)
  -v                    Verbose mode
  -h                    Show help message
```

### 2. Adaptive L1-TV Inversion (`l1tvinv`)

Executes an inversion using Adaptive L1-Total Variation (TV) regularization.

```bash
USAGE: ./l1tvinv -f <input_file> -l <log10(lambda)> [optional arguments]

Required Arguments:
  -f <input_file>       Path to the input observation data (magnetic anomaly) file
  -l <log10(lambda)>    Common logarithm of the regularization parameter (λ)

Optional Arguments:
  -g <guide_model_file>:<sigma>:<c1>:<c2>
                        - guide_model_file: guide model in mm_real format
                        - sigma: regularization parameter (lambda) used to derive the guide model
                        - c1, c2: coefficients for defining the active set
  -c <weight_exponent>  Weight exponent (default: 1)
  -t <terrain_file>     Path to the terrain file (default: terrain.in)
  -s <settings_file>    Path to the settings file (default: settings.par)
  -v                    Verbose mode
  -h                    Show help message
```

## Outputs

Upon successful completion, the program generates the following files:

* `model.data`: The estimated final 3D model (coordinates of each grid and estimated physical property values)
* `recovered.data`: The predicted magnetic anomaly data calculated (forward modeled) from the estimated model
* `beta_L1L2.vec` / `beta_L1TV.vec`: The model vector file before weight removal, output in `mm_real` format

## License

GPL.
