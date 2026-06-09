# mmreal test suite

A dependency-free regression and coverage test suite for the `mmreal` library. It exercises the major API groups and locks in the behavior of bugs fixed during development. There is no external test framework: the harness is a single header (`test_util.h`) of `assert`-style macros.

## Files

| File | Purpose |
|------|---------|
| `test_util.h` | Minimal test harness: `TEST`, `RUN`, `CHECK`, `CHECK_NEAR`, and a summary that sets the process exit code. |
| `test_mmreal.c` | 25 tests across creation/ownership, element access, conversion, assembly, AXPY, products, statistics, and file I/O. |
| `Makefile` | Standalone build that compiles the library sources together with the tests and runs them. (Provided as `Makefile.test`; rename to `Makefile` or use `make -f Makefile.test`.) |

## Layout assumed

By default the Makefile expects the library next to the tests:

```
project/
├── include/        # mmreal.h, _blas_.h, mmio.h
├── src/            # mmreal.c, mmio.c
└── tests/          # test_mmreal.c, test_util.h, Makefile
    └── (built here)
```

The defaults are `SRC_DIR = ../src` and `INC_DIR = ../include`. Override them if your layout differs:

```sh
make SRC_DIR=path/to/src INC_DIR=path/to/include
```

## Running the tests

### Quick start (GCC + reference BLAS)

```sh
make CC=cc
```

This builds and runs the suite. On success you will see each test reported `[ OK ]` and a final line:

```
==== 25 tests run, 0 failed ====
```

and the process exits with status `0`. Any failure exits non-zero, so the suite works in CI.

> The `ERROR:` lines printed during the run are **expected**. Several tests deliberately drive error paths (invalid dimensions, destructive ops on a view, out-of-bounds access, scalar-add on a sparse matrix, the `x_dot_yk` bounds check, a truncated binary header). Those messages are the library correctly reporting to `stderr`; the tests that produce them still pass.

### Targets

```sh
make            # build and run (default target)
make build      # build the test binary only
make run        # run the test binary
make clean      # remove objects and the binary
```

## Choosing a compiler and BLAS

The suite links the same BLAS the library uses. The key rule, learned the hard way, is that **the BLAS integer width must match `blas_int`**.

### GCC / Clang

```sh
make CC=cc   BLAS_LIB="-lblas"        # reference BLAS (portable default)
make CC=cc   BLAS_LIB="-lopenblas"    # OpenBLAS (faster)
```

The Makefile selects `-fopenmp` automatically for non-Intel compilers.

### Intel icx + MKL

```sh
make CC=icx \
     BLAS_LIB="`pkg-config mkl-dynamic-lp64-iomp --define-variable=MKLROOT=${MKLROOT} --libs`"
```

For Intel compilers the Makefile uses `-qopenmp`.

## 32-bit vs 64-bit BLAS integers

By default the library uses `blas_int = int` (32-bit), which matches reference BLAS, OpenBLAS, and **LP64** MKL. This is the recommended configuration.

Only if you link a **64-bit-integer (ILP64)** BLAS must you switch `blas_int` to 64-bit, by passing `ILP64=1`:

```sh
make ILP64=1 \
     BLAS_LIB="`pkg-config mkl-dynamic-ilp64-iomp --define-variable=MKLROOT=${MKLROOT} --libs`"
```

**Mismatching these is the most common failure.** Linking an ILP64 BLAS while leaving `blas_int = int` (or the reverse) is an ABI mismatch: the first BLAS call passes an integer of the wrong width and the program crashes (typically a segfault on the first `dcopy_`). If a freshly built test binary segfaults early, check that the BLAS interface layer and the `ILP64` setting agree. Always `make clean` when switching, since stale objects would mix integer widths.

## Memory checking (recommended before release)

The tests cover view (non-owning) objects and sparse/dense conversions, so running them under AddressSanitizer and LeakSanitizer is a good way to confirm memory health:

```sh
make clean
make CC=cc \
     OPTFLAGS="-O1 -g -fsanitize=address,leak,undefined -fno-omit-frame-pointer" \
     BLAS_LIB="-lblas"
```

A clean run with no sanitizer output indicates no leaks, no double frees, and no undefined behavior on the exercised paths.

## What is covered

1. **Creation / ownership** — allocation, dimension validation, identity, copy; views do not own their data and reject destructive operations.
2. **Element access** — dense get/set, sparse insertion in unsorted order, out-of-bounds returns `NAN`.
3. **Conversion** — sparse↔dense round-trip, symmetric→general.
4. **Assembly / extraction** — vertical/horizontal concatenation, column extraction.
5. **AXPY-like** — `axpy`, `scale`, and rejection of scalar-add on sparse matrices.
6. **Products** — dot product, dense and sparse matrix-vector products agree, `x_dot_yk` bounds checking.
7. **Statistics** — sum, mean, sum-of-squares, L2 norm, sample std, index of max magnitude.
8. **File I/O** — Matrix Market coordinate read with unordered entries and summed duplicates, symmetric read, binary round-trip (sparse and dense), rejection of a truncated binary header.

## Adding a test

Add a function with the `TEST` macro and register it with `RUN` in `main`:

```c
TEST (my_new_case)
{
    mm_real *x = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 2, 4);
    CHECK (x != NULL);
    CHECK_NEAR (mm_real_get (x, 0, 0), 0.0, 1e-12);
    mm_real_free (x);
}

/* in main(): */
RUN (my_new_case);
```

`CHECK` and `CHECK_NEAR` record failures but do not abort, so one failing test does not stop the rest of the suite. The process exit code is non-zero if any check failed.
