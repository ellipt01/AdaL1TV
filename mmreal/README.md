# mmreal

A C library for real-valued matrices in [Matrix Market](https://math.nist.gov/MatrixMarket/) format, supporting both sparse (CSC) and dense (column-major) storage with a unified API. It provides element access, format/type conversion, assembly, BLAS-backed linear algebra, column statistics, and text/binary file I/O.

The library is built on top of the NIST Matrix Market I/O routines (`mmio`), parameterized so that the index type and the BLAS-boundary integer type can be configured independently at build time.

---

## Features

- **Two storage formats behind one type.** A single `mm_real` object is either sparse (Compressed Sparse Column) or dense (column-major). The same public functions accept both and dispatch internally.
- **Symmetry-aware.** General, symmetric-upper, and symmetric-lower matrices are handled consistently across get/set, conversion, products, and statistics.
- **BLAS-backed.** Dense kernels delegate to Level 1/2/3 BLAS (`ddot`, `daxpy`, `dscal`, `dcopy`, `dgemv`, `dsymv`, `dgemm`, `dsymm`); sparse kernels use hand-written CSC traversals.
- **Views without copies.** `mm_real_view_array` wraps an existing data array without taking ownership; destructive operations are guarded so a view's backing memory is never freed or reallocated.
- **Configurable integer widths.** The matrix index type (`MM_INT`) and the BLAS integer type (`blas_int`) are decoupled, so the library works with both 32-bit (LP64 / reference BLAS) and 64-bit (ILP64) BLAS.
- **Robust file I/O.** The Matrix Market reader handles unordered coordinate input, sorts row indices within columns, and sums duplicate entries. A compact custom binary format is also provided, with header sanity checks and error propagation.

---

## Requirements

- A C11 compiler (uses `<threads.h>` `thread_local` and `<stdbool.h>`).
- A BLAS implementation (reference BLAS, OpenBLAS, or Intel MKL).
- The Matrix Market I/O sources (`mmio.h`, `mmio.c`), included alongside this library.

---

## Building

The library compiles to a static archive (`libmmreal.a`). The provided `Makefile` defaults to the Intel compiler (`icx`) and Intel MKL, but the code itself is vendor-neutral and builds with `gcc`/`clang` plus any BLAS.

### Default build (LP64 / 32-bit BLAS integers)

```sh
make
```

This produces `libmmreal.a` with `MM_INT = long` (64-bit indices) and `blas_int = int` (32-bit BLAS integers), linking against an LP64 BLAS. This is the recommended configuration and is compatible with reference BLAS and OpenBLAS.

### 64-bit BLAS build (ILP64)

Only needed when a single BLAS call must address more than `INT_MAX` elements:

```sh
make clean
make ILP64=1
```

This defines `-DMM_BLAS_ILP64` (making `blas_int = int64_t`) and links an ILP64 BLAS. Always run `make clean` first when switching, since stale objects would mix integer widths.

### Installing

```sh
make install      # copies libmmreal.a to ../lib
```

---

## Integer type configuration

The library uses two independent integer types. Understanding the distinction matters when linking against a specific BLAS.

| Type | Role | Default width | How to change |
|------|------|---------------|---------------|
| `MM_INT` | Matrix indices and sizes (`m`, `n`, `nnz`, CSC arrays) | `long` (64-bit) | `-DMMINT` in `mmio.h` makes it `int` |
| `blas_int` | Integer arguments passed to BLAS | `int` (32-bit) | `-DMM_BLAS_ILP64` makes it `int64_t` |

The CSC index arrays (`i`, `p`) are never passed to BLAS, so they keep the full `MM_INT` range regardless of the BLAS integer width. Values crossing the BLAS boundary (dimensions, element counts, strides) are converted through a checked cast; a dimension exceeding the BLAS integer range is rejected at matrix-creation time rather than silently truncated.

For a consistent build, keep both types at the same width: either 32-bit throughout (reference/LP64 BLAS) or 64-bit throughout (ILP64 BLAS). Mixing a 64-bit `blas_int` with a 32-bit BLAS library (or vice versa) is an ABI mismatch.

When building against Intel MKL specifically, the ILP64 configuration additionally requires `-DMKL_ILP64`, and the threading layer (`iomp`) should match the compiler's OpenMP runtime. The provided `Makefile` handles this via the `ILP64` switch.

---

## Quick start

```c
#include <stdio.h>
#include "mmreal.h"

int main (void)
{
    // Read a matrix from a Matrix Market file.
    FILE *fp = fopen ("matrix.mtx", "r");
    mm_real *A = mm_real_fread (fp);
    fclose (fp);
    if (A == NULL) {
        fprintf (stderr, "read failed: %s\n",
                 mm_result_to_string (mm_real_get_last_error ()));
        return 1;
    }

    // Make a dense right-hand-side vector x of all ones (A->n rows, 1 column).
    mm_real *x = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, A->n, 1, A->n);
    mm_real_set_all (x, 1.0);

    // Compute y = A * x  (y is a dense A->m x 1 vector).
    mm_real *y = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, A->m, 1, A->m);
    mm_real_x_dot_yk (false, 1.0, A, x, 0, 0.0, y);

    // Print the result.
    mm_real_fwrite (stdout, y, "%g");

    mm_real_free (A);
    mm_real_free (x);
    mm_real_free (y);
    return 0;
}
```

Compile and link (reference BLAS example):

```sh
cc quick_start.c -I./include -L./lib -lmmreal -lblas -o quick_start
```

---

## Core concepts

### The matrix object

Every matrix is an `mm_real` (aliased as `mm_sparse` and `mm_dense` for readability). Its key fields are the dimensions `m`, `n`, `nnz`; the data array `data`; the CSC arrays `i` (row indices) and `p` (column pointers, used only when sparse); and an `owner` flag indicating whether the object owns its data.

### Ownership and views

Most constructors allocate and own their data. `mm_real_view_array` instead wraps a caller-supplied array: the resulting object has `owner = false` and never frees that array. Operations that would free or reallocate the backing store (in-place transpose, format conversion, `realloc`) check ownership first and fail cleanly on a view rather than corrupting external memory.

### Error handling

Functions returning a pointer return `NULL` on failure; functions returning `bool` return `false`; scalar functions return `NAN`. The last error code is stored per-thread and retrieved with `mm_real_get_last_error`, then turned into a message with `mm_result_to_string`. The library reports errors to `stderr` but never calls `exit`.

---

## File formats

### Matrix Market text format

`mm_real_fread` / `mm_real_fwrite` read and write the standard Matrix Market format (coordinate for sparse, array for dense; real-valued, general or symmetric). Coordinate input need not be sorted or duplicate-free: row indices are sorted within each column on read, and duplicate `(i, j)` entries are summed.

### Custom binary format

`mm_real_fread_binary` / `mm_real_fwrite_binary` use a compact binary layout for fast round-tripping. The first four bytes are a typecode tag (`"MCRS"` for sparse, `"MARG"` for dense), followed by `m`, `n`, `nnz`, then the index and data arrays. The reader validates the header before allocating. Binary files are written with the build's `MM_INT` width, so reader and writer must share the same `MM_INT` configuration.

---

## API overview

The full function-by-function reference is in [`API.md`](API.md). The public API is organized into groups:

1. **Creation, destruction, copying** — `mm_real_new`, `mm_real_free`, `mm_real_copy`, `mm_real_eye`, `mm_real_view_array`, `mm_real_realloc`, `mm_real_resize`.
2. **Element access** — `mm_real_get`, `mm_real_set`, `mm_real_set_all`, `mm_real_memcpy`, `mm_real_transpose`, `mm_real_sort`.
3. **Format / type conversion** — sparse↔dense and symmetric↔general, in-place and copying variants.
4. **Assembly / extraction** — `mm_real_vertcat`, `mm_real_horzcat`, column and row extraction.
5. **AXPY-like operations** — `mm_real_axpy`, `mm_real_axjpy`, `mm_real_scale`, `mm_real_xj_scale`, `mm_real_add`, `mm_real_xj_add`.
6. **Products** — dot products, matrix-matrix (`mm_real_x_dot_y`), matrix-column (`mm_real_x_dot_yk`), and transposed-column products.
7. **Column statistics** — `iamax`, `asum`, `sum`, `nrm2`, `ssq`, `mean`, `std`.
8. **File I/O** — text and binary read/write.

---

## Notes and limitations

- Adding a scalar to a sparse matrix (`mm_real_add` / `mm_real_xj_add`) is rejected, because it would densify the matrix. Convert to dense first.
- Some symmetric sparse operations (row/column extraction, `asum`, `sum`) reconstruct the full column by searching, which is `O(n log m)` rather than `O(nnz)`. These are correct but not optimized for symmetric sparse input.
- `mm_real_fread_dense` reads `m * n` values; symmetric matrices stored in packed array form are not specially handled.

---

## Acknowledgements

The Matrix Market I/O layer (`mmio`) derives from the reference implementation distributed by the U.S. National Institute of Standards and Technology (NIST). See <https://math.nist.gov/MatrixMarket/> for the format specification.
