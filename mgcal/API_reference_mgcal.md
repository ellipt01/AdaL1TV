# mgcal (minimal) — API Reference

A minimal C library for computing magnetic forward-modeling sensitivity matrices
from rectangular-prism sources on a structured 3-D grid. This build is reduced to
the **prism (3-D)** forward model and the **`kernel_matrix_set`** workflow; dipole,
2-D (yz), scattered, mapped, and per-row/column kernel routines have been removed.

All field values are multiplied by a global `scale_factor` (default `100.0`).

---

## Conventions

- **Coordinate system:** z-axis points upward. The surface is at larger z, the
  subsurface at smaller (typically negative) z. A z-range such as `[0, -2]` runs
  from the surface down to 2 km depth, so cell thickness `dz` is stored as a
  **negative** value; the prism integrator compensates for the sign internally.
- **Memory ownership:** functions returning a pointer allocate it; the caller is
  responsible for freeing it with the matching `*_free` function (or `free()` for
  raw `double *` buffers).
- **Error handling:** invalid arguments and allocation failures call
  `error_and_exit` (from `private/util.h`), which prints a diagnostic and exits.
- **Angles:** inclination/declination are given in degrees.

---

## Header layout

Include the umbrella header:

```c
#include <mgcal.h>
```

It pulls in, in dependency order: `vector3d.h`, `data_array.h`, `grid.h`,
`source.h`, `calc.h`, `io.h`, `kernel.h`. Individual headers are not self-contained
and should not be included directly.

---

## Data types

### `vector3d`
Cartesian 3-vector.

```c
struct s_vector3d { double x, y, z; };
```

### `data_array`
Observation points and their measured values.

```c
struct s_data_array {
    int     n;       // number of points
    double *x, *y, *z;  // coordinates
    double *data;    // observed value at each point
};
```

### `grid`
Structured 3-D model grid (nx × ny × nz cells).

```c
struct s_grid {
    int     n, nh, nx, ny, nz;     // total, horizontal-layer, and per-axis counts
    double  xrange[2], yrange[2], zrange[2];
    double *x, *y, *z;             // cell-center coordinates per axis
    double *z1;                    // optional irregular surface (length nh)
    double *dx, *dy, *dz;          // per-axis cell sizes (dz may be negative)
    void   *data;
};
```

### `source`
A list of magnetized prism items plus an external-field direction. Built
incrementally with `source_append_item`; the field callbacks iterate from
`src->begin`.

### `mgcal_func`
Binds a forward-model callback with an optional user parameter.

```c
typedef double (*mgcal_theoretical)(const vector3d *pos, const source *src, void *data);
struct s_mgcal_func { mgcal_theoretical function; void *parameter; };
```

### `MgcalComponent`
Selects which field component a callback returns.

```c
typedef enum {
    MGCAL_X_COMPONENT,   // Hx
    MGCAL_Y_COMPONENT,   // Hy
    MGCAL_Z_COMPONENT,   // Hz
    MGCAL_TOTAL_FORCE    // total-force anomaly (exf . f)
} MgcalComponent;
```

---

## vector3d.h — 3-vector primitives

| Function | Summary |
|----------|---------|
| `vector3d *vector3d_new(double x, double y, double z)` | Allocate a vector with the given components. |
| `vector3d *vector3d_new_with_geodesic_poler(double r, double inc, double dec)` | Build a vector from magnitude and inclination/declination (degrees), geomagnetic convention: `x=r·cos(inc)·sin(dec)`, `y=r·cos(inc)·cos(dec)`, `z=−r·sin(inc)`. |
| `void vector3d_set(vector3d *cv, double x, double y, double z)` | Overwrite an existing vector's components in place. |
| `void vector3d_free(vector3d *cv)` | Free a vector (NULL-safe). |
| `void vector3d_scale(vector3d *x, double alpha)` | Multiply all components by `alpha` in place. |
| `vector3d *vector3d_copy(const vector3d *src)` | Allocate a copy of `src`. |
| `void vector3d_axpy(double alpha, const vector3d *x, vector3d *y)` | In-place `y = y + alpha·x`. |
| `double vector3d_dot(const vector3d *x, const vector3d *y)` | Dot product `x·y`. |
| `double vector3d_nrm(const vector3d *c)` | Euclidean norm `|c|`. |

---

## data_array.h — observation data

| Function | Summary |
|----------|---------|
| `data_array *data_array_new(int n)` | Allocate a data array for `n` points; coordinate and value buffers are zero-initialized. |
| `void data_array_free(data_array *array)` | Free a data array and its buffers (NULL-safe). |

---

## grid.h — model grid

| Function | Summary |
|----------|---------|
| `grid *grid_new(int nx, int ny, int nz, const double x[], const double y[], const double z[])` | Create a regular grid from corner ranges (each a `[min,max]` pair); cell sizes from even division. **Caveat:** for an axis with a single cell, the range's second value is ignored and that cell size becomes zero — use `grid_new_full` instead. |
| `grid *grid_new_full(int nx, int ny, int nz, const double x[], const double y[], const double z[], const double *dx, const double *dy, const double *dz, const double *z1)` | Create a grid with explicit per-axis cell sizes and an optional irregular surface `z1` (length `nx·ny`). Pass NULL for any of `dx/dy/dz` to fall back to even division. |
| `bool grid_set_surface(grid *g, const double *z1)` | Attach or replace an irregular top surface (`z1`, length `nx·ny`); each cell center's z is offset by `z1[h]` when queried. Frees any previous surface. Returns true on success. |
| `void grid_free(grid *g)` | Free a grid and all coordinate/spacing arrays (NULL-safe). |
| `void grid_get_index(const grid *g, int n, int *i, int *j, int *k, int *h)` | Decompose flat cell index `n` into `(i,j,k)` and horizontal index `h`; any output pointer may be NULL. |
| `void grid_get_nth(const grid *g, int n, vector3d *center, vector3d *dim)` | Get the center coordinate and/or cell dimension of cell `n` (center includes the `z1` offset when present); either output may be NULL. |
| `void grid_stretch_at_edge(grid *g, double l)` | Extend the outermost cells outward by length `l` without adding cells: horizontal edges (x, y) grow on both sides, and the deepest z-layer is extended downward (consistent with z-up, negative `dz`). |

---

## source.h — magnetized sources

| Function | Summary |
|----------|---------|
| `source *source_new(void)` | Create an empty source (no external field, no items). |
| `void source_free(source *src)` | Free a source, its item list, and the external field. |
| `void source_set_external(source *src, double inc, double dec)` | Set the external-field direction from inclination/declination (degrees) as a unit vector. |
| `int source_append_item(source *src)` | Append a new empty magnetized item; sets `src->begin` on first append, updates `src->end`, returns the item index. Fill `pos`/`dim`/`mgz` directly afterward. |

---

## calc.h — prism forward model

| Function | Summary |
|----------|---------|
| `vector3d *prism(const vector3d *obs, const source *s)` | Total magnetic field at `obs` from all prism items in `s`; sums the signed 8-vertex contributions and applies `scale_factor`. Returns a newly allocated vector (caller frees). |
| `double total_force(const vector3d *exf, const vector3d *f)` | Total-force anomaly: projection `exf·f` of field `f` onto the external-field direction. |
| `double x_component_prism(const vector3d *obs, const source *src, void *data)` | `mgcal_func` callback: x-component of the prism field. |
| `double y_component_prism(const vector3d *obs, const source *src, void *data)` | `mgcal_func` callback: y-component of the prism field. |
| `double z_component_prism(const vector3d *obs, const source *src, void *data)` | `mgcal_func` callback: z-component of the prism field. |
| `double total_force_prism(const vector3d *obs, const source *src, void *data)` | `mgcal_func` callback: total-force anomaly of the prism field. |

The four `*_component_*` callbacks all match the `mgcal_theoretical` signature and
are intended to be passed to `mgcal_func_new`. The `data` argument is unused.

**Singularity handling:** when an observation point coincides with a prism vertex
(`r = 0`), the kernel returns zero contribution instead of producing `NaN`.

---

## kernel.h — sensitivity matrix assembly

| Function | Summary |
|----------|---------|
| `mgcal_func *mgcal_func_new(mgcal_theoretical func, void *data)` | Bundle a forward-model callback with a user parameter. Returns a new `mgcal_func`. Ownership of `data` stays with the caller. |
| `void mgcal_func_free(mgcal_func *f)` | Free an `mgcal_func` (does not free the user parameter). |
| `void kernel_matrix_set(double *a, const data_array *array, const grid *g, const vector3d *mgz, const vector3d *exf, const mgcal_func *f)` | Fill a preallocated sensitivity matrix. Computes `a[l + j·m] = f(obs_l, cell_j)` for all `m` observations and all `g->n` cells, with fixed magnetization `mgz` and external field `exf`. Column-major in the cell index. OpenMP-parallel over z-layers. Caller allocates `m · g->n` doubles. |
| `double *kernel_matrix(const data_array *array, const grid *g, const vector3d *mgz, const vector3d *exf, const mgcal_func *f)` | Convenience wrapper: allocates `m · g->n` doubles, calls `kernel_matrix_set`, and returns the buffer (caller frees with `free()`). |

### Matrix layout

For `m` observation points and `g->n` grid cells, the output buffer holds the
matrix in cell-major order: cell `j` occupies the contiguous block
`a[j·m .. j·m + m − 1]`, where entry `l` within that block is the response of
observation `l` to a unit source in cell `j`. This corresponds to a column of the
sensitivity matrix `G` (observations × cells).

---

## io.h — file input / output

| Function | Summary |
|----------|---------|
| `data_array *fread_data_array(FILE *stream)` | Read observation data (`x y z value` per line) from a stream; skips comment (`#`) and malformed lines. Returns a new `data_array`. |
| `void fwrite_data_array_with_data(FILE *stream, const data_array *array, const double *data, const char *format)` | Write `x/y/z` from `array` paired with an external `data` vector, one record per line, using `format` (4 fields). |
| `void fwrite_data_array(FILE *stream, const data_array *array, const char *format)` | Write a data array using its own stored values. |
| `grid *fread_grid(FILE *stream)` | Parse a grid definition (dimensions, corner positions, per-axis coordinate and spacing arrays, optional `z1`) from a stream. Returns a new `grid`. |
| `void fwrite_grid(FILE *stream, const grid *g)` | Write a grid definition in the format readable by `fread_grid`. |
| `void fwrite_grid_to_xyz(FILE *stream, const grid *g, const char *format)` | Write the center coordinate of every cell as `x y z`. |
| `void fwrite_grid_with_data(FILE *stream, const grid *g, const double *data, const char *format)` | Write each cell center paired with a per-cell value (`x y z value`); NULL `data` writes zeros. Typically used to export inversion models. |

---

## mgcal.h — global scale factor

| Function | Summary |
|----------|---------|
| `void mgcal_set_scale_factor(double val)` | Set the global output scale applied to all field values. |
| `double mgcal_get_scale_factor(void)` | Return the current global scale factor. |
| `extern double scale_factor` | The global scale (default `100.0`); defined in `calc.c`. |

---

## Typical workflow

```c
#include <mgcal.h>

/* 1. Build the model grid (explicit dz, negative for downward). */
double x[2] = {0.0, 0.0}, y[2] = {0.0, 0.0}, z[2] = {0.0, 0.0};
double dx[1] = {1.0}, dy[1] = {1.0}, dz[1] = {-1.0};
grid *g = grid_new_full(1, 1, 1, x, y, z, dx, dy, dz, NULL);

/* 2. Load observations. */
data_array *obs = fread_data_array(fp);

/* 3. Choose magnetization and external field (e.g. inc=45, dec=0). */
vector3d *mgz = vector3d_new_with_geodesic_poler(1.0, 45.0, 0.0);
vector3d *exf = vector3d_new_with_geodesic_poler(1.0, 45.0, 0.0);

/* 4. Wrap the desired forward model (total-force here). */
mgcal_func *f = mgcal_func_new(total_force_prism, NULL);

/* 5. Build the sensitivity matrix G (obs × cells), cell-major. */
double *G = kernel_matrix(obs, g, mgz, exf, f);

/* ... use G in the inversion ... */

free(G);
mgcal_func_free(f);
vector3d_free(mgz);
vector3d_free(exf);
data_array_free(obs);
grid_free(g);
```

---

## Internal utilities

The sources depend on `src/util.h` / `src/util.c` for `error_and_exit`,
`array_copy`, `array_set_all`, and `set_range`. These are internal helpers, not
part of the public API, and `util.h` is not installed alongside the public headers.
