#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <limits>
#include <omp.h>
#include <mmreal.h>
#include "DiffOp.h"

/**
 * @file
 * @brief Implementation of the DiffOp class and related functions for creating
 * discrete differential operators on 3D grids.
 *
 * The DiffOp class generates sparse first-order forward difference operators
 * (Dx, Dy, Dz) along the x, y, and z axes. It can also construct a combined
 * operator D by vertically stacking these individual operators.
 *
 * The file also includes standalone factory functions to create second-order
 * difference operators (discrete Laplacians) like D^T*D, which are common in
 * scientific computing and optimization problems (e.g., for Total Variation).
 */

// =========================================================================
// DiffOp Class Implementation
// =========================================================================

/**
 * @brief Main factory method to build the combined first-order differential operator D.
 * @details Constructs D by vertically stacking individual operators.
 * The final structure can be either D = [I; Dx; Dy; Dz] or D = [Dx; Dy; Dz].
 *
 * @param nx Grid size in the x-dimension.
 * @param ny Grid size in the y-dimension.
 * @param nz Grid size in the z-dimension.
 * @param includeIdentity If true, prepends an identity matrix (the D0 form).
 * @return A pointer to the resulting sparse matrix D. The caller must free this memory.
 * @throw std::invalid_argument if grid dimensions are not positive.
 */
mm_real *
DiffOp::build (size_t nx, size_t ny, size_t nz, bool includeIdentity)
{
	if (nx <= 0 || ny <= 0 || nz <= 0)
		throw std::invalid_argument("DiffOp::build: Grid dimensions must be positive.");
	return (includeIdentity) ? buildD0_ (nx, ny, nz) : buildD1_ (nx, ny, nz);
}

/**
 * @brief Creates the sparse first-order forward difference operator for the x-axis (Dx).
 * @details This operator approximates the partial derivative with respect to x.
 * For a vector `v` representing the grid, `Dx * v` yields a vector
 * where each element approximates `v(i+1,j,k) - v(i,j,k)`.
 * The matrix is constructed in CSC (Compressed Sparse Column) format.
 *
 * @return A pointer to the sparse n x n matrix Dx.
 */
mm_real *
DiffOp::createDiffX (size_t nx, size_t ny, size_t nz)
{
	if (nx <= 0 || ny <= 0 || nz <= 0)
		throw std::invalid_argument("DiffOp::createDiffX: Grid dimensions must be positive.");
	size_t	n = nx * ny * nz;
	size_t	nnz = 2 * n;
	mm_real	*d = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, nnz);

	d->p[0] = 0;
	size_t	p = 0;
	size_t	l = 0;
	for (size_t k = 0; k < nz; k++) {
		for (size_t j = 0; j < ny; j++) {
			for (size_t i = 0; i < nx; i++) {
				if (i > 0) {
					d->i[p] = l - 1;
					d->data[p++] = 1.;
				}
				d->i[p] = l;
				d->data[p++] = -1.;
				d->p[++l] = p;
				if (p >= nnz) {
					nnz += n;
					mm_real_realloc (d, nnz);
				}
			}
		}
	}
	// Trim memory if the pre-allocation was an estimate.
	if (p != d->nnz) mm_real_realloc (d, p);
	return d;
}

/**
 * @brief Creates the sparse first-order forward difference operator for the y-axis (Dy).
 * @details Constructs the matrix in CSC format. Approximates `v(i,j+1,k) - v(i,j,k)`.
 *
 * @return A pointer to the sparse n x n matrix Dy.
 */
mm_real *
DiffOp::createDiffY (size_t nx, size_t ny, size_t nz)
{
	if (nx <= 0 || ny <= 0 || nz <= 0)
		throw std::invalid_argument("DiffOp::createDiffY: Grid dimensions must be positive.");
	size_t	n = nx * ny * nz;
	size_t	nnz = 2 * n;
	mm_real	*d = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, nnz);

	d->p[0] = 0;
	size_t	p = 0;
	size_t	l = 0;
	for (size_t k = 0; k < nz; k++) {
		for (size_t j = 0; j < ny; j++) {
			for (size_t i = 0; i < nx; i++) {
				if (j > 0) {
					d->i[p] = l - nx;
					d->data[p++] = 1.;
				}
				d->i[p] = l;
				d->data[p++] = -1.;
				d->p[++l] = p;
				if (p >= nnz) {
					nnz += n;
					mm_real_realloc (d, nnz);
				}

			}
		}
	}
	// Trim memory if the pre-allocation was an estimate.
	if (p != d->nnz) mm_real_realloc (d, p);
	return d;
}

/**
 * @brief Creates the sparse first-order forward difference operator for the z-axis (Dz).
 * @details Constructs the matrix in CSC format. Approximates `v(i,j,k+1) - v(i,j,k)`.
 *
 * @return A pointer to the sparse n x n matrix Dz.
 */
mm_real *
DiffOp::createDiffZ (size_t nx, size_t ny, size_t nz)
{
	if (nx <= 0 || ny <= 0 || nz <= 0)
		throw std::invalid_argument("DiffOp::createDiffZ: Grid dimensions must be positive.");
	size_t	n = nx * ny * nz;
	size_t	nnz = 2 * n;
	mm_real	*d = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, nnz);

	d->p[0] = 0;
	size_t	p = 0;
	size_t	l = 0;
	for (size_t k = 0; k < nz; k++) {
		for (size_t j = 0; j < ny; j++) {
			for (size_t i = 0; i < nx; i++) {
				if (k > 0) {
					d->i[p] = l - nx * ny;
					d->data[p++] = 1.;
				}
				d->i[p] = l;
				d->data[p++] = -1.;
				d->p[++l] = p;
				if (p >= nnz) {
					nnz += n;
					mm_real_realloc (d, nnz);
				}

			}
		}
	}
	// Trim memory if the pre-allocation was an estimate.
	if (p != d->nnz) mm_real_realloc (d, p);
	return d;
}

/**
 * @brief Creates the sparse second-order difference operator D^T*D for the x-axis.
 * @details This creates a discrete Laplacian matrix with a [-1, 2, -1] stencil
 * for interior points, adjusted at the boundaries. It's constructed in CSC format.
 *
 * @return A pointer to the sparse n x n matrix D^T*D.
 */
mm_real *
DiffOp::createLaplacianX (size_t nx, size_t ny, size_t nz)
{
	if (nx <= 0 || ny <= 0 || nz <= 0)
		throw std::invalid_argument("DiffOp::createLaplacianX: Grid dimensions must be positive.");
	size_t	n = nx * ny * nz;
	size_t	nnz = 3 * n;
	mm_real	*d = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, nnz);

	size_t	m = 0;
	size_t	p = 0;
	for (size_t k = 0; k < nz; k++) {
		for (size_t j = 0; j < ny; j++) {
			for (size_t i = 0; i < nx; i++) {
				if (i == 0) {
					d->i[p] = m;
					d->data[p++] = 1.;

					d->i[p] = m + 1;
					d->data[p++] = -1.;
				} else {
					d->i[p] = m - 1;
					d->data[p++] = -1.;

					d->i[p] = m;
					d->data[p++] = 2.;

					if (i < nx - 1) {
						d->i[p] = m + 1;
						d->data[p++] = -1.;
					}
				}
				if (p + 3 >= nnz) {
					nnz += n;
					mm_real_realloc (d, nnz);
				}
				d->p[m + 1] = p;
				m++;
			}
		}
	}
	// Trim memory if the pre-allocation was an estimate.
	if (p != d->nnz) mm_real_realloc (d, p);
	return d;
}

/**
 * @brief Creates the sparse second-order difference operator D^T*D for the y-axis.
 * @details Discrete Laplacian with a stencil applied along the y-axis.
 * @return A pointer to the sparse n x n matrix D^T*D.
 */
mm_real *
DiffOp::createLaplacianY (size_t nx, size_t ny, size_t nz)
{
	if (nx <= 0 || ny <= 0 || nz <= 0)
		throw std::invalid_argument("DiffOp::createLaplacianY: Grid dimensions must be positive.");
	size_t	n = nx * ny * nz;
	size_t	nnz = 3 * n;
	mm_real	*d = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, nnz);

	size_t	m = 0;
	size_t	p = 0;
	for (size_t k = 0; k < nz; k++) {
		for (size_t j = 0; j < ny; j++) {
			for (size_t i = 0; i < nx; i++) {
				if (j == 0) {
					d->i[p] = m;
					d->data[p++] = 1.;

					d->i[p] = m + nx;
					d->data[p++] = -1.;
				} else {
					d->i[p] = m - nx;
					d->data[p++] = -1.;

					d->i[p] = m;
					d->data[p++] = 2.;

					if (j < ny - 1) {
						d->i[p] = m + nx;
						d->data[p++] = -1.;
					}
				}
				if (p + 3 >= nnz) {
					nnz += n;
					mm_real_realloc (d, nnz);
				}
				d->p[m + 1] = p;
				m++;
			}
		}
	}
	// Trim memory if the pre-allocation was an estimate.
	if (p != d->nnz) mm_real_realloc (d, p);
	return d;
}

/**
 * @brief Creates the sparse second-order difference operator D^T*D for the z-axis.
 * @details Discrete Laplacian with a stencil applied along the z-axis.
 * @return A pointer to the sparse n x n matrix D^T*D.
 */
mm_real *
DiffOp::createLaplacianZ (size_t nx, size_t ny, size_t nz)
{
	if (nx <= 0 || ny <= 0 || nz <= 0)
		throw std::invalid_argument("DiffOp::createLaplacianZ: Grid dimensions must be positive.");
	size_t	n = nx * ny * nz;
	size_t	nnz = 3 * n;
	mm_real	*d = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, nnz);

	size_t	m = 0;
	size_t	p = 0;
	for (size_t k = 0; k < nz; k++) {
		for (size_t j = 0; j < ny; j++) {
			for (size_t i = 0; i < nx; i++) {
				if (k == 0) {
					d->i[p] = m;
					d->data[p++] = 1.;

					d->i[p] = m + nx * ny;
					d->data[p++] = -1.;
				} else {
					d->i[p] = m - nx * ny;
					d->data[p++] = -1.;

					d->i[p] = m;
					d->data[p++] = 2.;

					if (k < nz - 1) {
						d->i[p] = m + nx * ny;
						d->data[p++] = -1.;
					}
				}
				if (p + 3 >= nnz) {
					nnz += n;
					mm_real_realloc (d, nnz);
				}
				d->p[m + 1] = p;
				m++;
			}
		}
	}
	// Trim memory if the pre-allocation was an estimate.
	if (p != d->nnz) mm_real_realloc (d, p);
	return d;
}

/*
 * @brief Generates a column-weighted Laplacian in CSC format for the X-axis.
 * @details Calculates K = (Dx * Wx)^T * (Dx * Wx).
 * Wx is a diagonal matrix where `weights[i]` corresponds to the edge weight.
 * Mathematically, the component K_ij = weights[i] * L_ij * weights[j].
 *
 * @param weights Array of size n. weights[m] is the weight between node m and m+1 (X-direction).
 */
mm_real *
DiffOp::createColWeightedLaplacianX (size_t nx, size_t ny, size_t nz, const double *weights)
{
	if (nx <= 0 || ny <= 0 || nz <= 0)
		throw std::invalid_argument("createColWeightedLaplacianX: Grid dimensions must be positive.");
	size_t	n = nx * ny * nz;
	size_t	nnz = 3 * n;
	mm_real	*d = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, nnz);

	size_t	m = 0;
	size_t	p = 0;
	for (size_t k = 0; k < nz; k++) {
		for (size_t j = 0; j < ny; j++) {
			for (size_t i = 0; i < nx; i++) {
				
				// Weight of the current node/edge (m)
				double w_me = 1. / weights[m];

				if (i == 0) {
					// --- Left boundary (Stencil: 1, -1) ---
					double w_right = 1. / weights[m + 1];

					// Diagonal (m, m): 1 * w_me^2
					d->i[p] = m;
					d->data[p++] = 1.0 * w_me * w_me;

					// Right neighbor (m, m+1): -1 * w_me * w_right
					d->i[p] = m + 1;
					d->data[p++] = -1.0 * w_me * w_right;

				} else {
					// --- Middle / Right boundary (Stencil: -1, 2, -1) ---
					double w_left = 1. / weights[m - 1];

					// Left neighbor (m, m-1): -1 * w_me * w_left
					d->i[p] = m - 1;
					d->data[p++] = -1.0 * w_me * w_left;

					// Diagonal (m, m): 2 * w_me^2
					d->i[p] = m;
					d->data[p++] = 2.0 * w_me * w_me;

					if (i < nx - 1) {
						double w_right = 1. / weights[m + 1];
						// Right neighbor (m, m+1): -1 * w_me * w_right
						d->i[p] = m + 1;
						d->data[p++] = -1.0 * w_me * w_right;
					}
				}

				if (p + 3 >= nnz) {
					nnz += n;
					mm_real_realloc (d, nnz);
				}
				d->p[m + 1] = p;
				m++;
			}
		}
	}
	if (p != d->nnz) mm_real_realloc (d, p);
	return d;
}

/*
 * @brief Generates a column-weighted Laplacian in CSC format for the Y-axis.
 * @details Calculates K = (Dy * Wy)^T * (Dy * Wy).
 * Wy is a diagonal matrix where `weights[m]` is the weight associated with node m.
 * Component K_ij = weights[i] * (Dy^T Dy)_ij * weights[j].
 */
mm_real *
DiffOp::createColWeightedLaplacianY (size_t nx, size_t ny, size_t nz, const double *weights)
{
	if (nx <= 0 || ny <= 0 || nz <= 0)
		throw std::invalid_argument("createColWeightedLaplacianY: Grid dimensions must be positive.");
	size_t	n = nx * ny * nz;
	size_t	nnz = 3 * n;
	mm_real	*d = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, nnz);

	size_t	m = 0;
	size_t	p = 0;
	for (size_t k = 0; k < nz; k++) {
		for (size_t j = 0; j < ny; j++) {
			for (size_t i = 0; i < nx; i++) {
				
				// Weight of the current node (m)
				double w_me = 1. / weights[m];

				if (j == 0) {
					// --- Y-direction Start (Bottom boundary) (Stencil: 1, -1) ---
					double w_up = 1. / weights[m + nx]; // Top neighbor (j+1)

					// Diagonal (m, m): 1 * w_me^2
					d->i[p] = m;
					d->data[p++] = 1.0 * w_me * w_me;

					// Top neighbor (m, m+nx): -1 * w_me * w_up
					d->i[p] = m + nx;
					d->data[p++] = -1.0 * w_me * w_up;

				} else {
					// --- Middle / Y-direction End (Top boundary) (Stencil: -1, 2, -1) ---
					double w_down = 1. / weights[m - nx]; // Bottom neighbor (j-1)

					// Bottom neighbor (m, m-nx): -1 * w_me * w_down
					d->i[p] = m - nx;
					d->data[p++] = -1.0 * w_me * w_down;

					// Diagonal (m, m): 2 * w_me^2
					d->i[p] = m;
					d->data[p++] = 2.0 * w_me * w_me;

					if (j < ny - 1) {
						double w_up = 1. / weights[m + nx]; // Top neighbor (j+1)
						// Top neighbor (m, m+nx): -1 * w_me * w_up
						d->i[p] = m + nx;
						d->data[p++] = -1.0 * w_me * w_up;
					}
				}

				if (p + 3 >= nnz) {
					nnz += n;
					mm_real_realloc (d, nnz);
				}
				d->p[m + 1] = p;
				m++;
			}
		}
	}
	if (p != d->nnz) mm_real_realloc (d, p);
	return d;
}

/*
 * @brief Generates a column-weighted Laplacian in CSC format for the Z-axis.
 * @details Calculates K = (Dz * Wz)^T * (Dz * Wz).
 * Wz is a diagonal matrix where `weights[m]` is the weight associated with node m.
 * Component K_ij = weights[i] * (Dz^T Dz)_ij * weights[j].
 */
mm_real *
DiffOp::createColWeightedLaplacianZ (size_t nx, size_t ny, size_t nz, const double *weights)
{
	if (nx <= 0 || ny <= 0 || nz <= 0)
		throw std::invalid_argument("createColWeightedLaplacianZ: Grid dimensions must be positive.");
	size_t	n = nx * ny * nz;
	size_t	nnz = 3 * n;
	mm_real	*d = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, nnz);

	// One step in Z-direction (size of XY plane)
	size_t	slice = nx * ny;

	size_t	m = 0;
	size_t	p = 0;
	for (size_t k = 0; k < nz; k++) {
		for (size_t j = 0; j < ny; j++) {
			for (size_t i = 0; i < nx; i++) {
				
				// Weight of the current node (m)
				double w_me = 1. / weights[m];

				if (k == 0) {
					// --- Z-direction Start (Back boundary) (Stencil: 1, -1) ---
					double w_above = 1. / weights[m + slice]; // Layer above (k+1)

					// Diagonal (m, m): 1 * w_me^2
					d->i[p] = m;
					d->data[p++] = 1.0 * w_me * w_me;

					// Upper neighbor (m, m+slice): -1 * w_me * w_above
					d->i[p] = m + slice;
					d->data[p++] = -1.0 * w_me * w_above;

				} else {
					// --- Middle / Z-direction End (Front boundary) (Stencil: -1, 2, -1) ---
					double w_below = 1. / weights[m - slice]; // Layer below (k-1)

					// Lower neighbor (m, m-slice): -1 * w_me * w_below
					d->i[p] = m - slice;
					d->data[p++] = -1.0 * w_me * w_below;

					// Diagonal (m, m): 2 * w_me^2
					d->i[p] = m;
					d->data[p++] = 2.0 * w_me * w_me;

					if (k < nz - 1) {
						double w_above = 1. / weights[m + slice]; // Layer above (k+1)
						// Upper neighbor (m, m+slice): -1 * w_me * w_above
						d->i[p] = m + slice;
						d->data[p++] = -1.0 * w_me * w_above;
					}
				}

				if (p + 3 >= nnz) {
					nnz += n;
					mm_real_realloc (d, nnz);
				}
				d->p[m + 1] = p;
				m++;
			}
		}
	}
	if (p != d->nnz) mm_real_realloc (d, p);
	return d;
}

#include <vector>
#include <stdexcept>
#include <mmreal.h>

/*
 * Common Specification for Row Weighted Laplacians:
 * Formula: K = (D * W)^T * (D * W)
 * Component: K_ij = weights[i] * L_ij * weights[j]
 * (Where L_ij is the component of the unweighted Laplacian D^T*D)
 * This represents Node-based weighting (symmetric pre/post multiplication).
 */

// ----------------------------------------------------------------------
// X Direction
// ----------------------------------------------------------------------
mm_real *
DiffOp::createRowWeightedLaplacianX (size_t nx, size_t ny, size_t nz, const double *weights)
{
	if (nx <= 0 || ny <= 0 || nz <= 0)
		throw std::invalid_argument("createRowWeightedLaplacianX: Grid dimensions must be positive.");
	size_t	n = nx * ny * nz;
	size_t	nnz = 3 * n;
	mm_real	*d = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, nnz);

	size_t	m = 0;
	size_t	p = 0;
	for (size_t k = 0; k < nz; k++) {
		for (size_t j = 0; j < ny; j++) {
			for (size_t i = 0; i < nx; i++) {
				
				double w_me = 1. / weights[m];

				if (i == 0) {
					// --- Left boundary (Stencil: 1, -1) ---
					double w_right = 1. / weights[m + 1];

					// Diagonal (m, m)
					d->i[p] = m;
					d->data[p++] = 1.0 * w_me * w_me;

					// Right neighbor (m, m+1)
					d->i[p] = m + 1;
					d->data[p++] = -1.0 * w_me * w_right;

				} else {
					// --- Middle / Right boundary (Stencil: -1, 2, -1) ---
					double w_left = 1. / weights[m - 1];

					// Left neighbor (m, m-1)
					d->i[p] = m - 1;
					d->data[p++] = -1.0 * w_me * w_left;

					// Diagonal (m, m)
					d->i[p] = m;
					d->data[p++] = 2.0 * w_me * w_me;

					if (i < nx - 1) {
						double w_right = 1. / weights[m + 1];
						// Right neighbor (m, m+1)
						d->i[p] = m + 1;
						d->data[p++] = -1.0 * w_me * w_right;
					}
				}

				if (p + 3 >= nnz) {
					nnz += n;
					mm_real_realloc (d, nnz);
				}
				d->p[m + 1] = p;
				m++;
			}
		}
	}
	if (p != d->nnz) mm_real_realloc (d, p);
	return d;
}

// ----------------------------------------------------------------------
// Y Direction
// ----------------------------------------------------------------------
mm_real *
DiffOp::createRowWeightedLaplacianY (size_t nx, size_t ny, size_t nz, const double *weights)
{
	if (nx <= 0 || ny <= 0 || nz <= 0)
		throw std::invalid_argument("createRowWeightedLaplacianY: Grid dimensions must be positive.");
	size_t	n = nx * ny * nz;
	size_t	nnz = 3 * n;
	mm_real	*d = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, nnz);

	size_t	m = 0;
	size_t	p = 0;
	for (size_t k = 0; k < nz; k++) {
		for (size_t j = 0; j < ny; j++) {
			for (size_t i = 0; i < nx; i++) {
				
				double w_me = 1. / weights[m];

				if (j == 0) {
					// --- Y Start/Bottom (Stencil: 1, -1) ---
					double w_up = 1. / weights[m + nx]; // j+1

					// Diagonal (m, m)
					d->i[p] = m;
					d->data[p++] = 1.0 * w_me * w_me;

					// Upper neighbor (m, m+nx)
					d->i[p] = m + nx;
					d->data[p++] = -1.0 * w_me * w_up;

				} else {
					// --- Middle / Y End/Top (Stencil: -1, 2, -1) ---
					double w_down = 1. / weights[m - nx]; // j-1

					// Lower neighbor (m, m-nx)
					d->i[p] = m - nx;
					d->data[p++] = -1.0 * w_me * w_down;

					// Diagonal (m, m)
					d->i[p] = m;
					d->data[p++] = 2.0 * w_me * w_me;

					if (j < ny - 1) {
						double w_up = 1. / weights[m + nx]; // j+1
						// Upper neighbor (m, m+nx)
						d->i[p] = m + nx;
						d->data[p++] = -1.0 * w_me * w_up;
					}
				}

				if (p + 3 >= nnz) {
					nnz += n;
					mm_real_realloc (d, nnz);
				}
				d->p[m + 1] = p;
				m++;
			}
		}
	}
	if (p != d->nnz) mm_real_realloc (d, p);
	return d;
}

// ----------------------------------------------------------------------
// Z Direction
// ----------------------------------------------------------------------
mm_real *
DiffOp::createRowWeightedLaplacianZ (size_t nx, size_t ny, size_t nz, const double *weights)
{
	if (nx <= 0 || ny <= 0 || nz <= 0)
		throw std::invalid_argument("createRowWeightedLaplacianZ: Grid dimensions must be positive.");
	size_t	n = nx * ny * nz;
	size_t	nnz = 3 * n;
	mm_real	*d = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, nnz);

	size_t	slice = nx * ny; // Z-direction offset

	size_t	m = 0;
	size_t	p = 0;
	for (size_t k = 0; k < nz; k++) {
		for (size_t j = 0; j < ny; j++) {
			for (size_t i = 0; i < nx; i++) {
				
				double w_me = 1. / weights[m];

				if (k == 0) {
					// --- Z Start/Back (Stencil: 1, -1) ---
					double w_above = 1. / weights[m + slice]; // k+1

					// Diagonal (m, m)
					d->i[p] = m;
					d->data[p++] = 1.0 * w_me * w_me;

					// Upper neighbor (m, m+slice)
					d->i[p] = m + slice;
					d->data[p++] = -1.0 * w_me * w_above;

				} else {
					// --- Middle / Z End/Front (Stencil: -1, 2, -1) ---
					double w_below = 1. / weights[m - slice]; // k-1

					// Lower neighbor (m, m-slice)
					d->i[p] = m - slice;
					d->data[p++] = -1.0 * w_me * w_below;

					// Diagonal (m, m)
					d->i[p] = m;
					d->data[p++] = 2.0 * w_me * w_me;

					if (k < nz - 1) {
						double w_above = 1. / weights[m + slice]; // k+1
						// Upper neighbor (m, m+slice)
						d->i[p] = m + slice;
						d->data[p++] = -1.0 * w_me * w_above;
					}
				}

				if (p + 3 >= nnz) {
					nnz += n;
					mm_real_realloc (d, nnz);
				}
				d->p[m + 1] = p;
				m++;
			}
		}
	}
	if (p != d->nnz) mm_real_realloc (d, p);
	return d;
}


/*** Protected Helper Methods ***/

/**
 * @brief Helper to construct D = [I; Dx; Dy; Dz] via vertical concatenation.
 * @note Manages memory of intermediate matrices by freeing them after use.
 */
mm_real *
DiffOp::buildD0_ (size_t nx, size_t ny, size_t nz)
{
	mm_real	*dx = createDiffX (nx, ny, nz);
	mm_real	*d0 = mm_real_eye (MM_REAL_SPARSE, dx->n);
	mm_real	*tmp1 = mm_real_vertcat (d0, dx);
	mm_real_free (d0);
	mm_real_free (dx);

	mm_real	*dy = createDiffY (nx, ny, nz);
	mm_real	*tmp2 = mm_real_vertcat (tmp1, dy);
	mm_real_free (tmp1);
	mm_real_free (dy);

	mm_real	*dz = createDiffZ (nx, ny, nz);
	mm_real	*D = mm_real_vertcat (tmp2, dz);
	mm_real_free (tmp2);
	mm_real_free (dz);

	return D;
}

/**
 * @brief Helper to construct D = [Dx; Dy; Dz] via vertical concatenation.
 * @note Manages memory of intermediate matrices by freeing them after use.
 */
mm_real *
DiffOp::buildD1_ (size_t nx, size_t ny, size_t nz)
{
	mm_real	*dx = createDiffX (nx, ny, nz);
	mm_real	*dy = createDiffY (nx, ny, nz);
	mm_real	*tmp = mm_real_vertcat (dx, dy);
	mm_real_free (dx);
	mm_real_free (dy);

	mm_real	*dz = createDiffZ (nx, ny, nz);
	mm_real	*D = mm_real_vertcat (tmp, dz);
	mm_real_free (tmp);
	mm_real_free (dz);

	return D;
}

/**
 * @brief Adds three Laplacian matrices (L = Lx + Ly + Lz) using OpenMP.
 * * This function performs a parallelized 3-way merge sort on the CSC columns.
 * It is generally faster than calling MKL's add function twice because it
 * avoids creating intermediate matrices (Kernel Fusion) and minimizes memory bandwidth usage.
 *
 * @param uplo Matrix symmetry flag ('U', 'L', or 'G').
 * @param Lx First Laplacian matrix (Sorted CSC format).
 * @param Ly Second Laplacian matrix (Sorted CSC format).
 * @param Lz Third Laplacian matrix (Sorted CSC format).
 * @return mm_real* Resulting matrix L (Sorted CSC format). The caller must free this.
 */
mm_real *
DiffOp::addLaplacians (const mm_real* Lx, const mm_real* Ly, const mm_real* Lz)
{
	if (!Lx || !Ly || !Lz) return nullptr;
	
	// Assuming matrices are square and have the same dimensions
	size_t n = Lx->n;

	// To parallelize the construction of a CSR/CSC matrix, we need two passes:
	// Pass 1: Calculate the number of non-zero elements (NNZ) per column.
	// Pass 2: Fill the values into the pre-allocated memory.

	// Buffer to store NNZ count for each column
	std::vector<size_t> col_nnz(n);

	// =====================================================================
	// PASS 1: Count NNZ per column (Parallelized)
	// =====================================================================
#pragma omp parallel for
	for (size_t j = 0; j < n; ++j) {
		// Get column boundaries for each matrix
		size_t px = Lx->p[j], end_x = Lx->p[j+1];
		size_t py = Ly->p[j], end_y = Ly->p[j+1];
		size_t pz = Lz->p[j], end_z = Lz->p[j+1];
		
		size_t count = 0;

		// Merge sort logic to count unique row indices
		while (px < end_x || py < end_y || pz < end_z) {
			// Use SIZE_MAX (or equivalent) as a sentinel for exhausted columns
			size_t row_x = (px < end_x) ? Lx->i[px] : std::numeric_limits<size_t>::max();
			size_t row_y = (py < end_y) ? Ly->i[py] : std::numeric_limits<size_t>::max();
			size_t row_z = (pz < end_z) ? Lz->i[pz] : std::numeric_limits<size_t>::max();
			
			// Find the minimum row index among the three
			size_t min_row = row_x;
			if (row_y < min_row) min_row = row_y;
			if (row_z < min_row) min_row = row_z;
			
			// Increment count for this unique row index
			count++;
			
			// Advance pointers for matrices containing the min_row
			if (row_x == min_row) px++;
			if (row_y == min_row) py++;
			if (row_z == min_row) pz++;
		}
		col_nnz[j] = count;
	}

	// =====================================================================
	// INTERMEDIATE: Construct column pointers 'p' (Prefix Sum)
	// =====================================================================
	// We need to determine the total NNZ to allocate memory.
	// This part is sequential but very fast O(N).
	
	// Note: We cannot allocate 'L' yet because we need total_nnz.
	// Let's create a temporary p array first.
	std::vector<size_t> p_array(n + 1);
	p_array[0] = 0;
	for (size_t j = 0; j < n; ++j) {
		p_array[j+1] = p_array[j] + col_nnz[j];
	}
	size_t total_nnz = p_array[n];

	// Allocate the result matrix
	mm_real* L = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, total_nnz);

	// Copy the computed column pointers into the matrix structure
	// (Assuming mm_real->p is compatible with size_t or we cast loop-wise)
	for (size_t j = 0; j <= n; ++j) {
		L->p[j] = p_array[j];
	}

	// =====================================================================
	// PASS 2: Fill indices and values (Parallelized)
	// =====================================================================
#pragma omp parallel for
	for (size_t j = 0; j < n; ++j) {
		// Start position for writing in the result matrix
		size_t current_p = L->p[j];
		
		size_t px = Lx->p[j], end_x = Lx->p[j+1];
		size_t py = Ly->p[j], end_y = Ly->p[j+1];
		size_t pz = Lz->p[j], end_z = Lz->p[j+1];

		// Merge sort logic again to fill data
		while (px < end_x || py < end_y || pz < end_z) {
			size_t row_x = (px < end_x) ? Lx->i[px] : std::numeric_limits<size_t>::max();
			size_t row_y = (py < end_y) ? Ly->i[py] : std::numeric_limits<size_t>::max();
			size_t row_z = (pz < end_z) ? Lz->i[pz] : std::numeric_limits<size_t>::max();

			size_t min_row = row_x;
			if (row_y < min_row) min_row = row_y;
			if (row_z < min_row) min_row = row_z;

			double sum_val = 0.0;
			
			// Accumulate values for the current row index
			if (row_x == min_row) sum_val += Lx->data[px++];
			if (row_y == min_row) sum_val += Ly->data[py++];
			if (row_z == min_row) sum_val += Lz->data[pz++];

			// Write to the result matrix
			// Since we scan in ascending order, L->i is guaranteed to be sorted.
			L->i[current_p] = min_row;
			L->data[current_p] = sum_val;
			current_p++;
		}
	}

	return L;
}
