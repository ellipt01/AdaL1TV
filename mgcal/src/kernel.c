/*
 * kernel.c
 *
 *  Created on: 2015/03/15
 *      Author: utsugi
 *
 *  Minimal version: kernel_matrix_set / kernel_matrix only.
 *  scattered, jth_col, ith_row routines removed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "vector3d.h"
#include "source.h"
#include "data_array.h"
#include "grid.h"
#include "kernel.h"
#include "util.h"

static mgcal_func *
mgcal_func_alloc (void)
{
	mgcal_func	*f = (mgcal_func *) malloc (sizeof (mgcal_func));
	if (!f) error_and_exit ("mgcal_func_alloc", "failed to allocate mgcal_func.", __FILE__, __LINE__);
	f->function = NULL;
	f->parameter = NULL;
	return f;
}

/*
 * mgcal_func_new - bundle a theoretical-field callback with its user parameter.
 * Returns a newly allocated mgcal_func (free with mgcal_func_free).
 * Note: ownership of *data stays with the caller; mgcal_func_free does not touch it.
 */
mgcal_func *
mgcal_func_new (const mgcal_theoretical func, void *data)
{
	mgcal_func	*f = mgcal_func_alloc ();
	f->function = func;
	f->parameter = data;
	return f;
}

/* mgcal_func_free - free an mgcal_func (does not free the user parameter). */
void
mgcal_func_free (mgcal_func *f)
{
	if (f) free (f);
	return;
}

/*
 * kernel_matrix_set - fill a preallocated sensitivity matrix for a structured grid.
 *
 * Computes a[l + j*m] = f(obs_l, cell_j) for every observation point (m total)
 * and every grid cell (g->n total), using the field callback f with fixed
 * magnetization mgz and external field exf. Column-major in the cell index:
 * the j-th grid cell occupies a contiguous block of m entries.
 *
 * Parallelized over the z-layers with OpenMP; each thread owns a private
 * obs/source pair. The caller must allocate a with room for m * g->n doubles.
 * If g->z1 is set, the irregular surface offset is added per horizontal cell.
 */
void
kernel_matrix_set (double *a, const data_array *array, const grid *g, const vector3d *mgz, const vector3d *exf, const mgcal_func *f)
{
	int		m;
	int		nx;
	int		ny;
	int		nz;
	int		nh;

	if (!a) error_and_exit ("kernel_matrix_set", "double *a is empty.", __FILE__, __LINE__);

	m = array->n;
	nx = g->nx;
	ny = g->ny;
	nz = g->nz;
	nh = g->nh;

#pragma omp parallel
	{
		size_t		i, j, k, l;
		double		*z1 = NULL;
		vector3d	*obs = vector3d_new (0., 0., 0.);
		source		*src = source_new ();
		if (exf) src->exf = vector3d_copy (exf);
		source_append_item (src);
		src->begin->pos = vector3d_new (0., 0., 0.);
		src->begin->dim = vector3d_new (0., 0., 0.);
		if (mgz) src->begin->mgz = vector3d_copy (mgz);

#pragma omp for
		for (k = 0; k < nz; k++) {
			double			*zk = g->z + k;	// for parallel calculation
			double			*dzk = g->dz + k;	// for parallel calculation
			double			*yj = g->y;
			double			*dyj = g->dy;
			unsigned long	offsetk = ((unsigned long) k) * ((unsigned long) nh) * ((unsigned long) m);
			double			*ak = a + offsetk;
			for (j = 0; j < ny; j++) {
				double			*xi = g->x;
				double			*dxi = g->dx;
				unsigned long	offsetj = ((unsigned long) j) * ((unsigned long) nx) * ((unsigned long) m);
				double			*aj = ak + offsetj;
				if (g->z1) z1 = g->z1 + j * nx;
				for (i = 0; i < nx; i++) {
					double	*al = aj + i * m;
					double	*xl = array->x;
					double	*yl = array->y;
					double	*zl = array->z;
					double	z1k = *zk;
					if (z1) z1k += z1[i];
					vector3d_set (src->begin->pos, *xi, *yj, z1k);
					vector3d_set (src->begin->dim, *dxi, *dyj, *dzk);
					for (l = 0; l < m; l++) {
						vector3d_set (obs, *xl, *yl, *zl);
						*al = f->function (obs, src, f->parameter);
						al++;
						xl++;
						yl++;
						zl++;
					}
					xi++;
					dxi++;
				}
				yj++;
				dyj++;
			}
		}
		vector3d_free (obs);
		source_free (src);
	}
	return;
}

/*
 * kernel_matrix - allocate and fill a sensitivity matrix (convenience wrapper).
 * Allocates m * g->n doubles, calls kernel_matrix_set, and returns the buffer.
 * Caller frees the returned pointer with free().
 */
double *
kernel_matrix (const data_array *array, const grid *g, const vector3d *mgz, const vector3d *exf, const mgcal_func *f)
{
	int				m, n;
	double			*a;
	unsigned long	size;

	m = array->n;
	n = g->n;
	size = ((unsigned long) m) * ((unsigned long) n);
	a = (double *) malloc (size * sizeof (double));
	if (!a) error_and_exit ("kernel_matrix", "failed to allocate memory of *a.", __FILE__, __LINE__);
	kernel_matrix_set (a, array, g, mgz, exf, f);
	return a;
}
