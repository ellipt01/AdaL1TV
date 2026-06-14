/*
 * vector3d.c
 *
 *  Created on: 2015/03/14
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include "vector3d.h"
#include "util.h"

static vector3d *
vector3d_alloc (void)
{
	vector3d	*cv = (vector3d *) malloc (sizeof (vector3d));
	if (!cv) error_and_exit ("vector3d_alloc", "failed to allocate vector3d.", __FILE__, __LINE__);
	cv->x = 0.;
	cv->y = 0.;
	cv->z = 0.;
	return cv;
}

/* vector3d_new - allocate a vector3d with the given Cartesian components. */
vector3d *
vector3d_new (const double x, const double y, const double z)
{
	vector3d	*cv = vector3d_alloc ();
	cv->x = x;
	cv->y = y;
	cv->z = z;
	return cv;
}

/*
 * vector3d_new_with_geodesic_poler - build a vector from magnitude r and
 * inclination/declination (degrees). Uses the geomagnetic convention:
 * x=r*cos(inc)*sin(dec), y=r*cos(inc)*cos(dec), z=-r*sin(inc).
 */
vector3d *
vector3d_new_with_geodesic_poler (const double r, const double inc, const double dec)
{
	vector3d	*cv = vector3d_alloc ();
	double	t = deg2rad (inc);
	double	p = deg2rad (dec);
	cv->x = r * cos (t) * sin (p);
	cv->y = r * cos (t) * cos (p);
	cv->z = - r * sin (t);
	return cv;
}

/* vector3d_set - overwrite the components of an existing vector3d in place. */
void
vector3d_set (vector3d *cv, const double x, const double y, const double z)
{
	if (!cv) error_and_exit ("vector3d_set", "vector3d *cv is empty.", __FILE__, __LINE__);
	cv->x = x;
	cv->y = y;
	cv->z = z;
	return;
}

/* vector3d_free - free a vector3d (NULL-safe). */
void
vector3d_free (vector3d *cv)
{
	if (cv) free (cv);
	return;
}

/* vector3d_scale - multiply all components by alpha in place. */
void
vector3d_scale (vector3d *x, const double alpha)
{
	x->x *= alpha;
	x->y *= alpha;
	x->z *= alpha;
	return;
}

/* vector3d_copy - allocate a new vector3d with the same components as src. */
vector3d *
vector3d_copy (const vector3d *src)
{
	vector3d	*dist;
	if (!src) error_and_exit ("vector3d_copy", "vector3d *src is empty.", __FILE__, __LINE__);
	dist = vector3d_alloc ();
	vector3d_set (dist, src->x, src->y, src->z);
	return dist;
}

/* vector3d_axpy - in-place y = y + alpha * x (BLAS axpy for 3-vectors). */
void
vector3d_axpy (const double alpha, const vector3d *x, vector3d *y)
{
	if (!x) error_and_exit ("vector3d_axpy", "vector3d *x is empty.", __FILE__, __LINE__);
	if (!y) error_and_exit ("vector3d_axpy", "vector3d *y is empty.", __FILE__, __LINE__);
	y->x += alpha * x->x;
	y->y += alpha * x->y;
	y->z += alpha * x->z;
	return;
}

/* vector3d_dot - dot product x . y. */
double
vector3d_dot (const vector3d *x, const vector3d *y)
{
	return x->x * y->x + x->y * y->y + x->z * y->z;
}

/* vector3d_nrm - Euclidean norm |c|. */
double
vector3d_nrm (const vector3d *c)
{
	return sqrt (vector3d_dot (c, c));
}
