/*
 * calc.c
 *
 *  Created on: 2015/03/14
 *      Author: utsugi
 *
 *  Minimal version: prism (3D) only.
 *  dipole and yz (2D) routines removed.
 */

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include "vector3d.h"
#include "util.h"
#include "source.h"
#include "calc.h"

double scale_factor = 100.;

#define SIGN(a) ((a) < 0. ? -1. : +1.)

/*
 * prism_kernel - magnetic field of a half-space corner at relative position (x,y,z).
 * Evaluates the Bhattacharyya analytic kernel for a single prism vertex.
 * Returns zero contribution when the observation point coincides with the vertex (r=0).
 */
static void
prism_kernel (vector3d *f, const double x, const double y, const double z, const vector3d *mgz)
{
	double	fx, fy, fz;
	double	lnx, lny, lnz;

	double	r = sqrt (x * x + y * y + z * z);

	/* observation point coincides with a prism vertex: no contribution */
	if (r < DBL_EPSILON) {
		vector3d_set (f, 0., 0., 0.);
		return;
	}

	lnx = (fabs (r + x) > DBL_EPSILON) ? log (r + x) : - log (r - x);
	lny = (fabs (r + y) > DBL_EPSILON) ? log (r + y) : - log (r - y);
	lnz = (fabs (r + z) > DBL_EPSILON) ? log (r + z) : - log (r - z);

	{
		double	jx = mgz->x;
		double	jy = mgz->y;
		double	jz = mgz->z;

		fx = - jx * atan2 (y * z, x * r)
			+ jy * lnz
			+ jz * lny;

		fy = jx * lnz
			- jy * atan2 (x * z, y * r)
			+ jz * lnx;

		fz = jx * lny
			+ jy * lnx
			- jz * atan2 (x * y, z * r);
	}

	vector3d_set (f, fx, fy, fz);
	return;
}

/*
 * prism - total magnetic field produced by all prism sources in the source list.
 * Sums the signed 8-vertex contributions of each rectangular prism and applies
 * the global scale_factor. Returns a newly allocated vector3d (caller frees).
 */
vector3d *
prism (const vector3d *obs, const source *s)
{
	double		a[2], b[2], c[2];
	double		x, y, z;
	double		x0, y0, z0;
	vector3d		*f;
	vector3d		tmp[8];
	source_item	*cur;

	if (!obs) error_and_exit ("prism", "vector3d *obs is empty.", __FILE__, __LINE__);
	if (!s) error_and_exit ("prism", "source *s is empty.", __FILE__, __LINE__);

	x0 = obs->x;
	y0 = obs->y;
	z0 = obs->z;

	f = vector3d_new (0., 0., 0.);

	cur = s->begin;
	while (cur) {
		double	dx, dy, dz;
		double	flag;

		if (!cur->pos) error_and_exit ("prism", "position of source item is empty.", __FILE__, __LINE__);
		if (!cur->dim) error_and_exit ("prism", "dimension of source item is empty.", __FILE__, __LINE__);
		if (!cur->mgz) error_and_exit ("prism", "magnetization of source item is empty.", __FILE__, __LINE__);

		dx = cur->dim->x;
		dy = cur->dim->y;
		dz = cur->dim->z;
		flag = SIGN (dx) * SIGN (dy) * SIGN (dz);

		x = cur->pos->x;
		y = cur->pos->y;
		z = cur->pos->z;

		a[0] = x - 0.5 * dx - x0;
		b[0] = y - 0.5 * dy - y0;
		c[0] = z - 0.5 * dz - z0;

		a[1] = a[0] + dx;
		b[1] = b[0] + dy;
		c[1] = c[0] + dz;

		prism_kernel (&tmp[0], a[1], b[1], c[1], cur->mgz);
		prism_kernel (&tmp[2], a[1], b[0], c[1], cur->mgz);
		prism_kernel (&tmp[4], a[0], b[1], c[1], cur->mgz);
		prism_kernel (&tmp[6], a[0], b[0], c[1], cur->mgz);

		if (fabs (dz) < DBL_EPSILON) {
			vector3d_set (&tmp[1], 0., 0., 0.);
			vector3d_set (&tmp[3], 0., 0., 0.);
			vector3d_set (&tmp[5], 0., 0., 0.);
			vector3d_set (&tmp[7], 0., 0., 0.);
		} else {
			prism_kernel (&tmp[1], a[1], b[1], c[0], cur->mgz);
			prism_kernel (&tmp[3], a[1], b[0], c[0], cur->mgz);
			prism_kernel (&tmp[5], a[0], b[1], c[0], cur->mgz);
			prism_kernel (&tmp[7], a[0], b[0], c[0], cur->mgz);
		}

		f->x += flag * (tmp[0].x - tmp[1].x - tmp[2].x + tmp[3].x
			- tmp[4].x + tmp[5].x + tmp[6].x - tmp[7].x);

		f->y += flag * (tmp[0].y - tmp[1].y - tmp[2].y + tmp[3].y
			- tmp[4].y + tmp[5].y + tmp[6].y - tmp[7].y);

		f->z += flag * (tmp[0].z - tmp[1].z - tmp[2].z + tmp[3].z
			- tmp[4].z + tmp[5].z + tmp[6].z - tmp[7].z);

		cur = cur->next;
	}
	vector3d_scale (f, scale_factor);
	return f;
}

/*
 * total_force - projection of field vector f onto the external field direction exf.
 * Returns the total-force anomaly (dot product exf . f).
 */
double
total_force (const vector3d *exf, const vector3d *f)
{
	if (exf == NULL) error_and_exit ("total_force", "vector3d *exf is empty.", __FILE__, __LINE__);
	if (f == NULL) error_and_exit ("total_force", "vector3d *f is empty.", __FILE__, __LINE__);
	return exf->x * f->x + exf->y * f->y + exf->z * f->z;
}

/*
 * calc_component - extract the requested component (x/y/z or total force) from f.
 */
static double
calc_component (const vector3d *f, const source *src, MgcalComponent comp)
{
	double	val = 0.;
	switch (comp) {
	case MGCAL_X_COMPONENT:
		val = f->x;
		break;

	case MGCAL_Y_COMPONENT:
		val = f->y;
		break;

	case MGCAL_Z_COMPONENT:
		val = f->z;
		break;

	case MGCAL_TOTAL_FORCE:
		val = total_force (src->exf, f);
		break;

	}
	return val;
}

/*** prism ***/
/*
 * component_prism - evaluate prism field at obs and return the chosen component.
 * Internal helper shared by the x/y/z/total-force entry points below.
 */
static double
component_prism (const vector3d *obs, const source *src, MgcalComponent comp)
{
	vector3d	*f = prism (obs, src);
	double	val = calc_component (f, src, comp);
	vector3d_free (f);
	return val;
}

/* x_component_prism - mgcal_func callback: x-component of the prism field. */
double
x_component_prism (const vector3d *obs, const source *src, void *data)
{
	return component_prism (obs, src, MGCAL_X_COMPONENT);
}

/* y_component_prism - mgcal_func callback: y-component of the prism field. */
double
y_component_prism (const vector3d *obs, const source *src, void *data)
{
	return component_prism (obs, src, MGCAL_Y_COMPONENT);
}

/* z_component_prism - mgcal_func callback: z-component of the prism field. */
double
z_component_prism (const vector3d *obs, const source *src, void *data)
{
	return component_prism (obs, src, MGCAL_Z_COMPONENT);
}

/* total_force_prism - mgcal_func callback: total-force anomaly of the prism field. */
double
total_force_prism (const vector3d *obs, const source *src, void *data)
{
	return component_prism (obs, src, MGCAL_TOTAL_FORCE);
}
