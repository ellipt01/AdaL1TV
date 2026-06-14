/*
 * io.c
 *
 *  Created on: 2015/03/14
 *      Author: utsugi
 *
 *  Hardened version:
 *   - sprintf -> snprintf
 *   - sscanf / strtod return values checked
 *   - read_one_line takes a capacity limit (no buffer overrun)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "vector3d.h"
#include "data_array.h"
#include "grid.h"
#include "util.h"

typedef struct s_datalist	datalist;

struct s_datalist {
	double		x;
	double		y;
	double		z;
	double		data;
	datalist	*next;
};

static datalist *
datalist_alloc (void)
{
	datalist	*list = (datalist *) malloc (sizeof (datalist));
	if (!list) error_and_exit ("datalist_alloc", "failed to allocate datalist.", __FILE__, __LINE__);
	list->next = NULL;
	return list;
}

static datalist *
datalist_push_back (datalist *list, const double x, const double y, const double z, const double data)
{
	datalist	*p = list;
	while (p->next) p = p->next;
	p->next = datalist_alloc ();
	p = p->next;
	p->x = x;
	p->y = y;
	p->z = z;
	p->data = data;
	return p;
}

static datalist *
fread_datalist (FILE *stream, int *n)
{
	char		buf[BUFSIZ];
	datalist	*list = datalist_alloc ();
	datalist	*pp = list;
	int			count = 0;

	while (fgets (buf, BUFSIZ, stream) != NULL) {
		double	x, y, z, data;
		char	*p = buf;
		if (p[0] == '#' || p[0] == '\n') continue;
		while (p[0] == ' ' || p[0] == '\t') p++;
		/* require all four fields; skip malformed lines */
		if (sscanf (p, "%lf %lf %lf %lf", &x, &y, &z, &data) != 4) continue;
		pp = datalist_push_back (pp, x, y, z, data);
		count++;
	}
	*n = count;
	return list;
}

static void
datalist_free (datalist *list)
{
	datalist	*cur = list;
	while (cur) {
		datalist	*p = cur;
		cur = cur->next;
		free (p);
	}
	return;
}

/*
 * fread_data_array - read observation data (x y z value per line) from a stream.
 * Lines starting with '#' and malformed lines (fewer than 4 fields) are skipped.
 * Returns a newly allocated data_array (free with data_array_free).
 */
data_array *
fread_data_array (FILE *stream)
{
	data_array	*array;
	int			n, k;
	datalist	*list, *cur, *prev;

	list = fread_datalist (stream, &n);
	array = data_array_new (n);

	k = 0;
	prev = list;
	cur = list->next;
	while (cur) {
		array->x[k] = cur->x;
		array->y[k] = cur->y;
		array->z[k] = cur->z;
		array->data[k] = cur->data;
		free (prev);
		prev = cur;
		cur = cur->next;
		if (++k >= n) break;
	}
	if (cur) datalist_free (cur);
	else if (prev) free (prev);
	return array;
}

/*
 * fwrite_data_array_with_data - write x/y/z from array paired with an external
 * data vector, one record per line, using the given printf format (4 fields).
 */
void
fwrite_data_array_with_data (FILE *stream, const data_array *array, const double *data, const char *format)
{
	int		i;
	char	fm[BUFSIZ];
	if (!format) strcpy (fm, "%f %f %f %f\n");
	else snprintf (fm, sizeof (fm), "%s\n", format);
	for (i = 0; i < array->n; i++) fprintf (stream, fm, array->x[i], array->y[i], array->z[i], data[i]);
	return;
}

/* fwrite_data_array - write a data_array using its own stored data values. */
void
fwrite_data_array (FILE *stream, const data_array *array, const char *format)
{
	fwrite_data_array_with_data (stream, array, array->data, format);
	return;
}

static void
fprintf_array (FILE *stream, const int noneline, const int n, const double *array, const char *format)
{
	int		i;
	for (i = 0; i < n; i++) {
		fprintf (stream, format, array[i]);
		if (i < n - 1) {
			if ((i + 1) % noneline == 0) fprintf (stream, "\n");
			else fprintf (stream, " ");
		}
	}
	fprintf (stream, "\n");
	return;
}

static bool
zero_range (const double r[])
{
	return (fabs (r[1] - r[0]) < DBL_EPSILON);
}

static bool
is_grid_valid (const grid *g)
{
	if (!g) return false;
	if (g->nx <= 0 || g->ny <= 0 || g->nz <= 0) return false;
	if (g->nh <= 0 || g->n <= 0) return false;
	if (zero_range (g->xrange) && zero_range (g->yrange) && zero_range (g->zrange)) return false;
	if (!g->x) return false;
	if (!g->dx) return false;
	if (!g->y) return false;
	if (!g->dy) return false;
	if (!g->z) return false;
	if (!g->dz) return false;
	return true;
}

static char *
skip_blanks (char *buf)
{
	char	*p = buf;
	while (p[0] == ' ' || p[0] == '\t' || p[0] == '\r') p++;
	return p;
}

static bool
is_valid_line (char *p)
{
	if (p == NULL) return false;
	if (p[0] == '#' || p[0] == '\n') return false;
	return true;
}

static char	linebuf[BUFSIZ];

static char *
get_valid_line_body (FILE *stream)
{
	char	*p = NULL;
	while (1) {
		if (fgets (linebuf, BUFSIZ, stream) == NULL) return NULL;
		p = skip_blanks (linebuf);
		if (!is_valid_line (p)) continue;
		break;
	}
	return p;
}

/* parse up to cap doubles from buf; returns number actually stored */
static int
read_one_line (char *buf, double *x, int cap)
{
	int		i;
	char	*p;
	char	*endptr;
	if (!buf) return 0;
	for (i = 0, p = strtok (buf, " \t"); p && i < cap; p = strtok (NULL, " \t")) {
		double	v;
		if (p[0] == '\n' || p[0] == '\r') continue;
		v = strtod (p, &endptr);
		if (endptr == p) continue;	/* not a number: skip token */
		x[i++] = v;
	}
	return i;
}

const char *valname[] = {"x", "dx", "y", "dy", "z", "dz"};

/*
 * fread_grid - parse a grid definition (dimensions, corner positions, per-axis
 * coordinate and spacing arrays, optional z1 surface) from a stream. Comment ('#')
 * and blank lines are ignored. Returns a newly allocated grid (free with grid_free).
 */
grid *
fread_grid (FILE *stream)
{
	int		i, k;
	char	*p;
	grid	*g;

	g = (grid *) malloc (sizeof (grid));
	if (!g) error_and_exit ("fread_grid", "failed to allocate grid.", __FILE__, __LINE__);
	g->x = g->y = g->z = g->z1 = NULL;
	g->dx = g->dy = g->dz = NULL;
	g->data = NULL;

	// read dimensions
	p = get_valid_line_body (stream);
	if (!p) error_and_exit ("fread_grid", "cannot read grid dimension.", __FILE__, __LINE__);
	if (sscanf (p, "%d %d %d", &g->nx, &g->ny, &g->nz) != 3)
		error_and_exit ("fread_grid", "cannot parse grid dimension.", __FILE__, __LINE__);
	if (g->nx <= 0 || g->ny <= 0 || g->nz <= 0)
		error_and_exit ("fread_grid", "invalid grid dimension.", __FILE__, __LINE__);
	g->nh = g->nx * g->ny;
	g->n = g->nh * g->nz;

	// read positions
	p = get_valid_line_body (stream);
	if (!p) error_and_exit ("fread_grid", "read pos0: entry is empty.", __FILE__, __LINE__);
	if (sscanf (p, "%lf %lf %lf", &g->xrange[0], &g->yrange[0], &g->zrange[0]) != 3)
		error_and_exit ("fread_grid", "cannot parse pos0.", __FILE__, __LINE__);
	p = get_valid_line_body (stream);
	if (!p) error_and_exit ("fread_grid", "read pos1: entry is empty.", __FILE__, __LINE__);
	if (sscanf (p, "%lf %lf %lf", &g->xrange[1], &g->yrange[1], &g->zrange[1]) != 3)
		error_and_exit ("fread_grid", "cannot parse pos1.", __FILE__, __LINE__);

	for (k = 0; k <= 5; k++) {
		int		n = -1;
		double	*val = NULL;
		switch (k) {
			case 0:
				g->x = (double *) malloc (g->nx * sizeof (double));
				n = g->nx; val = g->x; break;
			case 1:
				g->dx = (double *) malloc (g->nx * sizeof (double));
				n = g->nx; val = g->dx; break;
			case 2:
				g->y = (double *) malloc (g->ny * sizeof (double));
				n = g->ny; val = g->y; break;
			case 3:
				g->dy = (double *) malloc (g->ny * sizeof (double));
				n = g->ny; val = g->dy; break;
			case 4:
				g->z = (double *) malloc (g->nz * sizeof (double));
				n = g->nz; val = g->z; break;
			case 5:
				g->dz = (double *) malloc (g->nz * sizeof (double));
				n = g->nz; val = g->dz; break;
			default:
				break;
		}
		if (!val) error_and_exit ("fread_grid", "failed to allocate grid array.", __FILE__, __LINE__);
		i = 0;
		while (i < n) {
			char	*q = get_valid_line_body (stream);
			if (q == NULL) break;
			i += read_one_line (q, val + i, n - i);
		}
		if (i != n) {
			char	msg[80];
			snprintf (msg, sizeof (msg), "size of %s is mismatch.", valname[k]);
			error_and_exit ("fread_grid", msg, __FILE__, __LINE__);
		}
	}
	// read z1 (optional surface topography)
	i = 0;
	while (i < g->nh) {
		char	*q = get_valid_line_body (stream);
		if (q == NULL) break;
		if (!g->z1) g->z1 = (double *) malloc (g->nh * sizeof (double));
		i += read_one_line (q, g->z1 + i, g->nh - i);
	}
	if (i > 0 && i != g->nh) error_and_exit ("fread_grid", "size of z1 is mismatch.", __FILE__, __LINE__);

	if (!is_grid_valid (g)) error_and_exit ("fread_grid", "cannot read grid correctly.", __FILE__, __LINE__);

	return g;
}

const int	n_oneline = 10;

/* fwrite_grid - write a grid definition in the format readable by fread_grid. */
void
fwrite_grid (FILE *stream, const grid *g)
{
	if (!is_grid_valid (g)) error_and_exit ("fwrite_grid", "grid is not valid.", __FILE__, __LINE__);

	fprintf (stream, "### GRID DATA ###\n");

	fprintf (stream, "# [NX, NY, NZ] : dimension\n");
	fprintf (stream, "%d %d %d\n\n", g->nx, g->ny, g->nz);

	fprintf (stream, "# P0 = [X0, Y0, Z0] : South-West (left-bottom) position\n");
	fprintf (stream, "%f %f %f\n\n", g->xrange[0], g->yrange[0], g->zrange[0]);

	fprintf (stream, "# P1 = [X1, Y1, Z1] : North-East (right-top) position\n");
	fprintf (stream, "%f %f %f\n\n", g->xrange[1], g->yrange[1], g->zrange[1]);

	fprintf (stream, "# X\n");
	fprintf_array (stream, n_oneline, g->nx, g->x, "%f");

	fprintf (stream, "# DX\n");
	fprintf_array (stream, n_oneline, g->nx, g->dx, "%f");
	fprintf (stream, "\n");

	fprintf (stream, "# Y\n");
	fprintf_array (stream, n_oneline, g->ny, g->y, "%f");

	fprintf (stream, "# DY\n");
	fprintf_array (stream, n_oneline, g->ny, g->dy, "%f");
	fprintf (stream, "\n");

	fprintf (stream, "# Z\n");
	fprintf_array (stream, n_oneline, g->nz, g->z, "%f");

	fprintf (stream, "# DZ\n");
	fprintf_array (stream, n_oneline, g->nz, g->dz, "%f");
	fprintf (stream, "\n");

	if (g->z1) {
		fprintf (stream, "# Z1 : surface topography\n");
		fprintf_array (stream, n_oneline, g->nh, g->z1, "%f");
		fprintf (stream, "\n");
	}

	return;
}

/* fwrite_grid_to_xyz - write the center coordinate of every grid cell as x y z. */
void
fwrite_grid_to_xyz (FILE *stream, const grid *g, const char *format)
{
	int			n;
	vector3d	*pos;
	char		fm[BUFSIZ];

	if (!g) error_and_exit ("fwrite_grid", "grid is empty.", __FILE__, __LINE__);

	if (!format) strcpy (fm, "%f %f %f\n");
	else snprintf (fm, sizeof (fm), "%s\n", format);

	pos = vector3d_new (0., 0., 0.);
	for (n = 0; n < g->n; n++) {
		grid_get_nth (g, n, pos, NULL);
		fprintf (stream, fm, pos->x, pos->y, pos->z);
	}
	vector3d_free (pos);
	return;
}

/*
 * fwrite_grid_with_data - write each cell center paired with a per-cell data value
 * (x y z value). Passing NULL for data writes zeros. Used to export inversion models.
 */
void
fwrite_grid_with_data (FILE *stream, const grid *g, const double *data, const char *format)
{
	int			n;
	vector3d	*pos;
	char		fm[BUFSIZ];

	if (!g) error_and_exit ("fwrite_grid", "grid is empty.", __FILE__, __LINE__);

	if (!format) strcpy (fm, "%f %f %f %f\n");
	else snprintf (fm, sizeof (fm), "%s\n", format);

	pos = vector3d_new (0., 0., 0.);
	for (n = 0; n < g->n; n++) {
		double	val = (data) ? data[n] : 0.;
		grid_get_nth (g, n, pos, NULL);
		fprintf (stream, fm, pos->x, pos->y, pos->z, val);
	}
	vector3d_free (pos);
	return;
}
