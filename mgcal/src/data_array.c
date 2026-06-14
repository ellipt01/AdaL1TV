/*
 * data_array.c
 *
 *  Created on: 2015/03/14
 *      Author: utsugi
 *
 *  Minimal version: data_array_ith_copy / data_array_copy removed.
 */

#include <stdlib.h>
#include "data_array.h"
#include "util.h"

static data_array *
data_array_alloc (void)
{
	data_array	*array = (data_array *) malloc (sizeof (data_array));
	if (!array) error_and_exit ("data_array_alloc", "failed to allocate data_array.", __FILE__, __LINE__);
	array->n = 0;
	array->x = NULL;
	array->y = NULL;
	array->z = NULL;
	array->data = NULL;
	return array;
}

/*
 * data_array_new - allocate a data_array for n observation points.
 * x/y/z/data arrays are zero-initialized. Free with data_array_free.
 */
data_array *
data_array_new (const int n)
{
	data_array	*array = data_array_alloc ();

	array->n = n;
	array->x = (double *) malloc (n * sizeof (double));
	array->y = (double *) malloc (n * sizeof (double));
	array->z = (double *) malloc (n * sizeof (double));
	array->data = (double *) malloc (n * sizeof (double));
	if (!array->x || !array->y || !array->z || !array->data)
		error_and_exit ("data_array_new", "failed to allocate arrays.", __FILE__, __LINE__);
	array_set_all (n, array->x, 0.);
	array_set_all (n, array->y, 0.);
	array_set_all (n, array->z, 0.);
	array_set_all (n, array->data, 0.);
	return array;
}

/* data_array_free - free a data_array and its x/y/z/data buffers (NULL-safe). */
void
data_array_free (data_array *array)
{
	if (array) {
		if (array->x) free (array->x);
		if (array->y) free (array->y);
		if (array->z) free (array->z);
		if (array->data) free (array->data);
		free (array);
	}
	return;
}
