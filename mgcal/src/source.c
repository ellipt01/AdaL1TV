/*
 * source.c
 *
 *  Created on: 2015/03/14
 *      Author: utsugi
 *
 *  Minimal version: set_position / set_dimension / set_magnetization removed
 *  (kernel_matrix_set writes pos/dim directly via vector3d_set).
 */

#include <stdlib.h>

#include "vector3d.h"
#include "util.h"
#include "source.h"

static source_item *
source_item_alloc (void)
{
	source_item	*item = (source_item *) malloc (sizeof (source_item));
	if (!item) error_and_exit ("source_item_alloc", "failed to allocate source_item.", __FILE__, __LINE__);
	item->mgz = NULL;
	item->pos = NULL;
	item->dim = NULL;
	item->next = NULL;
	return item;
}

static void
source_item_free (source_item *si)
{
	if (si) {
		if (si->mgz) free (si->mgz);
		if (si->pos) free (si->pos);
		if (si->dim) free (si->dim);
		free (si);
	}
	return;
}

static source *
source_alloc (void)
{
	source	*src = (source *) malloc (sizeof (source));
	if (!src) error_and_exit ("source_alloc", "failed to allocate source.", __FILE__, __LINE__);
	src->exf = NULL;
	src->item = source_item_alloc ();
	src->begin = NULL;
	src->end = NULL;
	return src;
}

/*
 * source_new - create an empty source (external field NULL, no magnetized items).
 * Items are added with source_append_item; free the whole source with source_free.
 */
source *
source_new (void)
{
	source	*src = source_alloc ();
	return src;
}

/*
 * source_free - free a source and every item in its list, plus the external field.
 */
void
source_free (source *src)
{
	if (src) {
		source_item	*cur;
		vector3d_free (src->exf);
		cur = src->item;
		while (cur) {
			source_item	*next = cur->next;
			source_item_free (cur);
			cur = next;
		}
		free (src);
	}
	return;
}

/*
 * source_set_external - set the external field direction from inclination/declination.
 * Builds a unit vector via the geodesic-polar convention and stores it as src->exf.
 */
void
source_set_external (source *src, const double inc, const double dec)
{
	src->exf = vector3d_new_with_geodesic_poler (1., inc, dec);
	return;
}

/*
 * source_append_item - append a new (empty) magnetized item to the source list.
 * Sets src->begin on the first append and updates src->end. Returns the index
 * of the newly appended item. Fill in pos/dim/mgz directly after appending.
 */
int
source_append_item (source *src)
{
	int			i = 0;
	source_item	*cur = src->item;
	while (cur->next) {
		cur = cur->next;
		i++;
	}
	cur->next = source_item_alloc ();
	if (!src->begin) src->begin = src->item->next;
	src->end = cur->next;
	return i;
}
