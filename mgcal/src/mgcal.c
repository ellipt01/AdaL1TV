/*
 * mgcal.c
 *
 *  Created on: 2015/04/17
 *      Author: utsugi
 */

#include <stdio.h>
#include "mgcal.h"

/* mgcal_set_scale_factor - set the global output scale applied to all field values. */
void
mgcal_set_scale_factor (const double val)
{
	scale_factor = val;
	return;
}

/* mgcal_get_scale_factor - return the current global output scale factor. */
double
mgcal_get_scale_factor (void)
{
	return scale_factor;
}
