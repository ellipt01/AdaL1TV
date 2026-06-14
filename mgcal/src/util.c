/*
 * util.c
 *
 *  Internal utilities for the mgcal library (not part of the public API).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"

/*
 * error_and_exit - print a diagnostic identifying the failing function, a
 * message, and the source location, then terminate the program.
 */
void
error_and_exit (const char *func, const char *msg, const char *file, const int line)
{
	fprintf (stderr, "ERROR [%s] %s (%s:%d)\n",
		(func ? func : "?"), (msg ? msg : "?"), (file ? file : "?"), line);
	exit (EXIT_FAILURE);
}

/*
 * array_copy - copy n doubles from src into dest.
 * Returns 1 on success, 0 if either pointer is NULL.
 */
int
array_copy (const int n, double *dest, const double *src)
{
	if (!dest || !src) return 0;
	if (n > 0) memcpy (dest, src, (size_t) n * sizeof (double));
	return 1;
}

/* array_set_all - assign val to all n elements of x. */
void
array_set_all (const int n, double *x, const double val)
{
	int	i;
	if (!x) return;
	for (i = 0; i < n; i++) x[i] = val;
}

/* set_range - store the pair (a, b) into r[0], r[1]. */
void
set_range (double r[], const double a, const double b)
{
	r[0] = a;
	r[1] = b;
}
