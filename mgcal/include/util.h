/*
 * util.h
 *
 *  Internal utilities for the mgcal library (not part of the public API).
 *  Provides error reporting and small array helpers used across the sources.
 */

#ifndef UTIL_H_
#define UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Print a diagnostic (function, message, file, line) to stderr and exit. */
void	error_and_exit (const char *func, const char *msg, const char *file, const int line);

/* Copy n doubles from src to dest. Returns 1 on success, 0 if either is NULL. */
int		array_copy (const int n, double *dest, const double *src);

/* Set all n elements of x to val. */
void	array_set_all (const int n, double *x, const double val);

/* Store the pair (a, b) into r[0], r[1]. */
void	set_range (double r[], const double a, const double b);

#ifdef __cplusplus
}
#endif

#endif /* UTIL_H_ */
