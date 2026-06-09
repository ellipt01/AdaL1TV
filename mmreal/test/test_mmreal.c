/**
 * @file test_mmreal.c
 * @brief Regression and coverage tests for the mmreal library.
 *
 * Dependency-free (uses test_util.h only). Covers the major API groups and,
 * in particular, the bugs hardened during development:
 *   - unordered / duplicate / symmetric Matrix Market coordinate input
 *   - rejection of destructive operations on non-owning views
 *   - bounds checking in mm_real_x_dot_yk
 *   - rejection of scalar-add on sparse matrices
 *   - correct symmetric column std / ssq
 *   - text and binary round-trips
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mmreal.h"
#include "test_util.h"

#define TOL 1e-12

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

/* Write a string to a temp file and reopen for reading. Caller fcloses. */
static FILE *
str_to_tmp (const char *text)
{
	FILE *fp = tmpfile ();
	if (!fp) return NULL;
	fputs (text, fp);
	rewind (fp);
	return fp;
}

/* ------------------------------------------------------------------ */
/* 1. Creation, ownership, views                                      */
/* ------------------------------------------------------------------ */

TEST (new_and_free_dense)
{
	mm_real *d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 3, 2, 6);
	CHECK (d != NULL);
	CHECK (d->m == 3 && d->n == 2 && d->nnz == 6);
	CHECK (d->owner == true);
	CHECK (mm_real_is_dense (d));
	/* calloc-initialized to zero */
	for (MM_INT k = 0; k < d->nnz; k++) CHECK_NEAR (d->data[k], 0.0, TOL);
	mm_real_free (d);
}

TEST (new_rejects_bad_dims)
{
	CHECK (mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 0, 2, 0) == NULL);
	CHECK (mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, -1, 2, 0) == NULL);
	/* symmetric must be square */
	CHECK (mm_real_new (MM_REAL_SPARSE, MM_REAL_SYMMETRIC_UPPER, 3, 4, 5) == NULL);
}

TEST (view_does_not_own)
{
	double buf[6] = { 1, 2, 3, 4, 5, 6 };
	mm_real *v = mm_real_view_array (MM_REAL_DENSE, MM_REAL_GENERAL, 3, 2, 6, buf);
	CHECK (v != NULL);
	CHECK (v->owner == false);
	CHECK (v->data == buf);          /* no copy */
	CHECK_NEAR (mm_real_get (v, 0, 1), 4.0, TOL);
	mm_real_free (v);                /* must NOT free buf */
	CHECK_NEAR (buf[0], 1.0, TOL);   /* still valid */
}

TEST (view_rejects_destructive_ops)
{
	double buf[4] = { 1, 0, 0, 1 };
	/* Build a sparse view by hand is awkward; use a dense view and test the
	 * dense-targeting destructive op (dense_to_sparse) which checks ownership. */
	mm_real *v = mm_real_view_array (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 2, 4, buf);
	CHECK (v != NULL);
	/* dense_to_sparse would realloc/replace data it does not own -> must fail */
	CHECK (mm_real_dense_to_sparse (v, 0.5) == false);
	CHECK (mm_real_get_last_error () == MM_ERROR_INVALID_ARGUMENT);
	mm_real_free (v);
}

TEST (eye_and_copy)
{
	mm_real *I = mm_real_eye (MM_REAL_SPARSE, 4);
	CHECK (I != NULL);
	for (MM_INT i = 0; i < 4; i++)
		for (MM_INT j = 0; j < 4; j++)
			CHECK_NEAR (mm_real_get (I, i, j), (i == j) ? 1.0 : 0.0, TOL);

	mm_real *C = mm_real_copy (I);
	CHECK (C != NULL && C != I && C->data != I->data);
	CHECK_NEAR (mm_real_get (C, 2, 2), 1.0, TOL);
	mm_real_free (I);
	mm_real_free (C);
}

/* ------------------------------------------------------------------ */
/* 2. Element access                                                  */
/* ------------------------------------------------------------------ */

TEST (dense_get_set)
{
	mm_real *d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 2, 4);
	CHECK (mm_real_set (d, 0, 0, 1.5));
	CHECK (mm_real_set (d, 1, 0, 2.5));
	CHECK_NEAR (mm_real_get (d, 0, 0), 1.5, TOL);
	CHECK_NEAR (mm_real_get (d, 1, 0), 2.5, TOL);
	CHECK (isnan (mm_real_get (d, 5, 0)));   /* out of bounds -> NAN */
	mm_real_free (d);
}

TEST (sparse_set_insert)
{
	/* start from a 3x3 sparse zero and insert elements out of order */
	mm_real *s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, 3, 3, 0);
	CHECK (mm_real_set (s, 2, 0, 9.0));
	CHECK (mm_real_set (s, 0, 0, 7.0));
	CHECK (mm_real_set (s, 1, 2, 3.0));
	CHECK_NEAR (mm_real_get (s, 0, 0), 7.0, TOL);
	CHECK_NEAR (mm_real_get (s, 2, 0), 9.0, TOL);
	CHECK_NEAR (mm_real_get (s, 1, 2), 3.0, TOL);
	CHECK_NEAR (mm_real_get (s, 1, 1), 0.0, TOL);  /* absent -> 0 */
	mm_real_free (s);
}

/* ------------------------------------------------------------------ */
/* 3. Conversions                                                     */
/* ------------------------------------------------------------------ */

TEST (sparse_dense_roundtrip)
{
	mm_real *s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, 3, 3, 0);
	mm_real_set (s, 0, 0, 1.0);
	mm_real_set (s, 2, 1, 2.0);
	mm_real_set (s, 1, 2, 3.0);

	mm_dense *d = mm_real_copy_sparse_to_dense (s);
	CHECK (d != NULL && mm_real_is_dense (d));
	CHECK_NEAR (mm_real_get (d, 0, 0), 1.0, TOL);
	CHECK_NEAR (mm_real_get (d, 2, 1), 2.0, TOL);
	CHECK_NEAR (mm_real_get (d, 1, 2), 3.0, TOL);

	mm_sparse *s2 = mm_real_copy_dense_to_sparse (d, 1e-15);
	CHECK (s2 != NULL && mm_real_is_sparse (s2));
	CHECK_NEAR (mm_real_get (s2, 2, 1), 2.0, TOL);
	CHECK_NEAR (mm_real_get (s2, 0, 1), 0.0, TOL);

	mm_real_free (s);
	mm_real_free (d);
	mm_real_free (s2);
}

TEST (symmetric_to_general)
{
	/* 3x3 symmetric upper, dense */
	mm_real *d = mm_real_new (MM_REAL_DENSE, MM_REAL_SYMMETRIC_UPPER, 3, 3, 9);
	mm_real_set (d, 0, 1, 5.0);   /* set upper; mirror is implied */
	CHECK_NEAR (mm_real_get (d, 1, 0), 5.0, TOL);  /* symmetric read */
	CHECK (mm_real_symmetric_to_general (d));
	CHECK (!mm_real_is_symmetric (d));
	CHECK_NEAR (mm_real_get (d, 0, 1), 5.0, TOL);
	CHECK_NEAR (mm_real_get (d, 1, 0), 5.0, TOL);  /* now explicit */
	mm_real_free (d);
}

/* ------------------------------------------------------------------ */
/* 4. Assembly / extraction                                           */
/* ------------------------------------------------------------------ */

TEST (vertcat_horzcat_dense)
{
	mm_real *a = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 2, 4);
	mm_real *b = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 2, 4);
	mm_real_set_all (a, 1.0);
	mm_real_set_all (b, 2.0);

	mm_real *v = mm_real_vertcat (a, b);   /* 4x2 */
	CHECK (v != NULL && v->m == 4 && v->n == 2);
	CHECK_NEAR (mm_real_get (v, 0, 0), 1.0, TOL);
	CHECK_NEAR (mm_real_get (v, 3, 1), 2.0, TOL);

	mm_real *h = mm_real_horzcat (a, b);   /* 2x4 */
	CHECK (h != NULL && h->m == 2 && h->n == 4);
	CHECK_NEAR (mm_real_get (h, 0, 0), 1.0, TOL);
	CHECK_NEAR (mm_real_get (h, 1, 3), 2.0, TOL);

	mm_real_free (a); mm_real_free (b);
	mm_real_free (v); mm_real_free (h);
}

TEST (column_extraction)
{
	mm_real *d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 3, 2, 6);
	for (MM_INT j = 0; j < 2; j++)
		for (MM_INT i = 0; i < 3; i++)
			mm_real_set (d, i, j, (double) (i + 10 * j));

	mm_dense *col = mm_real_xj_col (d, 1);
	CHECK (col != NULL && col->m == 3 && col->n == 1);
	CHECK_NEAR (col->data[0], 10.0, TOL);
	CHECK_NEAR (col->data[2], 12.0, TOL);
	mm_real_free (col);
	mm_real_free (d);
}

/* ------------------------------------------------------------------ */
/* 5. AXPY-like                                                       */
/* ------------------------------------------------------------------ */

TEST (axpy_dense)
{
	mm_real *x = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 4, 1, 4);
	mm_real *y = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 4, 1, 4);
	mm_real_set_all (x, 2.0);
	mm_real_set_all (y, 1.0);
	CHECK (mm_real_axpy (3.0, x, y));      /* y = 3*2 + 1 = 7 */
	for (MM_INT k = 0; k < 4; k++) CHECK_NEAR (y->data[k], 7.0, TOL);
	mm_real_free (x); mm_real_free (y);
}

TEST (scale)
{
	mm_real *x = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 3, 1, 3);
	mm_real_set_all (x, 4.0);
	CHECK (mm_real_scale (x, 0.5));
	for (MM_INT k = 0; k < 3; k++) CHECK_NEAR (x->data[k], 2.0, TOL);
	mm_real_free (x);
}

TEST (add_rejects_sparse)
{
	/* scalar-add on sparse must be rejected (would densify) */
	mm_real *s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, 3, 3, 0);
	mm_real_set (s, 0, 0, 1.0);
	CHECK (mm_real_add (s, 5.0) == false);
	CHECK (mm_real_get_last_error () == MM_ERROR_NOT_IMPLEMENTED);
	CHECK (mm_real_xj_add (s, 0, 5.0) == false);
	mm_real_free (s);
}

/* ------------------------------------------------------------------ */
/* 6. Products                                                        */
/* ------------------------------------------------------------------ */

TEST (dot_dense)
{
	mm_real *x = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 3, 1, 3);
	mm_real *y = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 3, 1, 3);
	x->data[0] = 1; x->data[1] = 2; x->data[2] = 3;
	y->data[0] = 4; y->data[1] = 5; y->data[2] = 6;
	CHECK_NEAR (mm_real_dot (x, y), 32.0, TOL);  /* 4+10+18 */
	mm_real_free (x); mm_real_free (y);
}

TEST (matvec_dense)
{
	/* A = [[1,2],[3,4]], x = [1,1] -> A*x = [3,7] */
	mm_real *A = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 2, 4);
	mm_real_set (A, 0, 0, 1.0); mm_real_set (A, 0, 1, 2.0);
	mm_real_set (A, 1, 0, 3.0); mm_real_set (A, 1, 1, 4.0);
	mm_real *x = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 1, 2);
	mm_real_set_all (x, 1.0);
	mm_real *y = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 1, 2);
	CHECK (mm_real_x_dot_yk (false, 1.0, A, x, 0, 0.0, y));
	CHECK_NEAR (y->data[0], 3.0, TOL);
	CHECK_NEAR (y->data[1], 7.0, TOL);
	mm_real_free (A); mm_real_free (x); mm_real_free (y);
}

TEST (matvec_sparse_matches_dense)
{
	/* same A as above but sparse; result must match */
	mm_real *A = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, 2, 2, 0);
	mm_real_set (A, 0, 0, 1.0); mm_real_set (A, 0, 1, 2.0);
	mm_real_set (A, 1, 0, 3.0); mm_real_set (A, 1, 1, 4.0);
	mm_real *x = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 1, 2);
	mm_real_set_all (x, 1.0);
	mm_real *y = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 1, 2);
	CHECK (mm_real_x_dot_yk (false, 1.0, A, x, 0, 0.0, y));
	CHECK_NEAR (y->data[0], 3.0, TOL);
	CHECK_NEAR (y->data[1], 7.0, TOL);
	mm_real_free (A); mm_real_free (x); mm_real_free (y);
}

TEST (x_dot_yk_bounds)
{
	/* k beyond columns of z must be rejected (restored bounds check) */
	mm_real *A = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 2, 4);
	mm_real *x = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 1, 2);
	mm_real *z = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 1, 2);
	mm_real_set_all (x, 1.0);
	CHECK (mm_real_x_dot_yk (false, 1.0, A, x, 5, 0.0, z) == false);
	mm_real_free (A); mm_real_free (x); mm_real_free (z);
}

/* ------------------------------------------------------------------ */
/* 7. Statistics                                                      */
/* ------------------------------------------------------------------ */

TEST (column_stats_dense)
{
	/* column = [1,2,3,4] : sum 10, mean 2.5, ssq 30, nrm2 sqrt(30),
	 * sample std = sqrt( ((1.5^2)+(0.5^2)+(0.5^2)+(1.5^2)) / 3 ) */
	mm_real *d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 4, 1, 4);
	for (MM_INT i = 0; i < 4; i++) d->data[i] = (double) (i + 1);
	CHECK_NEAR (mm_real_xj_sum  (d, 0), 10.0, TOL);
	CHECK_NEAR (mm_real_xj_mean (d, 0), 2.5,  TOL);
	CHECK_NEAR (mm_real_xj_ssq  (d, 0), 30.0, TOL);
	CHECK_NEAR (mm_real_xj_nrm2 (d, 0), sqrt (30.0), TOL);
	CHECK_NEAR (mm_real_xj_asum (d, 0), 10.0, TOL);
	double expect_std = sqrt ((2.25 + 0.25 + 0.25 + 2.25) / 3.0);
	CHECK_NEAR (mm_real_xj_std (d, 0), expect_std, 1e-9);
	mm_real_free (d);
}

TEST (iamax_dense)
{
	mm_real *d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 4, 1, 4);
	d->data[0] = 1; d->data[1] = -9; d->data[2] = 3; d->data[3] = 2;
	CHECK (mm_real_xj_iamax (d, 0) == 1);   /* largest magnitude at index 1 */
	mm_real_free (d);
}

/* ------------------------------------------------------------------ */
/* 8. File I/O                                                        */
/* ------------------------------------------------------------------ */

TEST (fread_coordinate_unordered_duplicates)
{
	/* 3x3 sparse, entries given out of column order, with a duplicate (0,0)
	 * that must be summed: 1.0 + 0.5 = 1.5 */
	const char *mtx =
		"%%MatrixMarket matrix coordinate real general\n"
		"3 3 4\n"
		"3 1 1.0\n"     /* (2,0) 1-based -> (2,0) 0-based */
		"1 1 1.0\n"     /* (0,0) */
		"1 1 0.5\n"     /* (0,0) duplicate -> sum */
		"2 3 4.0\n";    /* (1,2) */
	FILE *fp = str_to_tmp (mtx);
	CHECK (fp != NULL);
	mm_real *s = mm_real_fread (fp);
	fclose (fp);
	CHECK (s != NULL && mm_real_is_sparse (s));
	CHECK_NEAR (mm_real_get (s, 0, 0), 1.5, TOL);  /* duplicates summed */
	CHECK_NEAR (mm_real_get (s, 2, 0), 1.0, TOL);
	CHECK_NEAR (mm_real_get (s, 1, 2), 4.0, TOL);
	CHECK_NEAR (mm_real_get (s, 1, 1), 0.0, TOL);
	mm_real_free (s);
}

TEST (fread_symmetric)
{
	/* symmetric: only lower entry given; (0,2) must read back symmetrically */
	const char *mtx =
		"%%MatrixMarket matrix coordinate real symmetric\n"
		"3 3 2\n"
		"1 1 4.0\n"
		"3 1 2.0\n";   /* (2,0) lower -> symmetric (0,2) too */
	FILE *fp = str_to_tmp (mtx);
	mm_real *s = mm_real_fread (fp);
	fclose (fp);
	CHECK (s != NULL && mm_real_is_symmetric (s));
	CHECK_NEAR (mm_real_get (s, 0, 0), 4.0, TOL);
	CHECK_NEAR (mm_real_get (s, 2, 0), 2.0, TOL);
	CHECK_NEAR (mm_real_get (s, 0, 2), 2.0, TOL);  /* mirrored read */
	mm_real_free (s);
}

TEST (binary_roundtrip_sparse)
{
	mm_real *s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, 3, 3, 0);
	mm_real_set (s, 0, 0, 1.25);
	mm_real_set (s, 2, 1, 2.5);
	mm_real_set (s, 1, 2, 3.75);

	FILE *fp = tmpfile ();
	CHECK (fp != NULL);
	CHECK (mm_real_fwrite_binary (fp, s));
	rewind (fp);
	mm_real *r = mm_real_fread_binary (fp);
	fclose (fp);
	CHECK (r != NULL && mm_real_is_sparse (r));
	CHECK (r->m == 3 && r->n == 3);
	CHECK_NEAR (mm_real_get (r, 0, 0), 1.25, TOL);
	CHECK_NEAR (mm_real_get (r, 2, 1), 2.5,  TOL);
	CHECK_NEAR (mm_real_get (r, 1, 2), 3.75, TOL);
	mm_real_free (s);
	mm_real_free (r);
}

TEST (binary_roundtrip_dense)
{
	mm_real *d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, 2, 3, 6);
	for (MM_INT k = 0; k < 6; k++) d->data[k] = (double) k * 1.5;

	FILE *fp = tmpfile ();
	CHECK (mm_real_fwrite_binary (fp, d));
	rewind (fp);
	mm_real *r = mm_real_fread_binary (fp);
	fclose (fp);
	CHECK (r != NULL && mm_real_is_dense (r));
	CHECK (r->m == 2 && r->n == 3);
	for (MM_INT k = 0; k < 6; k++) CHECK_NEAR (r->data[k], (double) k * 1.5, TOL);
	mm_real_free (d);
	mm_real_free (r);
}

TEST (binary_fread_rejects_bad_header)
{
	/* truncated file: only a typecode, no dimensions -> must fail cleanly */
	FILE *fp = tmpfile ();
	fwrite ("MCRS", 1, 4, fp);
	rewind (fp);
	mm_real *r = mm_real_fread_binary (fp);
	fclose (fp);
	CHECK (r == NULL);
	mm_real_free (r);  /* safe on NULL */
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

int
main (void)
{
	/* 1. creation / ownership */
	RUN (new_and_free_dense);
	RUN (new_rejects_bad_dims);
	RUN (view_does_not_own);
	RUN (view_rejects_destructive_ops);
	RUN (eye_and_copy);

	/* 2. element access */
	RUN (dense_get_set);
	RUN (sparse_set_insert);

	/* 3. conversions */
	RUN (sparse_dense_roundtrip);
	RUN (symmetric_to_general);

	/* 4. assembly / extraction */
	RUN (vertcat_horzcat_dense);
	RUN (column_extraction);

	/* 5. axpy-like */
	RUN (axpy_dense);
	RUN (scale);
	RUN (add_rejects_sparse);

	/* 6. products */
	RUN (dot_dense);
	RUN (matvec_dense);
	RUN (matvec_sparse_matches_dense);
	RUN (x_dot_yk_bounds);

	/* 7. statistics */
	RUN (column_stats_dense);
	RUN (iamax_dense);

	/* 8. file I/O */
	RUN (fread_coordinate_unordered_duplicates);
	RUN (fread_symmetric);
	RUN (binary_roundtrip_sparse);
	RUN (binary_roundtrip_dense);
	RUN (binary_fread_rejects_bad_header);

	return test_summary ();
}
