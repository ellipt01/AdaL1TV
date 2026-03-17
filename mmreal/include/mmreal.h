/**
 * @file mm_real.h
 * @brief Header file for real-valued matrix operations in MatrixMarket format.
 * @author utsugi
 * @date 2014/06/20
 * @version 2.1 (Reorganized 2025)
 */

#ifndef MMREAL_H_
#define MMREAL_H_

#include <stdio.h>
#include <mmio.h>
#include <stdbool.h>
#include <threads.h> // C11 standard header for thread_local alias

#ifdef __cplusplus
extern "C" {
#endif

/* Aliases for matrix types */
typedef struct s_mm_real mm_real;
typedef struct s_mm_real mm_dense;
typedef struct s_mm_real mm_sparse;

/* Storage format: dense or sparse */
typedef enum {
	MM_REAL_SPARSE = 0,	// sparse matrix (CSC format)
	MM_REAL_DENSE  = 1	// dense matrix (column-major)
} MMRealFormat;

/* Symmetry properties */
enum {
	MM_GENERAL   = 1 << 0, // General, asymmetric
	MM_SYMMETRIC = 1 << 1  // Symmetric
};
enum {
	MM_UPPER = 1 << 2, // Upper triangular part is stored
	MM_LOWER = 1 << 3  // Lower triangular part is stored
};
typedef enum {
	MM_REAL_GENERAL = MM_GENERAL,
	MM_REAL_SYMMETRIC_UPPER = MM_SYMMETRIC | MM_UPPER,
	MM_REAL_SYMMETRIC_LOWER = MM_SYMMETRIC | MM_LOWER
} MMRealSymm;


/* The core matrix structure */
struct s_mm_real {
	MM_typecode	typecode; // MatrixMarket typecode
	MMRealSymm		symm;     // Symmetry property

	MM_INT		m;    // Number of rows
	MM_INT		n;    // Number of columns
	MM_INT		nnz;  // Number of non-zero elements (or m*n for dense)

	// Sparse format data (CSC: Compressed Sparse Column)
	MM_INT		*i; // Row indices array (size = nnz)
	MM_INT		*p; // Column pointers array (size = n + 1)
	
	// Data array for both sparse (non-zeros) and dense (all elements)
	MM_DBL		*data; // Values array (size = nnz)

	bool			owner; // True if this struct owns and should free the data arrays
};

/**
 * @brief Defines the result codes for library functions.
 */
typedef enum {
    /* --- Success --- */
    MM_SUCCESS = 0,                 // Operation was successful.

    /* --- General Errors --- */
    MM_ERROR_UNKNOWN = -1,          // An unknown or unspecified error occurred.
    MM_ERROR_NOT_IMPLEMENTED = -2,  // The requested feature is not yet implemented.

    /* --- Argument and Input Errors --- */
    MM_ERROR_NULL_ARGUMENT = -10,       // A required pointer argument was NULL.
    MM_ERROR_INVALID_ARGUMENT = -11,    // The value of an argument was invalid.
    MM_ERROR_DIMENSION_MISMATCH = -12,  // Matrix or vector dimensions were incompatible.
    MM_ERROR_INDEX_OUT_OF_BOUNDS = -13, // An index (i, j) was out of range.
    MM_ERROR_FORMAT_MISMATCH = -14,     // Matrix format (sparse/dense) was incorrect for the operation.

    /* --- Resource and System Errors --- */
    MM_ERROR_ALLOCATION_FAILED = -20, // Memory allocation failed.
    MM_ERROR_FILE_IO = -21,           // A file I/O operation failed.

} MMResult;

/**
 * @brief The thread-local variable to store the last error code.
 *
 * Each thread will have its own independent copy of this variable.
 * It's declared 'static' to be private to this implementation file.
 */
static thread_local MMResult g_last_error = MM_SUCCESS;

/* --- Type checking macros --- */
#define mm_real_is_sparse(a)    mm_is_sparse((a)->typecode)
#define mm_real_is_dense(a)     mm_is_dense((a)->typecode)
#define mm_real_is_symmetric(a) (mm_is_symmetric((a)->typecode) && ((a)->symm & MM_SYMMETRIC))
#define mm_real_is_upper(a)     ((a)->symm & MM_UPPER)
#define mm_real_is_lower(a)     ((a)->symm & MM_LOWER)


/*============================================================================
 * Function Prototypes
 *============================================================================*/

/* --- 0. support program --- */
MMResult	mm_real_get_last_error ();
const char	*mm_result_to_string (MMResult res);

/* --- 1. Creation, Destruction, and Copying --- */
mm_real	*mm_real_new (MMRealFormat format, MMRealSymm symm, MM_INT m, MM_INT n, MM_INT nnz);
void		mm_real_free (mm_real *mm);
mm_real	*mm_real_copy (const mm_real *mm);
mm_real	*mm_real_eye (MMRealFormat type, MM_INT n);
mm_real	*mm_real_view_array (MMRealFormat format, MMRealSymm symm, MM_INT m, MM_INT n, MM_INT nnz, MM_DBL *data);
bool		mm_real_realloc (mm_real *mm, MM_INT nnz);
bool		mm_real_resize (mm_real *x, MM_INT m, MM_INT n, MM_INT nnz, bool do_realloc);

/* --- 2. Element Access and Manipulation --- */
MM_DBL	mm_real_get (const mm_real *x, MM_INT i, MM_INT j);
bool		mm_real_set (mm_real *x, MM_INT i, MM_INT j, MM_DBL val);
bool		mm_real_set_all (mm_real *mm, MM_DBL val);
bool		mm_real_memcpy (mm_real *dest, const mm_real *src);
bool		mm_real_transpose (mm_real *x);
bool		mm_real_sort (mm_real *x);

/* --- 3. Format and Type Conversion --- */
bool		mm_real_sparse_to_dense (mm_sparse *s);
bool		mm_real_dense_to_sparse (mm_dense *d, MM_DBL threshold);
bool		mm_real_symmetric_to_general (mm_real *x);
bool		mm_real_general_to_symmetric (char uplo, mm_real *x);
mm_dense	*mm_real_copy_sparse_to_dense (const mm_sparse *s);
mm_sparse	*mm_real_copy_dense_to_sparse (const mm_dense *x, MM_DBL threshold);

/* --- 4. Matrix Assembly and Extraction --- */
mm_real	*mm_real_vertcat (const mm_real *x1, const mm_real *x2);
mm_real	*mm_real_horzcat (const mm_real *x1, const mm_real *x2);
bool		mm_real_xj_col_to (mm_real *xj, const mm_real *x, MM_INT j);
mm_dense	*mm_real_xj_col (const mm_real *x, MM_INT j);
bool		mm_real_xi_row_to (mm_real *xi, const mm_real *x, MM_INT i);
mm_dense	*mm_real_xi_row (const mm_real *x, MM_INT i);

/* --- 5. Linear Algebra: AXPY-like Operations --- */
bool		mm_real_axpy (MM_DBL alpha, const mm_real *x, mm_real *y);
bool		mm_real_axjpy (MM_DBL alpha, const mm_real *x, MM_INT j, mm_dense *y);
bool		mm_real_scale (mm_real *x, MM_DBL alpha);
bool		mm_real_xj_scale (mm_real *x, MM_INT j, MM_DBL alpha);
bool		mm_real_add (mm_real *x, MM_DBL alpha);
bool		mm_real_xj_add (mm_real *x, MM_INT j, MM_DBL alpha);

/* --- 6. Linear Algebra: Products --- */
MM_DBL	mm_real_dot (const mm_real *x, const mm_real *y);
bool		mm_real_x_dot_y (bool transx, bool transy, MM_DBL alpha, const mm_real *x, const mm_real *y, MM_DBL beta, mm_real *z);
bool		mm_real_x_dot_yk (bool trans, MM_DBL alpha, const mm_real *x, const mm_real *y, MM_INT k, MM_DBL beta, mm_dense *z);
bool		mm_real_xj_trans_dot_y_to (mm_real *dest, const mm_real *x, MM_INT j, const mm_real *y);
mm_dense	*mm_real_xj_trans_dot_y (const mm_real *x, MM_INT j, const mm_real *y);
MM_DBL	mm_real_xj_trans_dot_yk (const mm_real *x, MM_INT j, const mm_real *y, MM_INT k);

/* --- 7. Vector / Column Statistics --- */
MM_INT	mm_real_iamax (const mm_real *x);
MM_INT	mm_real_xj_iamax (const mm_real *x, MM_INT j);
MM_DBL	mm_real_xj_asum (const mm_real *x, MM_INT j);
MM_DBL	mm_real_xj_sum (const mm_real *x, MM_INT j);
MM_DBL	mm_real_xj_nrm2 (const mm_real *x, MM_INT j);
MM_DBL	mm_real_xj_ssq (const mm_real *x, MM_INT j);
MM_DBL	mm_real_xj_mean (const mm_real *x, MM_INT j);
MM_DBL	mm_real_xj_std (const mm_real *x, MM_INT j);

/* --- 8. File I/O --- */
mm_real	*mm_real_fread (FILE *fp);
bool		mm_real_fwrite (FILE *stream, const mm_real *x, const char *format);
bool		mm_real_fprintf (FILE *stream, const mm_real *a, const char *format, char delim);
mm_real	*mm_real_fread_binary (FILE *fp);
bool		mm_real_fwrite_binary (FILE *fp, const mm_real *x);


#ifdef __cplusplus
}
#endif

#endif /* MMREAL_H_ */
