#ifndef BLAS_INTERFACE_HEADER
#define BLAS_INTERFACE_HEADER

#ifdef __cplusplus
extern "C"{
#endif

#include <limits.h>
#include <stdint.h>
#if defined (MM_BLAS_ILP64)
  typedef int64_t blas_int;
  #define BLAS_INT_MAX  INT64_MAX
#else
  typedef int     blas_int;
  #define BLAS_INT_MAX  INT_MAX
#endif

//Structs
typedef struct complex_Tag{
  float r;
  float i;
} complex;

typedef struct doublecomplex_Tag{
  double r;
  double i;
} doublecomplex;


//blas_int xrebla_(const char *srname, const blas_int *info);

//Level1

//AXPY
void saxpy_(const blas_int *n, const float          *alpha, const float         *x, const blas_int *incx, float         *y, const blas_int *incy);
void daxpy_(const blas_int *n, const double        *alpha, const double        *x, const blas_int *incx, double        *y, const blas_int *incy);
void caxpy_(const blas_int *n, const complex       *alpha, const complex       *x, const blas_int *incx, complex       *y, const blas_int *incy);
void zaxpy_(const blas_int *n, const doublecomplex *alpha, const doublecomplex *x, const blas_int *incx, doublecomplex *y, const blas_int *incy);

//SUM
float   sasum_(const blas_int *n, const float         *x, const blas_int *incx);
float  scasum_(const blas_int *n, const complex       *x, const blas_int *incx);
double  dasum_(const blas_int *n, const double        *x, const blas_int *incx);
double dzasum_(const blas_int *n, const doublecomplex *x, const blas_int *incx);

//COPY
void scopy_(const blas_int *n, const float  *x, const blas_int *incx, float  *y, const blas_int *incy);
void dcopy_(const blas_int *n, const double *x, const blas_int *incx, double *y, const blas_int *incy);
void ccopy_(const blas_int *n, const float  *x, const blas_int *incx, float  *y, const blas_int *incy);
void zcopy_(const blas_int *n, const double *x, const blas_int *incx, double *y, const blas_int *incy);

//DOT
float  sdot_(const blas_int *n, const float  *x, const blas_int *incx, const float  *y, const blas_int *incy);
double ddot_(const blas_int *n, const double *x, const blas_int *incx, const double *y, const blas_int *incy);

//DOTC
complex       cdotc_(const blas_int *n, const complex       *x, const blas_int *incx, const complex       *y, const blas_int *incy);
doublecomplex zdotc_(const blas_int *n, const doublecomplex *x, const blas_int *incx, const doublecomplex *y, const blas_int *incy);

//DOTU
complex       cdotu_(const blas_int *n, const complex       *x, const blas_int *incx, const complex       *y, const blas_int *incy);
doublecomplex zdotu_(const blas_int *n, const doublecomplex *x, const blas_int *incx, const doublecomplex *y, const blas_int *incy);

//NRM2
float   snrm2_(const blas_int *n, const float         *x, const blas_int *incx);
double  dnrm2_(const blas_int *n, const double        *x, const blas_int *incx);
float  scnrm2_(const blas_int *n, const complex       *x, const blas_int *incx);
double dznrm2_(const blas_int *n, const doublecomplex *x, const blas_int *incx);

//ROT
void  srot_(const blas_int *n, float         *x, const blas_int *incx, float         *y, const blas_int *incy, const float  *c, const float  *s);
void  drot_(const blas_int *n, double        *x, const blas_int *incx, double        *y, const blas_int *incy, const double *c, const double *s);
void csrot_(const blas_int *n, complex       *x, const blas_int *incx, complex       *y, const blas_int *incy, const float  *c, const float  *s);
void zdrot_(const blas_int *n, doublecomplex *x, const blas_int *incx, doublecomplex *y, const blas_int *incy, const double *c, const double *s);

//ROTG
void srotg_(float         *a, float         *b, float  *c, float  *s);
void drotg_(double        *a, double        *b, double *c, double *s);
void crotg_(complex       *a, complex       *b, float  *c, float  *s);
void zrotg_(doublecomplex *a, doublecomplex *b, double *c, double *s);

//Stub
//ROTMG
//ROTM


//SCAL
void  sscal_(const blas_int *n,  const float         *a, float         *x, const blas_int *incx);
void  dscal_(const blas_int *n,  const double        *a, double        *x, const blas_int *incx);
void  cscal_(const blas_int *n,  const complex       *a, complex       *x, const blas_int *incx);
void  zscal_(const blas_int *n,  const doublecomplex *a, doublecomplex *x, const blas_int *incx);
void csscal_(const blas_int *n,  const float         *a, complex       *x, const blas_int *incx);
void zdscal_(const blas_int *n,  const double        *a, doublecomplex *x, const blas_int *incx);

//SWAP
void sswap_(const blas_int *n, float         *x, const blas_int *incx, float         *y, const blas_int *incy);
void dswap_(const blas_int *n, double        *x, const blas_int *incx, double        *y, const blas_int *incy);
void cswap_(const blas_int *n, complex       *x, const blas_int *incx, complex       *y, const blas_int *incy);
void zswap_(const blas_int *n, doublecomplex *x, const blas_int *incx, doublecomplex *y, const blas_int *incy);

//IAMAX
blas_int isamax_(const blas_int *n, const float         *x, const blas_int *incx);
blas_int idamax_(const blas_int *n, const double        *x, const blas_int *incx);
blas_int icamax_(const blas_int *n, const complex       *x, const blas_int *incx);
blas_int izamax_(const blas_int *n, const doublecomplex *x, const blas_int *incx);

//IAMIN
blas_int isamin_(const blas_int *n, const float         *x, const blas_int *incx);
blas_int idamin_(const blas_int *n, const double        *x, const blas_int *incx);
blas_int icamin_(const blas_int *n, const complex       *x, const blas_int *incx);
blas_int izamin_(const blas_int *n, const doublecomplex *x, const blas_int *incx);

//IMAX
blas_int ismax_(const blas_int *n, const float  *x, const blas_int *incx);
blas_int idmax_(const blas_int *n, const double *x, const blas_int *incx);

//IMIN
blas_int ismin_(const blas_int *n, const float  *x, const blas_int *incx);
blas_int idmin_(const blas_int *n, const double *x, const blas_int *incx);

//Level2

//GBMV
void sgbmv_(const char *trans, const blas_int *m, const blas_int *n, const blas_int *kl, const blas_int *ku,
            const float         *alpha, const float         *A, const blas_int *ldA, const float         *x, const blas_int *incx,
            const float         *beta , float         *y, const blas_int *incy);
void dgbmv_(const char *trans, const blas_int *m, const blas_int *n, const blas_int *kl, const blas_int *ku,
            const double        *alpha, const double        *A, const blas_int *ldA, const double        *x, const blas_int *incx,
            const double        *beta , double        *y, const blas_int *incy);
void cgbmv_(const char *trans, const blas_int *m, const blas_int *n, const blas_int *kl, const blas_int *ku,
            const complex       *alpha, const complex       *A, const blas_int *ldA, const complex       *x, const blas_int *incx,
            const complex       *beta , complex       *y, const blas_int *incy);
void zgbmv_(const char *trans, const blas_int *m, const blas_int *n, const blas_int *kl, const blas_int *ku,
            const doublecomplex *alpha, const doublecomplex *A, const blas_int *ldA, const doublecomplex *x, const blas_int *incx,
            const doublecomplex *beta , doublecomplex *y, const blas_int *incy);

//GEMV
void sgemv_(const char *trans, const blas_int *m, const blas_int *n,
            const float         *alpha, const float         *A, const blas_int *ldA, const float         *x, const blas_int *incx,
            const float         *beta , float         *y, const blas_int *incy);
void dgemv_(const char *trans, const blas_int *m, const blas_int *n,
            const double        *alpha, const double        *A, const blas_int *ldA, const double        *x, const blas_int *incx,
            const double        *beta , double        *y, const blas_int *incy);
void cgemv_(const char *trans, const blas_int *m, const blas_int *n,
            const complex       *alpha, const complex       *A, const blas_int *ldA, const complex       *x, const blas_int *incx,
            const complex       *beta , complex       *y, const blas_int *incy);
void zgemv_(const char *trans, const blas_int *m, const blas_int *n,
            const doublecomplex *alpha, const doublecomplex *A, const blas_int *ldA, const doublecomplex *x, const blas_int *incx,
            const doublecomplex *beta , doublecomplex *y, const blas_int *incy);

//GER
void sger_(const blas_int *m, const blas_int *n, const float  *alpha, const float  *x, const blas_int *incx, const float  *y, const blas_int *incy, float  *A, const blas_int *ldA);
void dger_(const blas_int *m, const blas_int *n, const double *alpha, const double *x, const blas_int *incx, const double *y, const blas_int *incy, double *A, const blas_int *ldA);

//GERC
void cgerc_(const blas_int *m, const blas_int *n, const complex       *alpha, const complex       *x, const blas_int *incx,
            const complex       *y, const blas_int *incy, complex       *A, const blas_int *ldA);
void zgerc_(const blas_int *m, const blas_int *n, const doublecomplex *alpha, const doublecomplex *x, const blas_int *incx,
            const doublecomplex *y, const blas_int *incy, doublecomplex *A, const blas_int *ldA);

//GREU
void cgeru_(const blas_int *m, const blas_int *n, const complex       *alpha, const complex       *x, const blas_int *incx,
            const complex       *y, const blas_int *incy, complex       *A, const blas_int *ldA);
void zgeru_(const blas_int *m, const blas_int *n, const doublecomplex *alpha, const doublecomplex *x, const blas_int *incx,
            const doublecomplex *y, const blas_int *incy, doublecomplex *A, const blas_int *ldA);

//HBMV
void chbmv_(const char *uplo, const blas_int *n, const blas_int *k, const complex       *alpha, const complex       *A, const blas_int *ldA,
            const complex       *x, const blas_int *incx, const complex       *beta, complex       *y, const blas_int *incy);
void zhbmv_(const char *uplo, const blas_int *n, const blas_int *k, const doublecomplex *alpha, const doublecomplex *A, const blas_int *ldA,
            const doublecomplex *x, const blas_int *incx, const doublecomplex *beta, doublecomplex *y, const blas_int *incy);

//HEMV
void chemv_(const char *uplo, const blas_int *n, const complex       *alpha, const complex       *A, const blas_int *ldA,
            const complex       *x, const blas_int *incx, const complex       *beta, complex       *y, const blas_int *incy);
void zhemv_(const char *uplo, const blas_int *n, const doublecomplex *alpha, const doublecomplex *A, const blas_int *ldA,
            const doublecomplex *x, const blas_int *incx, const doublecomplex *beta, doublecomplex *y, const blas_int *incy);

//HER
void cher_(const char *uplo, const blas_int *n, const float  *alpha, const complex       *x, const blas_int *incx, complex       *A, const blas_int *ldA);
void zher_(const char *uplo, const blas_int *n, const double *alpha, const doublecomplex *x, const blas_int *incx, doublecomplex *A, const blas_int *ldA);

//Stub
//HER2

//HPMV
void chpmv_(const char *uplo, const blas_int *n, const complex       *alpha, const complex       *A,
            const complex       *x, const blas_int *incx, const complex       *beta, complex       *y, const blas_int *incy);
void zhpmv_(const char *uplo, const blas_int *n, const doublecomplex *alpha, const doublecomplex *A,
            const doublecomplex *x, const blas_int *incx, const doublecomplex *beta, doublecomplex *y, const blas_int *incy);

//HPR
void chpr_ (const char *uplo, const blas_int *n, const float  *alpha, const complex       *x, const blas_int *incx, complex       *A);
void zhpr_ (const char *uplo, const blas_int *n, const double *alpha, const doublecomplex *x, const blas_int *incx, doublecomplex *A);

//Stub
//HPR2

//SBMV
void ssbmv_(const char *uplo, const blas_int *n, const blas_int *k, const float  *alpha, const float  *A, const blas_int *ldA,
            const float  *x, const blas_int *incx, const float  *beta, float  *y, const blas_int *incy);
void dsbmv_(const char *uplo, const blas_int *n, const blas_int *k, const double *alpha, const double *A, const blas_int *ldA,
            const double *x, const blas_int *incx, const double *beta, double *y, const blas_int *incy);

//SPMV
void sspmv_(const char *uplo, const blas_int *n, const float  *alpha, const float  *A, const float  *x, const blas_int *incx, const float  *beta, float  *y, const blas_int *incy);
void dspmv_(const char *uplo, const blas_int *n, const double *alpha, const double *A, const double *x, const blas_int *incx, const double *beta, double *y, const blas_int *incy);

//SPR
void sspr_(const char *uplo, const blas_int *n, const float  *alpha, const float  *x, const blas_int *incx, float  *A);
void dspr_(const char *uplo, const blas_int *n, const double *alpha, const double *x, const blas_int *incx, double *A);

//Stub
//SPR2

//SYMV
void ssymv_(const char *uplo, const blas_int *n, const float  *alpha, const float  *A, const blas_int *ldA,
            const float  *x, const blas_int *incx, const float  *beta, float  *y, const blas_int *incy);
void dsymv_(const char *uplo, const blas_int *n, const double *alpha, const double *A, const blas_int *ldA,
            const double *x, const blas_int *incx, const double *beta, double *y, const blas_int *incy);

//SYR
void ssyr_(const char *uplo, const blas_int *n, const float  *alpha, const float  *x, const blas_int *incx, float  *A, const blas_int *ldA);
void dsyr_(const char *uplo, const blas_int *n, const double *alpha, const double *x, const blas_int *incx, double *A, const blas_int *ldA);

//Stub
//SYR2

//TBMV
void stbmv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const blas_int *k, const float         *A, const blas_int *ldA, float         *x, const blas_int *incx);
void dtbmv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const blas_int *k, const double        *A, const blas_int *ldA, double        *x, const blas_int *incx);
void ctbmv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const blas_int *k, const complex       *A, const blas_int *ldA, complex       *x, const blas_int *incx);
void ztbmv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const blas_int *k, const doublecomplex *A, const blas_int *ldA, doublecomplex *x, const blas_int *incx);

//TBSV
void stbsv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const blas_int *k, const float         *A, const blas_int *ldA, float         *x, const blas_int *incx);
void dtbsv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const blas_int *k, const double        *A, const blas_int *ldA, double        *x, const blas_int *incx);
void ctbsv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const blas_int *k, const complex       *A, const blas_int *ldA, complex       *x, const blas_int *incx);
void ztbsv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const blas_int *k, const doublecomplex *A, const blas_int *ldA, doublecomplex *x, const blas_int *incx);

//TPMV
void stpmv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const float         *A, float         *x, const blas_int *incx);
void dtpmv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const double        *A, double        *x, const blas_int *incx);
void ctpmv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const complex       *A, complex       *x, const blas_int *incx);
void ztpmv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const doublecomplex *A, doublecomplex *x, const blas_int *incx);

//TPSV
void stpsv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const float         *A, float         *x, const blas_int *incx);
void dtpsv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const double        *A, double        *x, const blas_int *incx);
void ctpsv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const complex       *A, complex       *x, const blas_int *incx);
void ztpsv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const doublecomplex *A, doublecomplex *x, const blas_int *incx);

//TRSV
void strsv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const float         *A, const blas_int *ldA, float         *x, const blas_int *incx);
void dtrsv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const double        *A, const blas_int *ldA, double        *x, const blas_int *incx);
void ctrsv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const complex       *A, const blas_int *ldA, complex       *x, const blas_int *incx);
void ztrsv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const doublecomplex *A, const blas_int *ldA, doublecomplex *x, const blas_int *incx);

//TRMV
void strmv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const float         *A, const blas_int *ldA, float         *x, const blas_int *incx);
void dtrmv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const double        *A, const blas_int *ldA, double        *x, const blas_int *incx);
void ctrmv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const complex       *A, const blas_int *ldA, complex       *x, const blas_int *incx);
void ztrmv_(const char *uplo, const char *trans, const char *diag, const blas_int *n, const doublecomplex *A, const blas_int *ldA, doublecomplex *x, const blas_int *incx);

//Level3

//GEMM
void sgemm_(const char *transa, const char *transb, const blas_int *m, const blas_int *n, const blas_int *k,
            const float         *alpha, const float         *A, const blas_int *ldA, const float         *B, const blas_int *ldB,
            const float         *beta , float         *C, const blas_int *ldC);
void dgemm_(const char *transa, const char *transb, const blas_int *m, const blas_int *n, const blas_int *k,
            const double        *alpha, const double        *A, const blas_int *ldA, const double        *B, const blas_int *ldB,
            const double        *beta , double        *C, const blas_int *ldC);
void cgemm_(const char *transa, const char *transb, const blas_int *m, const blas_int *n, const blas_int *k,
            const complex       *alpha, const complex       *A, const blas_int *ldA, const complex       *B, const blas_int *ldB,
            const complex       *beta , complex       *C, const blas_int *ldC);
void zgemm_(const char *transa, const char *transb, const blas_int *m, const blas_int *n, const blas_int *k,
            const doublecomplex *alpha, const doublecomplex *A, const blas_int *ldA, const doublecomplex *B, const blas_int *ldB,
            const doublecomplex *beta , doublecomplex *C, const blas_int *ldC);

//HEMM
void chemm_(const char *side, const char *uplo, const blas_int *m, const blas_int *n, const complex       *alpha, const complex       *A, const blas_int *ldA,
            const complex       *B, const blas_int *ldB, const complex       *beta, complex       *C, const blas_int *ldC);
void zhemm_(const char *side, const char *uplo, const blas_int *m, const blas_int *n, const doublecomplex *alpha, const doublecomplex *A, const blas_int *ldA,
            const doublecomplex *B, const blas_int *ldB, const doublecomplex *beta, doublecomplex *C, const blas_int *ldC);

//HERK
void cherk_(const char *uplo, const char *trans, const blas_int *n, const blas_int *k, const float  *alpha, const complex       *A, const blas_int *ldA,
            const float  *beta , complex       *C, const blas_int *ldC);
void zherk_(const char *uplo, const char *trans, const blas_int *n, const blas_int *k, const double *alpha, const doublecomplex *A, const blas_int *ldA,
            const double *beta , doublecomplex *C, const blas_int *ldC);

//HERK2
void cher2k_(const char *uplo, const char *trans, const blas_int *n, const blas_int *k, const complex       *alpha, const complex       *A, const blas_int *ldA,
             const complex       *B, const blas_int *ldB, const float  *beta, complex       *C, const blas_int *ldC);
void zher2k_(const char *uplo, const char *trans, const blas_int *n, const blas_int *k, const doublecomplex *alpha, const doublecomplex *A, const blas_int *ldA,
             const doublecomplex *B, const blas_int *ldB, const double *beta, doublecomplex *C, const blas_int *ldC);

//SYMM
void ssymm_(const char *side, const char *uplo, const blas_int *m, const blas_int *n,
            const float         *alpha, const float         *A, const blas_int *ldA, const float         *B, const blas_int *ldB,
            const float         *beta , float         *C, const blas_int *ldC);
void dsymm_(const char *side, const char *uplo, const blas_int *m, const blas_int *n,
            const double        *alpha, const double        *A, const blas_int *ldA, const double        *B, const blas_int *ldB,
            const double        *beta , double        *C, const blas_int *ldC);
void csymm_(const char *side, const char *uplo, const blas_int *m, const blas_int *n,
            const complex       *alpha, const complex       *A, const blas_int *ldA, const complex       *B, const blas_int *ldB,
            const complex       *beta , complex       *C, const blas_int *ldC);
void zsymm_(const char *side, const char *uplo, const blas_int *m, const blas_int *n,
            const doublecomplex *alpha, const doublecomplex *A, const blas_int *ldA, const doublecomplex *B, const blas_int *ldB,
            const doublecomplex *beta , doublecomplex *C, const blas_int *ldC);

//SYRK
void ssyrk_(const char *uplo, const char *trans, const blas_int *n, const blas_int *k, const float  *alpha, const float         *A, const blas_int *ldA,
            const float  *beta , float         *C, const blas_int *ldC);
void dsyrk_(const char *uplo, const char *trans, const blas_int *n, const blas_int *k, const double *alpha, const double        *A, const blas_int *ldA,
            const double *beta , double        *C, const blas_int *ldC);
void csyrk_(const char *uplo, const char *trans, const blas_int *n, const blas_int *k, const float  *alpha, const complex       *A, const blas_int *ldA,
            const float  *beta , complex       *C, const blas_int *ldC);
void zsyrk_(const char *uplo, const char *trans, const blas_int *n, const blas_int *k, const double *alpha, const doublecomplex *A, const blas_int *ldA,
            const double *beta , doublecomplex *C, const blas_int *ldC);

//SYR2K
void ssyr2k_(const char *uplo, const char *trans, const blas_int *n, const blas_int *k, const float  *alpha, const float         *A, const blas_int *ldA, const float         *B, const blas_int *ldB,
             const float  *beta , float         *C, const blas_int *ldC);
void dsyr2k_(const char *uplo, const char *trans, const blas_int *n, const blas_int *k, const double *alpha, const double        *A, const blas_int *ldA, const double        *B, const blas_int *ldB,
             const double *beta , double        *C, const blas_int *ldC);
void csyr2k_(const char *uplo, const char *trans, const blas_int *n, const blas_int *k, const float  *alpha, const complex       *A, const blas_int *ldA, const complex       *B, const blas_int *ldB,
             const float  *beta , complex       *C, const blas_int *ldC);
void zsyr2k_(const char *uplo, const char *trans, const blas_int *n, const blas_int *k, const double *alpha, const doublecomplex *A, const blas_int *ldA, const doublecomplex *B, const blas_int *ldB,
             const double *beta , doublecomplex *C, const blas_int *ldC);

//TRMM
void strmm_(const char *side, const char *uplo, const char *trans, const char *diag, const blas_int *m, const blas_int *n,
            const float         *alpha, const float         *A, const blas_int *ldA, float         *B, const blas_int *ldB);
void dtrmm_(const char *side, const char *uplo, const char *trans, const char *diag, const blas_int *m, const blas_int *n,
            const double        *alpha, const double        *A, const blas_int *ldA, double        *B, const blas_int *ldB);
void ctrmm_(const char *side, const char *uplo, const char *trans, const char *diag, const blas_int *m, const blas_int *n,
            const complex       *alpha, const complex       *A, const blas_int *ldA, complex       *B, const blas_int *ldB);
void ztrmm_(const char *side, const char *uplo, const char *trans, const char *diag, const blas_int *m, const blas_int *n,
            const doublecomplex *alpha, const doublecomplex *A, const blas_int *ldA, doublecomplex *B, const blas_int *ldB);

//TRSM
void strsm_(const char *side, const char *uplo, const char *trans, const char *diag, const blas_int *m, const blas_int *n,
            const float         *alpha, const float         *A, const blas_int *ldA, float         *B, const blas_int *ldB);
void dtrsm_(const char *side, const char *uplo, const char *trans, const char *diag, const blas_int *m, const blas_int *n,
            const double        *alpha, const double        *A, const blas_int *ldA, double        *B, const blas_int *ldB);
void ctrsm_(const char *side, const char *uplo, const char *trans, const char *diag, const blas_int *m, const blas_int *n,
            const complex       *alpha, const complex       *A, const blas_int *ldA, complex       *B, const blas_int *ldB);
void ztrsm_(const char *side, const char *uplo, const char *trans, const char *diag, const blas_int *m, const blas_int *n,
            const doublecomplex *alpha, const doublecomplex *A, const blas_int *ldA, doublecomplex *B, const blas_int *ldB);

#ifdef __cplusplus
}
#endif

#endif
