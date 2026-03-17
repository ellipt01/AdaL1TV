#ifndef BLAS_INTERFACE_HEADER
#define BLAS_INTERFACE_HEADER

#ifdef __cplusplus
extern "C"{
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


//MM_INT xrebla_(const char *srname, const MM_INT *info);

//Level1

//AXPY
void saxpy_(const MM_INT *n, const float         *alpha, const float         *x, const MM_INT *incx, float         *y, const MM_INT *incy);
void daxpy_(const MM_INT *n, const double        *alpha, const double        *x, const MM_INT *incx, double        *y, const MM_INT *incy);
void caxpy_(const MM_INT *n, const complex       *alpha, const complex       *x, const MM_INT *incx, complex       *y, const MM_INT *incy);
void zaxpy_(const MM_INT *n, const doublecomplex *alpha, const doublecomplex *x, const MM_INT *incx, doublecomplex *y, const MM_INT *incy);

//SUM
float   sasum_(const MM_INT *n, const float         *x, const MM_INT *incx);
float  scasum_(const MM_INT *n, const complex       *x, const MM_INT *incx);
double  dasum_(const MM_INT *n, const double        *x, const MM_INT *incx);
double dzasum_(const MM_INT *n, const doublecomplex *x, const MM_INT *incx);

//COPY
void scopy_(const MM_INT *n, const float  *x, const MM_INT *incx, float  *y, const MM_INT *incy);
void dcopy_(const MM_INT *n, const double *x, const MM_INT *incx, double *y, const MM_INT *incy);
void ccopy_(const MM_INT *n, const float  *x, const MM_INT *incx, float  *y, const MM_INT *incy);
void zcopy_(const MM_INT *n, const double *x, const MM_INT *incx, double *y, const MM_INT *incy);

//DOT
float  sdot_(const MM_INT *n, const float  *x, const MM_INT *incx, const float  *y, const MM_INT *incy);
double ddot_(const MM_INT *n, const double *x, const MM_INT *incx, const double *y, const MM_INT *incy);

//DOTC
complex       cdotc_(const MM_INT *n, const complex       *x, const MM_INT *incx, const complex       *y, const MM_INT *incy);
doublecomplex zdotc_(const MM_INT *n, const doublecomplex *x, const MM_INT *incx, const doublecomplex *y, const MM_INT *incy);

//DOTU
complex       cdotu_(const MM_INT *n, const complex       *x, const MM_INT *incx, const complex       *y, const MM_INT *incy);
doublecomplex zdotu_(const MM_INT *n, const doublecomplex *x, const MM_INT *incx, const doublecomplex *y, const MM_INT *incy);

//NRM2
float   snrm2_(const MM_INT *n, const float         *x, const MM_INT *incx);
double  dnrm2_(const MM_INT *n, const double        *x, const MM_INT *incx);
float  scnrm2_(const MM_INT *n, const complex       *x, const MM_INT *incx);
double dznrm2_(const MM_INT *n, const doublecomplex *x, const MM_INT *incx);

//ROT
void  srot_(const MM_INT *n, float         *x, const MM_INT *incx, float         *y, const MM_INT *incy, const float  *c, const float  *s);
void  drot_(const MM_INT *n, double        *x, const MM_INT *incx, double        *y, const MM_INT *incy, const double *c, const double *s);
void csrot_(const MM_INT *n, complex       *x, const MM_INT *incx, complex       *y, const MM_INT *incy, const float  *c, const float  *s);
void zdrot_(const MM_INT *n, doublecomplex *x, const MM_INT *incx, doublecomplex *y, const MM_INT *incy, const double *c, const double *s);

//ROTG
void srotg_(float         *a, float         *b, float  *c, float  *s);
void drotg_(double        *a, double        *b, double *c, double *s);
void crotg_(complex       *a, complex       *b, float  *c, float  *s);
void zrotg_(doublecomplex *a, doublecomplex *b, double *c, double *s);

//Stub
//ROTMG
//ROTM


//SCAL
void  sscal_(const MM_INT *n,  const float         *a, float         *x, const MM_INT *incx);
void  dscal_(const MM_INT *n,  const double        *a, double        *x, const MM_INT *incx);
void  cscal_(const MM_INT *n,  const complex       *a, complex       *x, const MM_INT *incx);
void  zscal_(const MM_INT *n,  const doublecomplex *a, doublecomplex *x, const MM_INT *incx);
void csscal_(const MM_INT *n,  const float         *a, complex       *x, const MM_INT *incx);
void zdscal_(const MM_INT *n,  const double        *a, doublecomplex *x, const MM_INT *incx);

//SWAP
void sswap_(const MM_INT *n, float         *x, const MM_INT *incx, float         *y, const MM_INT *incy);
void dswap_(const MM_INT *n, double        *x, const MM_INT *incx, double        *y, const MM_INT *incy);
void cswap_(const MM_INT *n, complex       *x, const MM_INT *incx, complex       *y, const MM_INT *incy);
void zswap_(const MM_INT *n, doublecomplex *x, const MM_INT *incx, doublecomplex *y, const MM_INT *incy);

//IAMAX
MM_INT isamax_(const MM_INT *n, const float         *x, const MM_INT *incx);
MM_INT idamax_(const MM_INT *n, const double        *x, const MM_INT *incx);
MM_INT icamax_(const MM_INT *n, const complex       *x, const MM_INT *incx);
MM_INT izamax_(const MM_INT *n, const doublecomplex *x, const MM_INT *incx);

//IAMIN
MM_INT isamin_(const MM_INT *n, const float         *x, const MM_INT *incx);
MM_INT idamin_(const MM_INT *n, const double        *x, const MM_INT *incx);
MM_INT icamin_(const MM_INT *n, const complex       *x, const MM_INT *incx);
MM_INT izamin_(const MM_INT *n, const doublecomplex *x, const MM_INT *incx);

//IMAX
MM_INT ismax_(const MM_INT *n, const float  *x, const MM_INT *incx);
MM_INT idmax_(const MM_INT *n, const double *x, const MM_INT *incx);

//IMIN
MM_INT ismin_(const MM_INT *n, const float  *x, const MM_INT *incx);
MM_INT idmin_(const MM_INT *n, const double *x, const MM_INT *incx);

//Level2

//GBMV
void sgbmv_(const char *trans, const MM_INT *m, const MM_INT *n, const MM_INT *kl, const MM_INT *ku,
            const float         *alpha, const float         *A, const MM_INT *ldA, const float         *x, const MM_INT *incx,
            const float         *beta , float         *y, const MM_INT *incy);
void dgbmv_(const char *trans, const MM_INT *m, const MM_INT *n, const MM_INT *kl, const MM_INT *ku,
            const double        *alpha, const double        *A, const MM_INT *ldA, const double        *x, const MM_INT *incx,
            const double        *beta , double        *y, const MM_INT *incy);
void cgbmv_(const char *trans, const MM_INT *m, const MM_INT *n, const MM_INT *kl, const MM_INT *ku,
            const complex       *alpha, const complex       *A, const MM_INT *ldA, const complex       *x, const MM_INT *incx,
            const complex       *beta , complex       *y, const MM_INT *incy);
void zgbmv_(const char *trans, const MM_INT *m, const MM_INT *n, const MM_INT *kl, const MM_INT *ku,
            const doublecomplex *alpha, const doublecomplex *A, const MM_INT *ldA, const doublecomplex *x, const MM_INT *incx,
            const doublecomplex *beta , doublecomplex *y, const MM_INT *incy);

//GEMV
void sgemv_(const char *trans, const MM_INT *m, const MM_INT *n,
            const float         *alpha, const float         *A, const MM_INT *ldA, const float         *x, const MM_INT *incx,
            const float         *beta , float         *y, const MM_INT *incy);
void dgemv_(const char *trans, const MM_INT *m, const MM_INT *n,
            const double        *alpha, const double        *A, const MM_INT *ldA, const double        *x, const MM_INT *incx,
            const double        *beta , double        *y, const MM_INT *incy);
void cgemv_(const char *trans, const MM_INT *m, const MM_INT *n,
            const complex       *alpha, const complex       *A, const MM_INT *ldA, const complex       *x, const MM_INT *incx,
            const complex       *beta , complex       *y, const MM_INT *incy);
void zgemv_(const char *trans, const MM_INT *m, const MM_INT *n,
            const doublecomplex *alpha, const doublecomplex *A, const MM_INT *ldA, const doublecomplex *x, const MM_INT *incx,
            const doublecomplex *beta , doublecomplex *y, const MM_INT *incy);

//GER
void sger_(const MM_INT *m, const MM_INT *n, const float  *alpha, const float  *x, const MM_INT *incx, const float  *y, const MM_INT *incy, float  *A, const MM_INT *ldA);
void dger_(const MM_INT *m, const MM_INT *n, const double *alpha, const double *x, const MM_INT *incx, const double *y, const MM_INT *incy, double *A, const MM_INT *ldA);

//GERC
void cgerc_(const MM_INT *m, const MM_INT *n, const complex       *alpha, const complex       *x, const MM_INT *incx,
            const complex       *y, const MM_INT *incy, complex       *A, const MM_INT *ldA);
void zgerc_(const MM_INT *m, const MM_INT *n, const doublecomplex *alpha, const doublecomplex *x, const MM_INT *incx,
            const doublecomplex *y, const MM_INT *incy, doublecomplex *A, const MM_INT *ldA);

//GREU
void cgeru_(const MM_INT *m, const MM_INT *n, const complex       *alpha, const complex       *x, const MM_INT *incx,
            const complex       *y, const MM_INT *incy, complex       *A, const MM_INT *ldA);
void zgeru_(const MM_INT *m, const MM_INT *n, const doublecomplex *alpha, const doublecomplex *x, const MM_INT *incx,
            const doublecomplex *y, const MM_INT *incy, doublecomplex *A, const MM_INT *ldA);

//HBMV
void chbmv_(const char *uplo, const MM_INT *n, const MM_INT *k, const complex       *alpha, const complex       *A, const MM_INT *ldA,
            const complex       *x, const MM_INT *incx, const complex       *beta, complex       *y, const MM_INT *incy);
void zhbmv_(const char *uplo, const MM_INT *n, const MM_INT *k, const doublecomplex *alpha, const doublecomplex *A, const MM_INT *ldA,
            const doublecomplex *x, const MM_INT *incx, const doublecomplex *beta, doublecomplex *y, const MM_INT *incy);

//HEMV
void chemv_(const char *uplo, const MM_INT *n, const complex       *alpha, const complex       *A, const MM_INT *ldA,
            const complex       *x, const MM_INT *incx, const complex       *beta, complex       *y, const MM_INT *incy);
void zhemv_(const char *uplo, const MM_INT *n, const doublecomplex *alpha, const doublecomplex *A, const MM_INT *ldA,
            const doublecomplex *x, const MM_INT *incx, const doublecomplex *beta, doublecomplex *y, const MM_INT *incy);

//HER
void cher_(const char *uplo, const MM_INT *n, const float  *alpha, const complex       *x, const MM_INT *incx, complex       *A, const MM_INT *ldA);
void zher_(const char *uplo, const MM_INT *n, const double *alpha, const doublecomplex *x, const MM_INT *incx, doublecomplex *A, const MM_INT *ldA);

//Stub
//HER2

//HPMV
void chpmv_(const char *uplo, const MM_INT *n, const complex       *alpha, const complex       *A,
            const complex       *x, const MM_INT *incx, const complex       *beta, complex       *y, const MM_INT *incy);
void zhpmv_(const char *uplo, const MM_INT *n, const doublecomplex *alpha, const doublecomplex *A,
            const doublecomplex *x, const MM_INT *incx, const doublecomplex *beta, doublecomplex *y, const MM_INT *incy);

//HPR
void chpr_ (const char *uplo, const MM_INT *n, const float  *alpha, const complex       *x, const MM_INT *incx, complex       *A);
void zhpr_ (const char *uplo, const MM_INT *n, const double *alpha, const doublecomplex *x, const MM_INT *incx, doublecomplex *A);

//Stub
//HPR2

//SBMV
void ssbmv_(const char *uplo, const MM_INT *n, const MM_INT *k, const float  *alpha, const float  *A, const MM_INT *ldA,
            const float  *x, const MM_INT *incx, const float  *beta, float  *y, const MM_INT *incy);
void dsbmv_(const char *uplo, const MM_INT *n, const MM_INT *k, const double *alpha, const double *A, const MM_INT *ldA,
            const double *x, const MM_INT *incx, const double *beta, double *y, const MM_INT *incy);

//SPMV
void sspmv_(const char *uplo, const MM_INT *n, const float  *alpha, const float  *A, const float  *x, const MM_INT *incx, const float  *beta, float  *y, const MM_INT *incy);
void dspmv_(const char *uplo, const MM_INT *n, const double *alpha, const double *A, const double *x, const MM_INT *incx, const double *beta, double *y, const MM_INT *incy);

//SPR
void sspr_(const char *uplo, const MM_INT *n, const float  *alpha, const float  *x, const MM_INT *incx, float  *A);
void dspr_(const char *uplo, const MM_INT *n, const double *alpha, const double *x, const MM_INT *incx, double *A);

//Stub
//SPR2

//SYMV
void ssymv_(const char *uplo, const MM_INT *n, const float  *alpha, const float  *A, const MM_INT *ldA,
            const float  *x, const MM_INT *incx, const float  *beta, float  *y, const MM_INT *incy);
void dsymv_(const char *uplo, const MM_INT *n, const double *alpha, const double *A, const MM_INT *ldA,
            const double *x, const MM_INT *incx, const double *beta, double *y, const MM_INT *incy);

//SYR
void ssyr_(const char *uplo, const MM_INT *n, const float  *alpha, const float  *x, const MM_INT *incx, float  *A, const MM_INT *ldA);
void dsyr_(const char *uplo, const MM_INT *n, const double *alpha, const double *x, const MM_INT *incx, double *A, const MM_INT *ldA);

//Stub
//SYR2

//TBMV
void stbmv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const MM_INT *k, const float         *A, const MM_INT *ldA, float         *x, const MM_INT *incx);
void dtbmv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const MM_INT *k, const double        *A, const MM_INT *ldA, double        *x, const MM_INT *incx);
void ctbmv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const MM_INT *k, const complex       *A, const MM_INT *ldA, complex       *x, const MM_INT *incx);
void ztbmv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const MM_INT *k, const doublecomplex *A, const MM_INT *ldA, doublecomplex *x, const MM_INT *incx);

//TBSV
void stbsv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const MM_INT *k, const float         *A, const MM_INT *ldA, float         *x, const MM_INT *incx);
void dtbsv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const MM_INT *k, const double        *A, const MM_INT *ldA, double        *x, const MM_INT *incx);
void ctbsv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const MM_INT *k, const complex       *A, const MM_INT *ldA, complex       *x, const MM_INT *incx);
void ztbsv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const MM_INT *k, const doublecomplex *A, const MM_INT *ldA, doublecomplex *x, const MM_INT *incx);

//TPMV
void stpmv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const float         *A, float         *x, const MM_INT *incx);
void dtpmv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const double        *A, double        *x, const MM_INT *incx);
void ctpmv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const complex       *A, complex       *x, const MM_INT *incx);
void ztpmv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const doublecomplex *A, doublecomplex *x, const MM_INT *incx);

//TPSV
void stpsv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const float         *A, float         *x, const MM_INT *incx);
void dtpsv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const double        *A, double        *x, const MM_INT *incx);
void ctpsv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const complex       *A, complex       *x, const MM_INT *incx);
void ztpsv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const doublecomplex *A, doublecomplex *x, const MM_INT *incx);

//TRSV
void strsv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const float         *A, const MM_INT *ldA, float         *x, const MM_INT *incx);
void dtrsv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const double        *A, const MM_INT *ldA, double        *x, const MM_INT *incx);
void ctrsv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const complex       *A, const MM_INT *ldA, complex       *x, const MM_INT *incx);
void ztrsv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const doublecomplex *A, const MM_INT *ldA, doublecomplex *x, const MM_INT *incx);

//TRMV
void strmv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const float         *A, const MM_INT *ldA, float         *x, const MM_INT *incx);
void dtrmv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const double        *A, const MM_INT *ldA, double        *x, const MM_INT *incx);
void ctrmv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const complex       *A, const MM_INT *ldA, complex       *x, const MM_INT *incx);
void ztrmv_(const char *uplo, const char *trans, const char *diag, const MM_INT *n, const doublecomplex *A, const MM_INT *ldA, doublecomplex *x, const MM_INT *incx);

//Level3

//GEMM
void sgemm_(const char *transa, const char *transb, const MM_INT *m, const MM_INT *n, const MM_INT *k,
            const float         *alpha, const float         *A, const MM_INT *ldA, const float         *B, const MM_INT *ldB,
            const float         *beta , float         *C, const MM_INT *ldC);
void dgemm_(const char *transa, const char *transb, const MM_INT *m, const MM_INT *n, const MM_INT *k,
            const double        *alpha, const double        *A, const MM_INT *ldA, const double        *B, const MM_INT *ldB,
            const double        *beta , double        *C, const MM_INT *ldC);
void cgemm_(const char *transa, const char *transb, const MM_INT *m, const MM_INT *n, const MM_INT *k,
            const complex       *alpha, const complex       *A, const MM_INT *ldA, const complex       *B, const MM_INT *ldB,
            const complex       *beta , complex       *C, const MM_INT *ldC);
void zgemm_(const char *transa, const char *transb, const MM_INT *m, const MM_INT *n, const MM_INT *k,
            const doublecomplex *alpha, const doublecomplex *A, const MM_INT *ldA, const doublecomplex *B, const MM_INT *ldB,
            const doublecomplex *beta , doublecomplex *C, const MM_INT *ldC);

//HEMM
void chemm_(const char *side, const char *uplo, const MM_INT *m, const MM_INT *n, const complex       *alpha, const complex       *A, const MM_INT *ldA,
            const complex       *B, const MM_INT *ldB, const complex       *beta, complex       *C, const MM_INT *ldC);
void zhemm_(const char *side, const char *uplo, const MM_INT *m, const MM_INT *n, const doublecomplex *alpha, const doublecomplex *A, const MM_INT *ldA,
            const doublecomplex *B, const MM_INT *ldB, const doublecomplex *beta, doublecomplex *C, const MM_INT *ldC);

//HERK
void cherk_(const char *uplo, const char *trans, const MM_INT *n, const MM_INT *k, const float  *alpha, const complex       *A, const MM_INT *ldA,
            const float  *beta , complex       *C, const MM_INT *ldC);
void zherk_(const char *uplo, const char *trans, const MM_INT *n, const MM_INT *k, const double *alpha, const doublecomplex *A, const MM_INT *ldA,
            const double *beta , doublecomplex *C, const MM_INT *ldC);

//HERK2
void cher2k_(const char *uplo, const char *trans, const MM_INT *n, const MM_INT *k, const complex       *alpha, const complex       *A, const MM_INT *ldA,
             const complex       *B, const MM_INT *ldB, const float  *beta, complex       *C, const MM_INT *ldC);
void zher2k_(const char *uplo, const char *trans, const MM_INT *n, const MM_INT *k, const doublecomplex *alpha, const doublecomplex *A, const MM_INT *ldA,
             const doublecomplex *B, const MM_INT *ldB, const double *beta, doublecomplex *C, const MM_INT *ldC);

//SYMM
void ssymm_(const char *side, const char *uplo, const MM_INT *m, const MM_INT *n,
            const float         *alpha, const float         *A, const MM_INT *ldA, const float         *B, const MM_INT *ldB,
            const float         *beta , float         *C, const MM_INT *ldC);
void dsymm_(const char *side, const char *uplo, const MM_INT *m, const MM_INT *n,
            const double        *alpha, const double        *A, const MM_INT *ldA, const double        *B, const MM_INT *ldB,
            const double        *beta , double        *C, const MM_INT *ldC);
void csymm_(const char *side, const char *uplo, const MM_INT *m, const MM_INT *n,
            const complex       *alpha, const complex       *A, const MM_INT *ldA, const complex       *B, const MM_INT *ldB,
            const complex       *beta , complex       *C, const MM_INT *ldC);
void zsymm_(const char *side, const char *uplo, const MM_INT *m, const MM_INT *n,
            const doublecomplex *alpha, const doublecomplex *A, const MM_INT *ldA, const doublecomplex *B, const MM_INT *ldB,
            const doublecomplex *beta , doublecomplex *C, const MM_INT *ldC);

//SYRK
void ssyrk_(const char *uplo, const char *trans, const MM_INT *n, const MM_INT *k, const float  *alpha, const float         *A, const MM_INT *ldA,
            const float  *beta , float         *C, const MM_INT *ldC);
void dsyrk_(const char *uplo, const char *trans, const MM_INT *n, const MM_INT *k, const double *alpha, const double        *A, const MM_INT *ldA,
            const double *beta , double        *C, const MM_INT *ldC);
void csyrk_(const char *uplo, const char *trans, const MM_INT *n, const MM_INT *k, const float  *alpha, const complex       *A, const MM_INT *ldA,
            const float  *beta , complex       *C, const MM_INT *ldC);
void zsyrk_(const char *uplo, const char *trans, const MM_INT *n, const MM_INT *k, const double *alpha, const doublecomplex *A, const MM_INT *ldA,
            const double *beta , doublecomplex *C, const MM_INT *ldC);

//SYR2K
void ssyr2k_(const char *uplo, const char *trans, const MM_INT *n, const MM_INT *k, const float  *alpha, const float         *A, const MM_INT *ldA, const float         *B, const MM_INT *ldB,
             const float  *beta , float         *C, const MM_INT *ldC);
void dsyr2k_(const char *uplo, const char *trans, const MM_INT *n, const MM_INT *k, const double *alpha, const double        *A, const MM_INT *ldA, const double        *B, const MM_INT *ldB,
             const double *beta , double        *C, const MM_INT *ldC);
void csyr2k_(const char *uplo, const char *trans, const MM_INT *n, const MM_INT *k, const float  *alpha, const complex       *A, const MM_INT *ldA, const complex       *B, const MM_INT *ldB,
             const float  *beta , complex       *C, const MM_INT *ldC);
void zsyr2k_(const char *uplo, const char *trans, const MM_INT *n, const MM_INT *k, const double *alpha, const doublecomplex *A, const MM_INT *ldA, const doublecomplex *B, const MM_INT *ldB,
             const double *beta , doublecomplex *C, const MM_INT *ldC);

//TRMM
void strmm_(const char *side, const char *uplo, const char *trans, const char *diag, const MM_INT *m, const MM_INT *n,
            const float         *alpha, const float         *A, const MM_INT *ldA, float         *B, const MM_INT *ldB);
void dtrmm_(const char *side, const char *uplo, const char *trans, const char *diag, const MM_INT *m, const MM_INT *n,
            const double        *alpha, const double        *A, const MM_INT *ldA, double        *B, const MM_INT *ldB);
void ctrmm_(const char *side, const char *uplo, const char *trans, const char *diag, const MM_INT *m, const MM_INT *n,
            const complex       *alpha, const complex       *A, const MM_INT *ldA, complex       *B, const MM_INT *ldB);
void ztrmm_(const char *side, const char *uplo, const char *trans, const char *diag, const MM_INT *m, const MM_INT *n,
            const doublecomplex *alpha, const doublecomplex *A, const MM_INT *ldA, doublecomplex *B, const MM_INT *ldB);

//TRSM
void strsm_(const char *side, const char *uplo, const char *trans, const char *diag, const MM_INT *m, const MM_INT *n,
            const float         *alpha, const float         *A, const MM_INT *ldA, float         *B, const MM_INT *ldB);
void dtrsm_(const char *side, const char *uplo, const char *trans, const char *diag, const MM_INT *m, const MM_INT *n,
            const double        *alpha, const double        *A, const MM_INT *ldA, double        *B, const MM_INT *ldB);
void ctrsm_(const char *side, const char *uplo, const char *trans, const char *diag, const MM_INT *m, const MM_INT *n,
            const complex       *alpha, const complex       *A, const MM_INT *ldA, complex       *B, const MM_INT *ldB);
void ztrsm_(const char *side, const char *uplo, const char *trans, const char *diag, const MM_INT *m, const MM_INT *n,
            const doublecomplex *alpha, const doublecomplex *A, const MM_INT *ldA, doublecomplex *B, const MM_INT *ldB);

#ifdef __cplusplus
}
#endif

#endif
