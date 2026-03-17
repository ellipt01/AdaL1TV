#include <iostream>
#include <cstdio>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <mmreal.h>
#include <mkl.h>
#include "ADMM.h"

/****** public ******/

// Constructor
ADMM::ADMM (size_t m, size_t n, double alpha, double log10_lambda)
{
	if (m <= 0 || n <= 0) 
		throw std::runtime_error ("specified size of the equation is invalid.");

	m_ = m;
	n_ = n;
	
	alpha_  = alpha;
	lambda_ = pow (10., log10_lambda);
}

// Destructor: free all allocated memory
ADMM::~ADMM ()
{
	if (w_) mm_real_free (w_);
	if (beta_) mm_real_free (beta_);
	if (beta_prev_) mm_real_free (beta_prev_);
	if (s_) mm_real_free (s_);
	if (u_) mm_real_free (u_);
	if (t_) mm_real_free (t_);
	if (v_) mm_real_free (v_);
	if (lower_) mm_real_free (lower_);
	if (upper_) mm_real_free (upper_);
	if (c_) mm_real_free (c_);
	if (b_) mm_real_free (b_);
	if (Ci_) mm_real_free (Ci_);
	if (tmp_m_) mm_real_free (tmp_m_);
	if (tmp_n_) mm_real_free (tmp_n_);
}

// Setup the linear system f = X * beta
void ADMM::setupLinearSystem (mm_real *f, mm_real *X, bool do_normalize)
{
	if (f->m != X->m) throw std::runtime_error ("matrix and vector size do not match.");
	if (f->m != m_)   throw std::runtime_error ("size of data vector f is incompatible.");
	if (X->n != n_)   throw std::runtime_error ("second dimension of matrix X is incompatible.");

	f_ = f;
	X_ = X;

	if (do_normalize) {
		if (w_) mm_real_free (w_);
		w_ = normalize_matrix (X_);
	}

	// Allocate model vector and its backup
	if (beta_) mm_real_free (beta_);
	beta_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
	mm_real_set_all (beta_, 0.);

	if (beta_prev_) mm_real_free (beta_prev_);
	beta_prev_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
	mm_real_set_all (beta_prev_, 0.);
}

// Set L1-L2 regularization (elastic net style)
void ADMM::setL1L2Regularization (double mu)
{
	if (mu <= std::numeric_limits<double>::epsilon ()) throw std::runtime_error ("specified mu is invalid.");

	mu_ = mu;

	// Slack variable
	if (s_) mm_real_free (s_);
	s_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
	mm_real_set_all (s_, 0.);

	// Dual variable
	if (u_) mm_real_free (u_);
	u_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
	mm_real_set_all (u_, 0.);
}

// Set bound constraints (box constraints)
void ADMM::setBoundConstraint (double nu, double lower, double upper)
{
	if (upper <= lower) throw std::runtime_error ("lower and/or upper bounds are invalid.");

	if (lower_) mm_real_free (lower_);
	lower_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
	mm_real_set_all (lower_, lower);

	if (upper_) mm_real_free (upper_);
	upper_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
	mm_real_set_all (upper_, upper);

	// Scale bounds if normalization is applied
	if (w_) {
		for (size_t j = 0; j < n_; j++) {
			lower_->data[j] *= w_->data[j];
			upper_->data[j] *= w_->data[j];
		}
	}

	nu_ = nu;

	// Slack variable
	if (t_) mm_real_free (t_);
	t_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
	mm_real_set_all (t_, 0.);

	// Dual variable
	if (v_) mm_real_free (v_);
	v_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
	mm_real_set_all (v_, 0.);
}

// Run ADMM iterations until convergence
size_t ADMM::solve (size_t maxiter, double tol)
{
	if (m_ <= 0 || n_ == 0)
		throw std::runtime_error ("size of the problem has not yet been specified or it is invalid.");
	if (f_ == nullptr || X_ == nullptr)
		throw std::runtime_error ("simultaneous equation has not yet been defined. call setupLinearSystem () first.");
	if (mu_ <= 0) throw std::runtime_error ("TV regularization is not yet defined.");

	// Prepare temporary vectors
	if (tmp_m_ == nullptr) tmp_m_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m_, 1, m_);
	if (tmp_n_ == nullptr) tmp_n_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);

	if (Ci_ == nullptr) factorize_C ();

	size_t niter = 0;
	while (niter < maxiter) {
		iterate ();
		residuals_ = eval_residuals ();
		if (residuals_ < tol) break;
		if (verbose_ && niter % 100 == 0)
			fprintf (stderr, "residual[%04zu] = %.4e / %.4e\n", niter, residuals_, tol);
		niter++;
		if (niter == maxiter && verbose_)
			fprintf(stderr, "WARNING: ADMM reached maxiter without convergence.\n");
	}
	return niter;
}

// Return the estimated model
mm_real *ADMM::getModel (bool remove_weight)
{
	mm_real *beta = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
	mm_real_memcpy (beta, beta_);
	if (remove_weight && w_) {
		for (size_t j = 0; j < n_; j++) beta->data[j] /= w_->data[j];
	}
	return beta;
}

// Recover predicted data f = X * beta
mm_real *ADMM::recover ()
{
	mm_real *f = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m_, 1, m_);
	mm_real_x_dot_yk (false, 1., X_, beta_, 0, 0., f);
	return f;
}

/****** protected ******/

// One ADMM iteration step
void ADMM::iterate ()
{
	update_rhs ();
	update_beta ();
	update_slack_s ();
	update_dual_u ();
	if (nu_ > std::numeric_limits<double>::epsilon ()) {
		update_slack_t ();
		update_dual_v ();
	}
}

// Update right-hand side vector b
void ADMM::update_rhs ()
{
	// b = c + mu * (s + u) + nu * (t + v), where c = X.T * f
	if (b_ == nullptr) b_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);

	if (c_ == nullptr) {
		c_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
		mm_real_x_dot_yk (true, 1., X_, f_, 0, 0., c_);
	}

	if (s_ == nullptr) throw std::runtime_error ("Slack vector s is not yet allocated.");

	mm_real_memcpy (b_, c_);
	mm_real_memcpy (tmp_n_, s_);
	mm_real_axjpy (1., u_, 0, tmp_n_);	// tmp_n = s + u
	mm_real_axjpy (mu_, tmp_n_, 0, b_);	// b = c + mu * (s + u)

	if (nu_ > std::numeric_limits<double>::epsilon ()) {
		mm_real_memcpy (tmp_n_, t_);
		mm_real_axjpy (1., v_, 0, tmp_n_);	// tmp_n = t + v
		mm_real_axjpy (nu_, tmp_n_, 0, b_);   // b = c + mu * (s + u) + nu * (t + v)
	}
}

// Update primal variable beta
void ADMM::update_beta ()
{
	if (beta_) mm_real_memcpy (beta_prev_, beta_);
	solve_with_SMW (mu_ + nu_, b_, beta_);
}

// Update slack variable s
void ADMM::update_slack_s ()
{
	double ck = mu_ / (mu_ + (1. - alpha_) * lambda_);

#pragma omp parallel for
	for (size_t j = 0; j < n_; j++) {
		double rj = beta_->data[j] - u_->data[j];
		double cj = soft_threshold (rj, alpha_ * lambda_ / mu_);
		s_->data[j] = ck * cj; 
	}
}

// Update dual variable u
void ADMM::update_dual_u ()
{
	for (size_t j = 0; j < n_; j++)
		u_->data[j] += mu_ * (s_->data[j] - beta_->data[j]);
}

// Update slack variable t
void ADMM::update_slack_t ()
{
#pragma omp parallel for
	for (size_t j = 0; j < n_; j++) {
		double lj = lower_->data[j];
		double qj = beta_->data[j] - v_->data[j];
		t_->data[j] = (lj < qj) ? qj : lj;
	}
}

// Update dual variable v
void ADMM::update_dual_v ()
{
	for (size_t j = 0; j < n_; j++) {
		v_->data[j] += nu_ * (t_->data[j] - beta_->data[j]);
	}
}

// Factorize matrix C = (X * X.T / coef + I)^-1
void ADMM::factorize_C ()
{
	if (Ci_ == nullptr) Ci_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m_, m_, m_ * m_);

	double coef = mu_ + nu_;
	mm_real_x_dot_y (false, true, 1. / coef, X_, X_, 0., Ci_);
	for (size_t j = 0; j < m_; j++) Ci_->data[j + j * m_] += 1.;

	factorize_matrix (Ci_);
}

// Evaluate primal and dual residuals
double ADMM::eval_residuals ()
{
	double d1 = 0., d2 = 0., d3 = 0.;

	// ds = s - beta
	mm_real_memcpy (tmp_n_, s_);
	mm_real_axjpy (-1., beta_, 0, tmp_n_);
	d1 = mm_real_xj_nrm2 (tmp_n_, 0) / sqrt ((double) tmp_n_->m);

	// db = mu * (beta - beta_prev)
	mm_real_memcpy (tmp_n_, beta_);
	mm_real_axjpy (-1., beta_prev_, 0, tmp_n_);
	mm_real_scale (tmp_n_, mu_);
	d2 = mm_real_xj_nrm2 (tmp_n_, 0) / sqrt ((double) tmp_n_->m);

	if (nu_ > std::numeric_limits<double>::epsilon ()) {
		// dt = t - beta
		mm_real_memcpy (tmp_n_, t_);
		mm_real_axjpy (-1., beta_, 0, tmp_n_);
		d3 = mm_real_xj_nrm2 (tmp_n_, 0) / sqrt ((double) tmp_n_->m);
	}

	return std::max (std::max (d1, d2), d3);
}

// Factorize a real-symmetry matrix using Cholesky
void ADMM::factorize_matrix (mm_real *C)
{
	if (!mm_real_is_symmetric (C)) mm_real_general_to_symmetric ('U', C);
	cholfact_ (C);
}

/****** private ******/

// Normalize columns of matrix X
mm_real *ADMM::normalize_matrix (mm_real *X)
{
	mm_real *w = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, X->n, 1, X->n);
	for (size_t j = 0; j < X->n; j++) {
		double	norm = mm_real_xj_nrm2 (X_, j);
		if (norm < std::numeric_limits<double>::epsilon ())
			throw std::runtime_error ("Column norm is zero in normalize_matrix(): column = " + std::to_string (j));	
		w->data[j] = norm;
		mm_real_xj_scale (X, j, 1. / w->data[j]);
	}

	FILE	*fp = fopen ("w.vec", "w");
	if (fp) {
		mm_real_fwrite (fp, w, "%.12e");
		fclose (fp);
	}

	return w;
}

// Soft-thresholding operator
double ADMM::soft_threshold (double gamma, double lambda)
{
	double sign = (gamma >= 0.) ? 1. : -1.;
	double ci = fabs (gamma) - lambda;
	return sign * ((ci >= 0) ? ci : 0.);
}

// Solve a symmetric positive definite linear system using its Cholesky factorization.
// The input matrix Ci must have been factorized by dpotrf beforehand.
void
ADMM::cholsolve (mm_real *Ci, mm_real *b)
{
	MKL_INT	info;
	char		uplo = (Ci->symm == MM_REAL_SYMMETRIC_UPPER) ? 'U' : 'L';
	MKL_INT	m = (MKL_INT) Ci->m;
	MKL_INT	nrhs = 1;
	MKL_INT	lda = (MKL_INT) Ci->m;
	MKL_INT	ldb = (MKL_INT) b->m;
	info = LAPACKE_dpotrs (LAPACK_COL_MAJOR, uplo, m, nrhs, Ci->data, lda, b->data, ldb);
	if (info != 0) throw std::runtime_error ("Cholesky factorization (dpotrs) failed.");
}

// Solve with Sherman–Morrison–Woodbury formula
void ADMM::solve_with_SMW (double coef, mm_real *b, mm_real *x)
{
	if (x == nullptr) throw std::runtime_error ("ERROR: vector x is empty.");
	if (x->m != n_) throw std::runtime_error ("ERROR: Dimension of vector x incompatible.");

	// tmp_m = X * b
	mm_real_x_dot_yk (false, 1., X_, b, 0, 0., tmp_m_);

	// tmp_m = Ci * (X * b)
	cholsolve (Ci_, tmp_m_);

	// x = -X.T * Ci * X * b / coef
	mm_real_x_dot_yk (true, -1. / coef, X_, tmp_m_, 0, 0., x);

	// x = b - X.T * Ci * X * b / coef
	mm_real_axjpy (1., b, 0, x);

	// Scale by 1 / coef
	if (coef > 1.) mm_real_xj_scale (x, 0, 1. / coef);
}

// Factorize matrix using Cholesky decomposition
void ADMM::cholfact_ (mm_real *C)
{
	MKL_INT	info;
	char		uplo = (C->symm == MM_REAL_SYMMETRIC_UPPER) ? 'U' : 'L';
	MKL_INT	m = (MKL_INT) C->m;
	MKL_INT	lda = (MKL_INT) C->m;
	info = LAPACKE_dpotrf (LAPACK_COL_MAJOR, uplo, m, C->data, lda);
	if (info != 0) throw std::runtime_error ("Cholesky factorization (dpotrf) failed");
}

