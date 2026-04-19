#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <cstdio>
#include <mkl.h>
#include <omp.h>
#include <mmreal.h>
#include "DiffOp.h"
#include "PardisoSolver.h"
#include "ADMM.h"
#include "ADMM_AdaL1TV.h"

/****** public ******/

// Destructor: release allocated memory and resources
ADMM_AdaL1TV::~ADMM_AdaL1TV ()
{
	if (adap_w_) delete [] adap_w_; // adaptive weight vector
	if (P_) mm_real_free (P_); // inverse operator for SMW
	if (invP_XT_) mm_real_free (invP_XT_); // = inv(P) * XT
	if (solverP_) delete solverP_;   // PardisoSolver
	if (d_) mm_real_free (d_);   // D * beta temporary
	if (tmp_kn_) mm_real_free (tmp_kn_); // temporary vector of size k * n
	if (diff_) delete diff_;
	if (Dx_) mm_real_free (Dx_);
	if (Dy_) mm_real_free (Dy_);
	if (Dz_) mm_real_free (Dz_);
}

// Constructor: initialize problem size and regularization parameter
ADMM_AdaL1TV::ADMM_AdaL1TV (size_t m, size_t n, double log10_lambda)
{
	if (m <= 0 || n <= 0) throw std::runtime_error ("Invalid system size: both m and n must be positive.");

	m_ = m;
	n_ = n;
	
	// lambda = 10^(log10_lambda)
	lambda_ = pow (10., log10_lambda);
}

// Set up total variation (TV) regularization matrices
// Dx, Dy, Dz are discrete differential operators in x, y, z directions
void
ADMM_AdaL1TV::setTVRegularization (double mu, size_t nx, size_t ny, size_t nz)
{
	if (mu <= std::numeric_limits<double>::epsilon ())
		throw std::runtime_error ("Invalid parameter mu: must be greater than zero.");

	if (nx * ny * nz != n_) throw std::runtime_error ("Dimension mismatch: differential operators must have column size = n.");

	if (diff_ == nullptr) diff_ = new DiffOp ();

	/* Construct operator Dx, Dy, Dz */
	if (Dx_) mm_real_free (Dx_);
	Dx_ = diff_->createDiffX (nx, ny, nz);
	if (Dy_) mm_real_free (Dy_);
	Dy_ = diff_->createDiffY (nx, ny, nz);
	if (Dz_) mm_real_free (Dz_);
	Dz_ = diff_->createDiffZ (nx, ny, nz);

	// Apply optional weighting to differential operators
	if (w_) {
#pragma omp parallel for
		for (size_t j = 0; j < n_; j++) {
			mm_real_xj_scale (Dx_, j, 1. / w_->data[j]);
			mm_real_xj_scale (Dy_, j, 1. / w_->data[j]);
			mm_real_xj_scale (Dz_, j, 1. / w_->data[j]);
		}
	}

	// Construct (weighted) Laplacian P_ = Dx.T * Dx + Dy.T * Dy + Dz.T * Dz
	mm_real	*Lx; // = Dx.T * Dx
	mm_real	*Ly; // = Dy.T * Dy
	mm_real	*Lz; // = Dz.T * Dz
	if (w_) {
		Lx = diff_->createColWeightedLaplacianX (nx, ny, nz, w_->data);
		Ly = diff_->createColWeightedLaplacianY (nx, ny, nz, w_->data);
		Lz = diff_->createColWeightedLaplacianZ (nx, ny, nz, w_->data);
	} else {
		Lx = diff_->createLaplacianX (nx, ny, nz);
		Ly = diff_->createLaplacianY (nx, ny, nz);
		Lz = diff_->createLaplacianZ (nx, ny, nz);
	}
	P_ = diff_->addLaplacians (Lx, Ly, Lz);
	mm_real_free (Lx);
	mm_real_free (Ly);
	mm_real_free (Lz);

	mm_real_general_to_symmetric ('U', P_);

	// prepare slack variable s and dual variable u for TV penalty
	mu_ = mu;

	// s = D * beta, dimension = k*n × 1
	if (s_) mm_real_free (s_);
	s_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, k_ * n_, 1, k_ * n_);
	mm_real_set_all (s_, 0.);

	// Lagrange multiplier u
	if (u_) mm_real_free (u_);
	u_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, k_ * n_, 1, k_ * n_);
	mm_real_set_all (u_, 0.);
}

// Solve equation by runing ADMM iteration until convergence
// maxiter : maximum iterations
// tol     : tolerance for residual stopping criterion
size_t
ADMM_AdaL1TV::solve (size_t maxiter, double tol)
{
	if (m_ <= 0 || n_ == 0)
		throw std::runtime_error ("Problem size is undefined or invalid. Call constructor with valid dimensions.");
	if (f_ == nullptr || X_ == nullptr)
		throw std::runtime_error ("System matrix or RHS vector is not set. Call simeq() before run().");
	if (mu_ <= 0) throw std::runtime_error ("TV regularization is not initialized. Call set_total_variation() first.");

	// Prepare temporary vectors for Sherman-Morrison-Woodbury (SMW) solver
	if (tmp_m_ == nullptr) tmp_m_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m_, 1, m_);
	if (tmp_n_ == nullptr) tmp_n_  = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
	if (tmp_kn_ == nullptr) tmp_kn_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, k_ * n_, 1, k_ * n_);

	/* P = mu * (Lx + Ly * Lz) + (mu + nu) * I */
	if (P_is_modified_ == false) {
		mm_real_scale (P_, mu_);
		for (size_t j = 0; j < P_->n; j++) {
			for (size_t k = P_->p[j]; k < P_->p[j + 1]; k++) {
				if (P_->i[k] == j) {
					P_->data[k] += mu_ + nu_;
					break;
				}
			}
		}
		P_is_modified_ = true;
	}
	factorize_P ();
	if (Ci_ == nullptr) factorize_C ();
	
	// Main ADMM loop
	size_t	niter = 0;
	while (true) {
		iterate ();                   // perform one ADMM update cycle
		residuals_ = eval_residuals (); // evaluate primal/dual residuals

		if (residuals_ < tol) break;     // stop if converged

		if (verbose_ && niter % 100 == 0)
			fprintf (stderr, "residual[%04zu] = %.4e / %.4e\n", niter, residuals_, tol);

		if (++niter >= maxiter) break;
	}
	return niter;
}

/* Set up adaptive weighting for AdaL1TV */
// wj = ( s / ( | dj | + eps))^gamma, where d = D * guide,
// mm_real *guide is dsensitivity weighting guide model
// s is a median of the active set:
//     | beta_guide | < c1 * sigma, | D * beta_guide | < c2 * sigma
void
ADMM_AdaL1TV::setAdaptiveWeighting (double sigma, double gamma, const mm_real *guide_model, const double c1, const double c2)
{
	if (guide_model->m != n_)
		throw std::runtime_error ("size of guide model incompatible.");

	double	eps = 1.e-3;
	if (adap_w_) delete [] adap_w_;
	size_t	size = k_ * n_;
	adap_w_ = new double [size];

	/*** construct base weight ***/
	mm_real	*d = D_dot_y (guide_model); // = D * beta

	/*** scaling: scale weight by the mean of nonzero | vk | ***/

	// | beta |
	double	*absd = new double [n_];
	for (size_t i = 0; i < n_; i++) absd[i] = fabs (d->data[i]);
	LAPACKE_dlasrt ('D', (MKL_INT) n_, absd);
	size_t	count = 0;
	for (size_t i = 0; i < n_; i++) {
		if (absd[i] <= c1 * sigma) break;
		count++;
	}
	if (count == 0) throw std::runtime_error ("No elements exceed the threshold (c1 * sigma). Cannot compute median.");
	double	median = absd[count / 2];
	if (median <= 0.) median = c1 * sigma;
	// Adaptive weighting for |D * beta | based on | D * beta* |
#pragma omp parallel for
	for (size_t i = 0; i < n_; i++) {
		// wj = s / (| d_guide_j | + eps )^gamma
		double	val = median / (fabs (d->data[i]) + eps);
		adap_w_[i] = pow (val, gamma);
	}
#ifdef DEBUG
	std::cout << "0: count = " << count << ", median = " << median << std::endl;
#endif
	delete [] absd;

	// | D * beta |
	size = (k_ - 1) * n_;
	absd = new double [size];
	for (size_t i = 0; i < size; i++) absd[i] = fabs (d->data[i + n_]);
	LAPACKE_dlasrt ('D', (MKL_INT) (k_ - 1) * n_, absd);
	count = 0;
	for (size_t i = 0; i < (k_ - 1) * n_; i++) {
		if (absd[i] <= c2 * sigma) break;
		count++;
	}
	if (count == 0) throw std::runtime_error ("No elements exceed the threshold (c2 * sigma). Cannot compute median.");
	median = absd[count / 2];
	if (median <= 0.) median = c2 * sigma;
#pragma omp parallel for
	for (size_t i = n_; i < k_ * n_; i++) {
		// wj = s / (| d_guide_j | + eps )^gamma
		double	val = median / (fabs (d->data[i]) + eps);
		adap_w_[i] = pow (val, gamma);
	}
#ifdef DEBUG
	std::cout << "1: count = " << count << ", median = " << median << std::endl;
#endif
	delete [] absd;

	mm_real_free (d);

	export_adaptive_weight_ ();
}

/****** protected/private ******/

// Perform a single ADMM update cycle
void
ADMM_AdaL1TV::iterate ()
{
	// Update right hand side vect
	update_rhs ();
	
	// Update model vector beta
	update_beta ();

	// Update slack vector s and dual vector u
	update_slack_s ();
	update_dual_u ();

	// Update slack vector t and dual vector v if bound constraints are applied
	if (nu_ > std::numeric_limits<double>::epsilon ()) {
		update_slack_t ();
		update_dual_v ();
	}
}

// Update right hand side vector b:
// b = c + mu * D.T * (s + u) + nu * (t + v),
// where c = X.T * f
void
ADMM_AdaL1TV::update_rhs ()
{
	// b = c + mu * D.T * (s + u) + nu * (t + v), c = X.T * f
	if (b_ == nullptr) b_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);

	// c = X.T * f
	if (c_ == nullptr) {
		c_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
		mm_real_x_dot_yk (true, 1., X_, f_, 0, 0., c_);
	}

	// b = c + mu * D.T * (s + u)
	mm_real_memcpy (b_, c_);
	mm_real_memcpy (tmp_kn_, s_);
	mm_real_axjpy (1., u_, 0, tmp_kn_);	// tmp_kn = s + u
	mm_real	*p = DT_dot_y (tmp_kn_);	// p = D.T * (s + u)
	mm_real_axjpy (mu_, p, 0, b_); // b = c + mu * D.T * (s + u)
	mm_real_free (p);

	if (nu_ > std::numeric_limits<double>::epsilon ()) {
		mm_real_memcpy (tmp_n_, t_);
		mm_real_axjpy (1., v_, 0, tmp_n_); // tmp_n = t + v
		mm_real_axjpy (nu_, tmp_n_, 0, b_); // b = c + mu * D.T * (s + u) + nu * (t + v)
	}
}

// Update solution vector beta using conjugate gradient or SMW solver
void
ADMM_AdaL1TV::update_beta ()
{
	if (beta_) mm_real_memcpy (beta_prev_, beta_);
	solve_with_SMW (b_, beta_);

	if (d_) mm_real_free (d_);
	d_ = D_dot_y (beta_);
}

// Update slack vector s:
// s = S( D * beta - u, lambda / mu).
// where S() is soft thresholding function
void
ADMM_AdaL1TV::update_slack_s ()
{
#pragma omp parallel for
	for (size_t j = 0; j < k_ * n_; j++) {
		double	rj = d_->data[j] - u_->data[j];
		double	lj = lambda_ / mu_;
		if (adap_w_) lj *= adap_w_[j];
		s_->data[j] = soft_threshold (rj, lj);
	}
}

// Update Lagrange dual vector u:
// u = u + mu * (s - D * beta)
void
ADMM_AdaL1TV::update_dual_u ()
{
	mm_real_memcpy (tmp_kn_, s_);
	mm_real_axjpy (-1., d_, 0, tmp_kn_);	 // tmp_kn = s - d
	mm_real_axjpy (mu_, tmp_kn_, 0, u_); // u = u + mu * (s - d)
}

// Evaluate primal and dual residuals for stopping criteria
double
ADMM_AdaL1TV::eval_residuals ()
{
	double	d1 = 0.;
	double	d2 = 0.;
	double	d3 = 0.;

	// ds = s - D * beta
	mm_real_memcpy (tmp_kn_, s_);
	mm_real_axjpy (-1., d_, 0, tmp_kn_); // tmp_kn = s - D * beta
	d1 = mm_real_xj_nrm2 (tmp_kn_, 0) / sqrt ((double) tmp_kn_->m);

	// db = beta - beta_prev
	mm_real_memcpy (tmp_n_, beta_);
	mm_real_axjpy (-1., beta_prev_, 0, tmp_n_);
	mm_real_scale (tmp_n_, mu_); // tmp_n = mu * (beta - beta_prev)
	d2 = mm_real_xj_nrm2 (tmp_n_, 0) / sqrt ((double) tmp_n_->m);

	if (nu_ > std::numeric_limits<double>::epsilon ()) {
		// dt = t - beta
		mm_real_memcpy (tmp_n_, t_);
		mm_real_axjpy (-1., beta_, 0, tmp_n_); // tmp_n = t - beta
		d3 = mm_real_xj_nrm2 (tmp_n_, 0) / sqrt ((double) tmp_n_->m);
	}

	return std::max (std::max (d1, d2), d3);
}

// Export adaptive weights to disk for visualization
void
ADMM_AdaL1TV::export_adaptive_weight_ ()
{
	char	fn[256];
	FILE	*fp;
	for (size_t k = 0; k < k_; k++) {
		char	c;
		switch (k) {
			case 0:
				c = '0';
				break;
			case 1:
				c = 'x';
				break;
			case 2:
				c = 'y';
				break;
			case 3:
				c = 'z';
				break;
			default:
				throw std::runtime_error("Invalid k index");
		}
		sprintf (fn, "weight_d%c.vec", c);
		fp = fopen (fn, "w");
		if (!fp) throw std::runtime_error ("export_adaptive_weight_: Failed to open file");

		mm_real	*dk = mm_real_view_array (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_, adap_w_ + k * n_);
		mm_real_fwrite (fp, dk, "%.8e");
		fclose (fp);
		mm_real_free (dk);

	}
}

/****** private ******/

// Compute D * y
//    = [ I * y; Dx * y; Dy * y; Dz * y ],
// where dimension of y = n, and dimension of D * y = k * n
mm_real *
ADMM_AdaL1TV::D_dot_y (const mm_real *y)
{
	mm_real	*d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, k_ * n_, 1, k_ * n_);
	if (!d) throw std::runtime_error ("D_dot_y: Failed to allocate d");

	for (size_t j = 0; j < k_; j++) {
		mm_real	*dj = mm_real_view_array (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_, d->data + j * n_);
		switch (j) {
			case 0:
				// dj = I * y
				mm_real_memcpy (dj, y);
				break;
			case 1:
				// dj = Dx * y
				mm_real_x_dot_yk (false, 1., Dx_, y, 0, 0., dj);
				break;
			case 2:
				// dj = Dy * y
				mm_real_x_dot_yk (false, 1., Dy_, y, 0, 0., dj);
				break;
			case 3:
				// dj = Dz * y
				mm_real_x_dot_yk (false, 1., Dz_, y, 0, 0., dj);
				break;
		}
		mm_real_free (dj);
	}
	return d;
}

// Compute D^T * y
// D.T = [ I, Dx.T, Dy.T, Dz.T ]
// D.T * y = I * y[:n] + Dx.T * y[n:2*n] + Dy.T * y[2*n:3*n] + Dz.T * y[3*n:],
// where dimension of y = k * n, and dimension of D.T * y = n
mm_real *
ADMM_AdaL1TV::DT_dot_y (const mm_real *y)
{
	mm_real	*c = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
	for (size_t j = 0; j < k_; j++) {
		mm_real	*yj = mm_real_view_array (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_, y->data + j * n_);
		switch (j) {
			case 0:
				// c = I * yj
				mm_real_memcpy (c, yj);
				break;
			case 1:
				// c += Dx.T * yj
				mm_real_x_dot_yk (true, 1., Dx_, yj, 0, 1., c);
				break;
			case 2:
				// c += Dy.T * yj
				mm_real_x_dot_yk (true, 1., Dy_, yj, 0, 1., c);
				break;
			case 3:
				// c += Dz.T * yj
				mm_real_x_dot_yk (true, 1., Dz_, yj, 0, 1., c);
				break;
		}
		mm_real_free (yj);
	}
	return c;
}

// Factorize C = I + X * Pi * X.T,
void
ADMM_AdaL1TV::factorize_C ()
{
	if (solverP_ == nullptr) factorize_P ();

	size_t	m = X_->m;
	size_t	n = X_->n;
	size_t	nnz = m * n;

	std::vector<double> Xt_data ((MKL_INT) nnz);
	mkl_domatcopy ('C', 'T', (MKL_INT) m, (MKL_INT) n, 1.0, X_->data, (MKL_INT) m, Xt_data.data (), (MKL_INT) n);
	mm_real	*Xt = mm_real_view_array (MM_REAL_DENSE, MM_REAL_GENERAL, n, m, nnz, Xt_data.data ());
	if (invP_XT_) mm_real_free (invP_XT_);
	invP_XT_ = solverP_->solve (Xt, false);
	mm_real_free (Xt);
	std::vector<double>().swap (Xt_data);

	if (Ci_) mm_real_free (Ci_);
	Ci_ = mm_real_eye (MM_REAL_DENSE, m);
	mm_real_x_dot_y (false, false, 1., X_, invP_XT_, 1., Ci_);

	factorize_matrix (Ci_);
}


// Factorize P = (mu + nu) * I + mu * (Dx.T * Dx + Dy.T * Dy + Dz.T * Dz),
void
ADMM_AdaL1TV::factorize_P ()
{
	if (solverP_ == nullptr) solverP_ = new PardisoSolver ();

	// factorize P
	// If matrix P is general, mtype = 11
	// else if P is symmetric, mtype = 2
	solverP_->setParam (PardisoParam::mtype, 2);
	solverP_->setParam (PardisoParam::maxfct, 1);

	solverP_->setIparm (2, omp_get_max_threads ()); // Use all available OpenMP threads.
	solverP_->setIparm (27, 0); // Use 0-based indexing for CSR arrays.
	solverP_->setIparm (34, 1); // Use 0-based indexing for CSR arrays.
	//solverP_->setIparm (59, 1); // Enable Out-of-Core (OOC) PARDISO if memory is an issue.

	solverP_->factorize (P_, false);
}

// SMW solver: compute (Pi - Pi * K.T * Ci * K * Pi) * b,
// where Ci = (K * Pi * K.T + I)^-1
void
ADMM_AdaL1TV::solve_with_SMW (mm_real *b, mm_real *x)
{
	if (x == nullptr) throw std::runtime_error ("solve_with_SMW: vector x is empty.");

	/*
		H = X.T * X + P,
		H^-1 * b = [ invP - invP * X.T * Ci * X * invP ] * b,
		where Ci = inv( I + X * invP * X.T )
	*/
	if (solverP_ == nullptr) factorize_P ();

	// r = invP * b
	mm_real	*r = solverP_->solve (b, false);

	// tmp_m = X * r
	mm_real_x_dot_yk (false, 1., X_, r, 0, 0., tmp_m_);

	if (Ci_ == nullptr) factorize_C ();

	// tmp_m = Ci * (X * r)
	cholsolve (Ci_, tmp_m_);
		
	// tmp_n = X.T * tmp_m2
	//mm_real_x_dot_yk (true, 1., X_, tmp_m2_, 0, 0., tmp_n_);

	// c = Pi * (X.T * tmp_m2) = (Pi * X.T) * tmp_m2
	//   = invP_XT * tmp_m2
	mm_real	*c = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, 1, n_);
	mm_real_x_dot_yk (false, 1., invP_XT_, tmp_m_, 0, 0., c);

	
	// r = - c + r
	mm_real_axjpy (-1., c, 0, r);
	mm_real_free (c);

	// r = H^-1 * b
	mm_real_memcpy (x, r);
	mm_real_free (r);
}

