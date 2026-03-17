#ifndef _ADMM_ADA_L1_TV_H_
#define _ADMM_ADA_L1_TV_H_

/*
	ADMM_AdaL1TV: ADMM solver for adaptive L1-TV regularization.

	Objective:
	    minimize   (1/2) * ||X * beta - y||_2^2 + lambda * Σ_i w_i |D_i * beta|

	Features:
	- Adaptive weights for differential operators
	- 3D derivative operators Dx, Dy, Dz
	- Linear solvers: SMW or CG (with optional diagonal preconditioning)
 */
class ADMM_AdaL1TV : public ADMM {

	size_t		k_ = 4; // dim(D) = k * n × n  (identity + Dx + Dy + Dz)

	// Adaptive weights for differential operators (length = n)
	double		*adap_w_ = nullptr;

	// Differential operators
	mm_real		*Dx_ = nullptr;
	mm_real		*Dy_ = nullptr;
	mm_real		*Dz_ = nullptr;
	/*
		P = mu * D.T * D + (mu + nu) * I
		    = mu * (Dx.T * Dx + Dy.T * Dy + Dz.T * Dz) + (mu + nu) * I
	*/
	bool			P_is_modified_ = false;
	mm_real		*P_ =  nullptr;  // A symmetric sparse matrix P = mu * D.T * D + (mu + nu) * I
	mm_real		*invP_XT_ = nullptr;  // Solution of a linear system X.T = inv(P) * invP_XT
	PardisoSolver	*solverP_ = nullptr;  // Pardiso solver for linear system P * x = b

	mm_real		*d_ = nullptr; // d = D * beta, dim(d) = k * n

	// Temporary vector of size k * n
	mm_real		*tmp_kn_ = nullptr;

	DiffOp		*diff_ = nullptr;

public:
	ADMM_AdaL1TV (size_t m, size_t n, double log10_lambda);
	~ADMM_AdaL1TV ();

	// Set adaptive weighting parameter gamma and initial model
	void		setAdaptiveWeighting (double sigma, double gamma, const mm_real *guide_model, const double c1 = 0.1, const double c2 = 0.01);

	// Set TV regularization parameters and differential operators
	void		setTVRegularization (double mu, size_t nx, size_t ny, size_t nz);

	// Run ADMM iterations until convergence
	size_t	solve (size_t maxiter, double tol);

	// Return d = [I, Dx, Dy, Dz]^T * beta
	mm_real	*getRegularizationVector () { return d_; }

protected:
	// One ADMM iteration
	void		iterate ();

	// Update primal and dual vectors
	void		update_rhs ();     // update right-hand side vector b
	void		update_beta ();    // update solution vector beta
	void		update_slack_s (); // update slack vector s
	void		update_dual_u ();  // update dual vector u

	// Evaluate primal and dual residuals
	double	eval_residuals ();

	// Export adaptive weights to files
	void		export_adaptive_weight_ ();

private:
	// Apply linear operator D: returns vector of size k * n
	mm_real	*D_dot_y (const mm_real *y);

	// Apply transpose D.T: returns vector of size n
	mm_real	*DT_dot_y (const mm_real *y);

	// SMW solver related
	void		factorize_C (); // C = I + X * Pi * X.T
	void		factorize_P (); // P = mu * D.T * D + nu * I
	void		solve_with_SMW (mm_real *b, mm_real *x);
};

#endif // _ADMM_ADA_L1_TV_H_

