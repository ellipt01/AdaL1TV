#ifndef _ADMM_H_
#define _ADMM_H_

/*
	Alternating Direction Method of Multipliers (ADMM) base class.

	This class provides a general ADMM framework for solving inverse problems of the form:

	minimize   0.5 * || f - X * beta ||_2^2 + λ R(beta)

	where
	    f     : data vector
	    X     : sensitivity (forward operator) matrix
	    beta  : model vector
	    R(.)  : regularization term (e.g., L1-L2, bound constraints)

	The method introduces slack variables and dual variables to handle constraints,
	and updates them iteratively according to the ADMM scheme:

	    primal update   (beta, s, t)
	    dual update     (u, v)

	Convergence is monitored via primal and dual residuals, following the KKT conditions.
*/

class ADMM {
protected:
	size_t	m_ = 0; // number of data points (dim(f))
	size_t	n_ = 0; // number of model parameters (dim(beta))

	double	alpha_ = 0.;  // mixing ratio between data misfit and regularization
	double	lambda_ = 0.; // regularization weight (λ)

	// f and X are not owned by ADMM. Caller must manage lifetime.
	mm_real	*f_ = nullptr; // observed data vector f (dim m)
	mm_real	*X_ = nullptr; // sensitivity matrix X (dim m × n)

	mm_real	*w_ = nullptr; // optional sensitivity weighting

	mm_real	*beta_ = nullptr;      // current model vector beta
	mm_real	*beta_prev_ = nullptr; // previous model vector (for monitoring updates)

	// --- L1-L2 regularization (soft-threshold penalty) ---
	double	mu_ = 0.; // penalty parameter μ for L1-L2 term
	mm_real	*s_ = nullptr; // slack variable for L1-L2: enforces s ≈ beta
	mm_real	*u_ = nullptr; // dual variable (Lagrange multiplier) for L1-L2

	// --- Bound constraints (box constraints: lower ≤ beta ≤ upper) ---
	double	nu_ = 0.; // penalty parameter ν for bound constraints
	mm_real	*t_ = nullptr; // slack variable for bounds: enforces t ≈ beta
	mm_real	*v_ = nullptr; // dual variable (Lagrange multiplier) for bounds

	// Bounds
	mm_real	*lower_ = nullptr; // element-wise lower bounds
	mm_real	*upper_ = nullptr; // element-wise upper bounds

	// --- Right-hand side of the linear system ---
	mm_real	*c_ = nullptr; // c = X^T f
	mm_real	*b_ = nullptr; // b = c + μ(s + u) + ν(t + v), updated each iteration

	mm_real	*Ci_ = nullptr; // factorize of matrix (X^T X + ...) used in data-space inversion

	double	residuals_ = 0.; // maximum of primal and dual residuals (for convergence check)

	bool		verbose_ = false; // verbose mode (print iteration logs)

	// Temporary work vectors for Sherman-Morrison-Woodbury (SMW) inversion
	mm_real	*tmp_m_ = nullptr; // dim m
	mm_real	*tmp_n_ = nullptr; // dim n

public:
	ADMM () { }
	~ADMM ();

	ADMM (const ADMM&) = delete;
	ADMM& operator = (const ADMM&) = delete;

	ADMM (size_t m, size_t n, double alpha, double log10_lambda);

	// Setup linear system: normalize matrix if requested
	void		setupLinearSystem (mm_real *f, mm_real *X, bool do_normalize);

	// Enable L1-L2 regularization with penalty parameter μ
	void		setL1L2Regularization (double mu);

	// Enable bound constraints with penalty parameter ν and given limits
	void		setBoundConstraint (double nu, double lower, double upper);

	// Solve inverse problem via ADMM iterations:
	//   Iterate until maxiter or until primal/dual residuals < tol
	size_t	solve (size_t maxiter, double tol);

	// Return the estimated model beta.
	// If remove_weight is true, result is converted back to physical units (A/m).
	mm_real	*getModel (bool remove_weight = true);

	// Return current residual (max of primal and dual residuals)
	double	getResiduals () { return residuals_; }

	// Reconstruct predicted anomaly f_hat = X * beta
	mm_real	*recover ();

	// Enable verbose logging
	void		setVerbose () { verbose_ = true; }

protected:
	void		iterate (); // one ADMM iteration

	void		update_rhs ();   // update RHS vector b
	void		update_beta ();  // update primal variable beta
	void		update_slack_s (); // update slack variable s
	void		update_dual_u ();  // update dual variable u
	void		update_slack_t (); // update slack variable t
	void		update_dual_v ();  // update dual variable v

	void		factorize_C (); // factorize matrix C

	double	eval_residuals (); // compute primal and dual residual norms

	void		factorize_matrix (mm_real *C); // general matrix factorization based on Cholesky

protected:
	mm_real	*normalize_matrix (mm_real *X); // normalize sensitivity matrix X
	double	soft_threshold (double gamma, double lambda); // soft-threshold operator: prox for L1
	void		cholsolve (mm_real *Ci, mm_real *b); // solve simultaneous equation using Cholesky decomposition
	void		solve_with_SMW (double coef, mm_real *b, mm_real *x); // solve system using SMW identity

private:
	void		cholfact_ (mm_real *C); // factorize matrix Cholesky decomposition

};

#endif // _ADMM_H_

