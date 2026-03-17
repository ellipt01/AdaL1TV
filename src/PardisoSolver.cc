#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm> // For std::fill
#include <mmreal.h>
#include "PardisoSolver.h"

/*** Public Methods ***/

PardisoSolver::PardisoSolver ()
{
	iparm_.fill(0);
	iparm_[0] = 0;  // use MKL defaults
}

PardisoSolver::~PardisoSolver ()
{
	if (is_factorized_) {
		MKL_INT	phase	= -1;
		MKL_INT	error	= 0;
		MKL_INT	nrhs	= 1; // Dummy value for cleanup phase
		pardiso (pt_handle_.data (), &maxfct_, &mnum_, &mtype_, &phase, &n_,
				 nullptr, nullptr, nullptr, nullptr, &nrhs,
				 iparm_.data (), &msglvl_, nullptr, nullptr, &error);
	}
}

void
PardisoSolver::setParam (PardisoParam param_type, MKL_INT value)
{
	switch (param_type) {
		case PardisoParam::mtype:	mtype_ = value; break;
		case PardisoParam::maxfct:	maxfct_ = value; break;
		case PardisoParam::mnum:	mnum_ = value; break;
		case PardisoParam::msglvl:	msglvl_ = value; break;
	}
}

void
PardisoSolver::setIparm (MKL_INT index, MKL_INT value)
{
	if (index < 0 || index >= 64) {
		throw std::out_of_range ("Invalid iparm index. Must be 0 <= index < 64.");
	}
	iparm_[0] = 1; // Ensure user settings are active.
	iparm_[index] = value;
}

void
PardisoSolver::factorize (const mm_real* mat_csc, bool verbose)
{
	if (!mat_csc) {
		throw std::invalid_argument ("Input CSC matrix is null.");
	}
	if (mtype_ == 0)
		throw std::runtime_error("PARDISO mtype not set.");
   
	n_ = mat_csc->n;

	convertCscToCsr (mat_csc);

	MKL_INT	phase	= 11;
	MKL_INT	error	= 0;
	MKL_INT	nrhs	= 1; // Dummy value for analysis/factorization

	// --- Phase 11: Analysis ---
	pardiso (pt_handle_.data (), &maxfct_, &mnum_, &mtype_, &phase, &n_,
			 mat_csr_.data.data (), mat_csr_.p.data (), mat_csr_.j.data (), nullptr, &nrhs,
			 iparm_.data (), &msglvl_, nullptr, nullptr, &error);
	if (error != 0) {
		throw std::runtime_error ("PARDISO error in phase 11 (Analysis): code " + std::to_string (error));
	}
	if (verbose) std::cerr << "Phase 11 (Analysis)... OK" << std::endl;

	// --- Phase 22: Numerical Factorization ---
	phase = 22;
	pardiso (pt_handle_.data (), &maxfct_, &mnum_, &mtype_, &phase, &n_,
			 mat_csr_.data.data (), mat_csr_.p.data (), mat_csr_.j.data (), nullptr, &nrhs,
			 iparm_.data (), &msglvl_, nullptr, nullptr, &error);
	if (error != 0) {
		throw std::runtime_error ("PARDISO error in phase 22 (Factorization): code " + std::to_string (error));
	}
	is_factorized_ = true;
	if (verbose) std::cerr << "Phase 22 (Factorization)... OK" << std::endl;
}

mm_real *
PardisoSolver::solve (const mm_real* vec_rhs, bool verbose)
{
	if (!is_factorized_) {
		throw std::runtime_error ("Matrix has not been factorized. Call factorize() first.");
	}
	if (!vec_rhs) {
		throw std::invalid_argument ("RHS vector is null.");
	}

	// Check if the dimension of the RHS vector matches the matrix dimension.
	if (vec_rhs->m != n_) {
		throw std::invalid_argument ("RHS vector dimension (" + std::to_string (vec_rhs->m) +
									 ") does not match the matrix dimension (" + std::to_string (n_) + ").");
	}

	MKL_INT	nrhs	= vec_rhs->n;
	mm_real*	x = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n_, nrhs, n_ * nrhs);

	// --- Phase 33: Solve ---
	MKL_INT	phase	= 33;
	MKL_INT	error	= 0;
	pardiso (pt_handle_.data (), &maxfct_, &mnum_, &mtype_, &phase, &n_,
			 mat_csr_.data.data (), mat_csr_.p.data (), mat_csr_.j.data (), nullptr, &nrhs,
			 iparm_.data (), &msglvl_, vec_rhs->data, x->data, &error);
	if (error != 0) {
		mm_real_free (x); // Clean up allocated memory on failure
		throw std::runtime_error ("PARDISO error in phase 33 (Solve): code " + std::to_string (error));
	}

	if (verbose) std::cerr << "Phase 33 (Solve)... OK" << std::endl;
	return x;
}

/*** Private Methods ***/

void
PardisoSolver::convertCscToCsr (const mm_real* mat_csc)
{
	mat_csr_.m = mat_csc->m;
	mat_csr_.n = mat_csc->n;
	mat_csr_.nnz = mat_csc->nnz;
	mat_csr_.p.resize (mat_csr_.m + 1);
	mat_csr_.j.resize (mat_csr_.nnz);
	mat_csr_.data.resize (mat_csr_.nnz);

	// --- Pass 1: Calculate row counts ---
	std::fill (mat_csr_.p.begin (), mat_csr_.p.end (), 0);
	for (size_t k = 0; k < mat_csc->nnz; ++k) {
		mat_csr_.p[mat_csc->i[k] + 1]++;
	}

	// --- Create row pointers via cumulative sum ---
	for (size_t i = 1; i <= mat_csr_.m; ++i) {
		mat_csr_.p[i] += mat_csr_.p[i - 1];
	}

	// --- Pass 2: Populate CSR arrays ---
	std::vector<MKL_INT>	row_slots = mat_csr_.p; // Copy p to use as write pointers
	for (size_t j = 0; j < mat_csc->n; ++j) {
		for (size_t k = mat_csc->p[j]; k < mat_csc->p[j + 1]; ++k) {
			size_t	i = mat_csc->i[k];
			size_t	dest_idx = row_slots[i]++;
			mat_csr_.j[dest_idx] = j;
			mat_csr_.data[dest_idx] = mat_csc->data[k];
		}
	}
}
