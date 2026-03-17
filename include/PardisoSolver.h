#ifndef PARDISO_SOLVER_H
#define PARDISO_SOLVER_H

#include <vector>
#include <array>
#include <cstddef> // For size_t
#include <mkl.h>
#include <mmreal.h>

/**
 * @file
 * @brief Declares the PardisoSolver class, a C++ wrapper for the MKL PARDISO solver.
 */

// A modern C++ struct for CSR matrix data using std::vector for RAII.
struct MklCsrMatrix {
	size_t			m = 0;
	size_t			n = 0;
	size_t			nnz = 0;
	std::vector<MKL_INT>	p;    // Row pointers (ia)
	std::vector<MKL_INT>	j;    // Column indices (ja)
	std::vector<double>	data; // Non-zero values (a)
};

// Enum for setting PARDISO parameters in a type-safe way.
enum class PardisoParam {
	mtype,
	maxfct,
	mnum,
	msglvl
};

/**
 * @class PardisoSolver
 * @brief A C++ class to encapsulate the PARDISO solver's state and logic.
 *
 * This class manages the lifecycle of the PARDISO solver, including matrix
 * factorization and solving linear systems. It uses RAII to ensure that
 * resources are automatically released.
 */
class PardisoSolver {
private:
	// --- Member Variables ---
	MKL_INT				n_ = 0; // Dimension of the matrix, set during factorization.

	// PARDISO internal data handle and control arrays.
	std::array<void*, 64>		pt_handle_{};
	std::array<MKL_INT, 64>	iparm_{};

	// PARDISO control parameters.
	MKL_INT				mtype_ = 0;
	MKL_INT				maxfct_ = 1;
	MKL_INT				mnum_ = 1;
	MKL_INT				msglvl_ = 0;

	// The solver owns and manages its internal CSR matrix.
	MklCsrMatrix			mat_csr_;

	// Flag to ensure factorization is complete before solving.
	bool					is_factorized_ = false;

public:
	// Default constructor.
	PardisoSolver ();
	// Destructor automatically handles resource cleanup.
	~PardisoSolver ();

	// Prohibit copying to prevent issues with resource management.
	PardisoSolver (const PardisoSolver&) = delete;
	PardisoSolver& operator = (const PardisoSolver&) = delete;

	/**
	 * @brief Analyzes and factorizes the given sparse matrix.
	 * @param mat_csc The input matrix in CSC format. The solver does not take ownership.
	 * @param verbose If true, prints progress messages to stderr.
	 */
	void factorize (const mm_real* mat_csc, bool verbose = false);

	/**
	 * @brief Solves the system Ax=b for a given right-hand side vector.
	 * @param vec_rhs The right-hand side vector/matrix 'b'.
	 * @return A new mm_real object containing the solution 'x'. The caller is
	 * responsible for freeing this object using mm_real_free().
	 * @param verbose If true, prints progress messages to stderr.
	 */
	mm_real* solve (const mm_real* vec_rhs, bool verbose = false);

	// Sets a single PARDISO parameter.
	void setParam (PardisoParam param_type, MKL_INT value);
	// Sets a value in the iparm array by index.
	void setIparm (MKL_INT index, MKL_INT value);

private:
	// Converts a CSC matrix to the internal CSR format.
	void convertCscToCsr (const mm_real* mat_csc);
};

#endif // PARDISO_SOLVER_H
