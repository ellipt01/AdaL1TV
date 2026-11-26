#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <errno.h>
#include <unistd.h>
#include <mgcal.h>
#include <mmreal.h>
#include "Kernel.h"
#include "MagKernel.h"
#include "DiffOp.h"
#include "PardisoSolver.h"
#include "ADMM.h"
#include "ADMM_AdaL1TV.h"
#include "L1L2.h"
#include "AdaL1TV.h"

/****** public ******/

// Export recovered anomaly and model (+ regularization vector)
void
AdaL1TV::exportResults (const char *ofn_model, const char *ofn_recovered)
{
	// model / recovered anomaly は基底に任せる
	L1L2::exportResults (ofn_model, ofn_recovered);

	// admm_ は start_ADMM で必ず ADMM_AdaL1TV を入れているので static_cast で安全
	ADMM_AdaL1TV	*a = static_cast<ADMM_AdaL1TV *> (admm_);
	mm_real		*d = a->getRegularizationVector ();
	FILE			*fp = fopen ("regularization.vec", "w");
	if (fp) {
		mm_real_fwrite (fp, d, "%.6e");
		fclose (fp);
	} else {
		std::cerr << "WARNING: cannot open regularization.vec for writing" << std::endl;
	}
}

/****** protected ******/

// Start ADMM iteration until convergence
void
AdaL1TV::start_ADMM (mm_real *f, mm_real *K)
{
	size_t m = f->m;
	size_t n = K->n;

	ADMM_AdaL1TV	*a = new ADMM_AdaL1TV (m, n, aopts_.lambda);
	admm_ = a;		// 基底の ADMM * に格納(delete は基底のデストラクタ)

	if (opts_.verbose) a->setVerbose ();

	a->setupLinearSystem (f, K, true);
	a->setTVRegularization (opts_.mu, opts_.nx, opts_.ny, opts_.nz);
	if (opts_.nu > 0.) a->setBoundConstraint (opts_.nu, opts_.lower, opts_.upper);

	if (!aopts_.guidefn.empty ()) {
		FILE *fp = fopen (aopts_.guidefn.c_str (), "r");
		if (!fp)
			throw std::runtime_error ("cannot open specified guide model file " + aopts_.guidefn);
		mm_real *guide_model = mm_real_fread (fp);
		fclose (fp);
		if (!guide_model)
			throw std::runtime_error ("cannot read guide model file " + aopts_.guidefn);
		a->setAdaptiveWeighting (aopts_.contrast, aopts_.sigma, guide_model, aopts_.c1, aopts_.c2);
		mm_real_free (guide_model);
	}

	// Print settings to stderr
	export_settings (stderr);

	size_t niter = a->solve (opts_.maxiter, opts_.tolerance);
	fprintf (stderr, "residual[%04zu] = %.4e / %.4e\n", niter, a->getResiduals (), opts_.tolerance);
}

// Export settings to stream
void
AdaL1TV::export_settings (FILE *stream)
{
	const bool	adaptive = !aopts_.guidefn.empty ();

	fprintf (stream, "#################################################################\n");
	fprintf (stream, "# *** penalty type: %s ***\n", adaptive ? "AdaL1TV" : "L1TV");
	fprintf (stream, "# \n");
	fprintf (stream, "# num of grid cells:nx, ny, nz: %d, %d, %d\n", opts_.ny, opts_.nx, opts_.nz);
	fprintf (stream, "# range of the model space:\n");
	fprintf (stream, "#    x range (south-north):     [%.2f, %.2f]\n", opts_.yy[0], opts_.yy[1]);
	fprintf (stream, "#    y range (west-east):       [%.2f, %.2f]\n", opts_.xx[0], opts_.xx[1]);
	fprintf (stream, "#    z range (top-bottom):      [%.2f, %.2f]\n", - opts_.zz[0], - opts_.zz[1]);
	fprintf (stream, "# exf. and mag. inc, dec.:      %.2f, %.2f, %.2f, %.2f\n",
		opts_.exf_inc, opts_.exf_dec, opts_.mgz_inc, opts_.mgz_dec);
	fprintf (stream, "# tolerance and num of maxiter: %.2e, %zu\n", opts_.tolerance, opts_.maxiter);
	fprintf (stream, "# penalty parameter:mu:         %.2f\n", opts_.mu);
	fprintf (stream, "# penalty parameter:nu:         %.2f\n", opts_.nu);
	if (opts_.nu > 0.) {
		fprintf (stream, "# lower and upper bounds:       %.2f, %.2f\n", opts_.lower, opts_.upper);
	}
	fprintf (stream, "# \n");
	fprintf (stream, "# input anomaly data file name: %s\n", opts_.infn.c_str ());
	fprintf (stream, "# log10(lambda):                %.4f\n", aopts_.log10_lambda);
	fprintf (stream, "# apply adaptive weighting:     %s\n", adaptive ? "true" : "false");
	if (adaptive) {
		fprintf (stream, "# guide model file name:        %s\n", aopts_.guidefn.c_str ());
		fprintf (stream, "# penalty contrast:             %.3f\n", aopts_.contrast);
	}
	fprintf (stream, "# verbose mode:                 %s\n", opts_.verbose ? "true" : "false");
	fprintf (stream, "#################################################################\n");
}
