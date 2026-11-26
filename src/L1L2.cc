#include <iostream>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <unistd.h>
#include <mgcal.h>
#include <mmreal.h>
#include "Kernel.h"
#include "MagKernel.h"
#include "ADMM.h"
#include "L1L2.h"
#include "AdaL1TVOpts.h"

/****** Public methods ******/

L1L2::~L1L2()
{
	if (magker_) delete magker_;
	if (admm_) delete admm_;
}

// Run L1-L2 ADMM inversion
void
L1L2::solve ()
{
	FILE* fp = fopen (opts_.infn.c_str (), "r");         // Open data file
	if (!fp) throw std::runtime_error ("Cannot open specified input file " + opts_.infn);
	data_ = fread_data_array (fp);
	for (size_t i = 0; i < data_->n; i++) data_->z[i] *= -1.;
	fclose (fp);

	// Create RHS vector as mm_real view
	f_ = mm_real_view_array (MM_REAL_DENSE, MM_REAL_GENERAL, data_->n, 1, data_->n, data_->data);

	// Compute sensitivity (kernel) matrix
	magker_ = new MagKernel (opts_.exf_inc, opts_.exf_dec, opts_.mgz_inc, opts_.mgz_dec);
	magker_->setRange (opts_.nx, opts_.ny, opts_.nz, opts_.xx.data (), opts_.yy.data (), opts_.zz.data (), 1000.);
	if (!opts_.terrainfn.empty ()) set_terrain (opts_.terrainfn.c_str ());

	fp = fopen ("grid.data", "w");
	if (fp) {
		fwrite_grid (fp, magker_->getGrid ());
		fclose (fp);
	}

	magker_->setData (data_);
	K_ = magker_->getKernel ();

	// Start ADMM iteration
	start_ADMM (f_, K_);
}

// Export recovered anomaly and model
void
L1L2::exportResults(const char* ofn_model, const char* ofn_recovered)
{
	FILE* fp = fopen (ofn_model, "w");
	if (!fp) throw std::runtime_error ("Cannot open file " + std::string (ofn_model));

	mm_real	*beta = admm_->getModel ();
	grid		*g = magker_->getGrid ();
	vector3d	*pos = vector3d_new (0., 0., 0.);
	for (size_t n = 0; n < g->n; n++) {
		grid_get_nth (g, n, pos, NULL);
		fprintf (fp, "%.4e\t%.4e\t%.4e\t%.6e\n", pos->x, pos->y, -pos->z, beta->data[n]);
	}
	fclose (fp);
	vector3d_free (pos);
	mm_real_free (beta);

	mm_real* h = admm_->recover ();
	fp = fopen (ofn_recovered, "w");
	if (!fp) throw std::runtime_error ("Cannot open file " + std::string (ofn_recovered));

	for (int i = 0; i < data_->n; i++) {
		fprintf (fp, "%.4e\t%.4e\t%.4e\t%.6e\n", data_->x[i], data_->y[i], -data_->z[i], h->data[i]);
	}
	fclose (fp);
	mm_real_free (h);
}

/****** Protected / private methods ******/

// Set terrain data
void
L1L2::set_terrain (const char *fn)
{
	int		n = opts_.nx * opts_.ny;
	double	*zsurf = new double [n];
	FILE		*fp = fopen (fn, "r");
	if (!fp) throw std::runtime_error ("Cannot open terrain file.");
	char		buf[256];
	size_t	k = 0;
	while (fgets (buf, BUFSIZ, fp) != nullptr) {
		double	x, y, z;
		sscanf (buf, "%lf\t%lf\t%lf", &x, &y, &z);
		zsurf[k++] = -z;
		if (k >= n) break;
	}
	if (k != n) throw std::runtime_error ("Number of terrain data is incompatible with number of grid.");
	magker_->setSurface (zsurf);
	delete [] zsurf;
}

// Start ADMM solver with given RHS and kernel
void
L1L2::start_ADMM (mm_real* f, mm_real* K)
{
	size_t m = f->m;
	size_t n = K->n;

	const L1L2Opts	&lo = static_cast<const L1L2Opts &> (opts_);
	admm_ = new ADMM (m, n, lo.lambda1, lo.lambda2);

	if (opts_.verbose) admm_->setVerbose ();

	admm_->setupLinearSystem (f, K, true);
	admm_->setL1L2Regularization (opts_.mu);
	if (opts_.nu > 0.) admm_->setBoundConstraint (opts_.nu, opts_.lower, opts_.upper);

	// Print settings to stderr
	export_settings (stderr);

	size_t niter = admm_->solve (opts_.maxiter, opts_.tolerance);
	fprintf (stderr, "residual[%04zu] = %.4e / %.4e\n", niter, admm_->getResiduals (), opts_.tolerance);
}

// Export settings to file stream
void
L1L2::export_settings (FILE* stream)
{
	fprintf (stream, "#################################################################\n");
	fprintf (stream, "# *** penalty type: L1L2 ***\n");
	fprintf (stream, "# \n");
	fprintf (stream, "# num of grid cells:nx, ny, nz: %d, %d, %d\n", opts_.ny, opts_.nx, opts_.nz);
	fprintf (stream, "# range of the model space:\n");
	fprintf (stream, "#    x range (south-north):    [%.2f, %.2f]\n", opts_.yy[0], opts_.yy[1]);
	fprintf (stream, "#    y range (west-east):      [%.2f, %.2f]\n", opts_.xx[0], opts_.xx[1]);
	fprintf (stream, "#    z range (top-bottom):     [%.2f, %.2f]\n", - opts_.zz[0], - opts_.zz[1]);
	fprintf (stream, "# exf/mgz inc, dec:             %.2f, %.2f, %.2f, %.2f\n",
			opts_.exf_inc, opts_.exf_dec, opts_.mgz_inc, opts_.mgz_dec);
	fprintf (stream, "# tolerance and num of maxiter: %.2e, %zu\n", opts_.tolerance, opts_.maxiter);
	fprintf (stream, "# penalty parameter:mu:         %.2f\n", opts_.mu);
	fprintf (stream, "# penalty parameter:nu:         %.2f\n", opts_.nu);
	if (opts_.nu > 0.) {
		fprintf (stream, "# upper and lower bounds:       %.2f. %.2f\n", opts_.lower, opts_.upper);
	}
	fprintf (stream, "# \n");
	fprintf (stream, "# input anomaly data file name: %s\n", opts_.infn.c_str ());
	const L1L2Opts &lo = static_cast<const L1L2Opts &> (opts_);
	if (lo.log10_lambdas_.size () == 1) {
		fprintf (stream, "# alpha:                        %.4f\n", lo.alpha);
		fprintf (stream, "# log10(lambda):                %.4f\n", lo.log10_lambdas_[0]);
	} else {
		fprintf (stream, "# log10(lambda1):                %.4f\n", lo.log10_lambdas_[0]);
		fprintf (stream, "# log10(lambda2):                %.4f\n", lo.log10_lambdas_[1]);
	}
	fprintf (stream, "# verbose mode:                 %s\n", (opts_.verbose) ? "true" : "false");
	fprintf (stream, "#################################################################\n");
}

