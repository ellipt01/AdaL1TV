#include <iostream>
#include <cstring>
#include <limits>
#include <unistd.h>
#include <mgcal.h>
#include <mmreal.h>
#include "Kernel.h"
#include "MagKernel.h"
#include "ADMM.h"
#include "L1L2.h"

/****** Public methods ******/

// Constructor: store tool/application name
L1L2::L1L2 (const char* toolname)
{
	char buf[BUFFER_SIZE_];
	strncpy (buf, toolname, BUFFER_SIZE_);
	char* p = strrchr (buf, '/');
	if (p) strcpy (toolname_, ++p);
	else strcpy (toolname_, buf);
}

L1L2::~L1L2()
{
	if (infile_) delete [] infile_;
	if (terrain_fn_) delete [] terrain_fn_;
	if (xrange_) delete [] xrange_;
	if (yrange_) delete [] yrange_;
	if (zrange_) delete [] zrange_;
	if (magker_) delete magker_;
	if (admm_) delete admm_;
}

// Print usage message
void
L1L2::printUsage ()
{
	fprintf (stderr, "USAGE: %s\n", toolname_);
	fprintf (stderr, "       -f <input_file>\n");
	fprintf (stderr, "       -l <log10(lambda)> regularization parameter\n");
	fprintf (stderr, "[optional arguments]\n");
	fprintf (stderr, "       -a <alpha>         mixing ratio of L1 and L2 reguralization (default: 0.9)\n");
	fprintf (stderr, "       -t <terrain_file>  (default: terrain.in)\n");
	fprintf (stderr, "       -s <settings_file> (default: settings.par)\n");
	fprintf (stderr, "       -v                 verbose mode\n");
	fprintf (stderr, "       -h                 show this message\n");
}

// Initialize inversion object from command-line arguments and settings file
void
L1L2::initializeFromArgs (int argc, char** argv)
{
	// Parse inline parameters
	parse_command_line_args (argc, argv);
	validate_inline_params ();

	// Parse settings file
	FILE* fp = fopen (settings_, "r");
	if (!fp) throw std::runtime_error ("cannot open specified settings file" + std::string (settings_));
	parse_settings_file (fp);
	fclose (fp);
	validate_settings ();

	// Print settings to stderr
	export_settings (stderr);
}

// Run L1-L2 ADMM inversion
void
L1L2::solve ()
{
	FILE* fp = fopen (infile_, "r");         // Open data file
	if (!fp) throw std::runtime_error ("Cannot open specified input file.");
	data_ = fread_data_array (fp);
	fclose (fp);

	// Create RHS vector as mm_real view
	f_ = mm_real_view_array (MM_REAL_DENSE, MM_REAL_GENERAL, data_->n, 1, data_->n, data_->data);

	// Compute sensitivity (kernel) matrix
	magker_ = new MagKernel (exf_inc_, exf_dec_, mgz_inc_, mgz_dec_);
	magker_->setRange (nx_, ny_, nz_, xrange_, yrange_, zrange_, 1000.);
	if (terrain_fn_) set_terrain (terrain_fn_);

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

	mm_real* beta = admm_->getModel ();
	fwrite_grid_with_data (fp, magker_->getGrid (), beta->data, "%.4f\t%.4f\t%.4f\t%.6e");
	fclose (fp);
	mm_real_free (beta);

	mm_real* h = admm_->recover ();
	fp = fopen (ofn_recovered, "w");
	if (!fp) throw std::runtime_error ("Cannot open file " + std::string (ofn_recovered));

	fwrite_data_array_with_data (fp, data_, h->data, "%.4f\t%.4f\t%.4f\t%.6e");
	fclose (fp);
	mm_real_free (h);
}

/****** Protected / private methods ******/

// Set terrain data
void
L1L2::set_terrain (const char *fn)
{
	if (nx_ == 0 || ny_ == 0)
		throw std::runtime_error ("Number of grid not specified. Call initializeFromArgs first.");

	size_t	n = nx_ * ny_;
	double	*zsurf = new double [n];
	FILE		*fp = fopen (terrain_fn_, "r");
	if (!fp) throw std::runtime_error ("Cannot open terrain file.");
	char		buf[256];
	size_t	k = 0;
	while (fgets (buf, BUFSIZ, fp) != nullptr) {
		double	x, y, z;
		sscanf (buf, "%lf\t%lf\t%lf", &x, &y, &z);
		zsurf[k++] = z;
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

	admm_ = new ADMM (m, n, alpha_, log10_lambda_);

	if (verbose_) admm_->setVerbose ();

	admm_->setupLinearSystem (f, K, true);
	admm_->setL1L2Regularization (mu_);
	if (nu_ > 0.) admm_->setBoundConstraint (nu_, lower_, upper_);

	size_t niter = admm_->solve (maxiter_, tolerance_);
	fprintf (stderr, "residual[%04zu] = %.4e / %.4e\n", niter, admm_->getResiduals (), tolerance_);
}

// Parse command-line arguments
void
L1L2::parse_command_line_args (int argc, char** argv)
{
	int	solver_type_int = -1;
	char	opt;
	while ((opt = getopt (argc, argv, ":f:l:a:t:s:vh")) != -1)
	{
		switch (opt)
		{
			case 'f':
				infile_ = new char[BUFFER_SIZE_];
				strncpy (infile_, optarg, BUFFER_SIZE_);
				break;
			case 'a':
				alpha_ = atof (optarg);
				break;
			case 'l':
				log10_lambda_ = atof (optarg);
				lambda_specified_ = true;
				break;
			case 't':
				terrain_fn_ = new char[BUFFER_SIZE_];
				strncpy (terrain_fn_, optarg, BUFFER_SIZE_);
				break;
			case 's':
				strncpy (settings_, optarg, BUFFER_SIZE_);
				break;
			case 'v':
				verbose_ = true;
				break;
			case 'h':
				printUsage ();
				exit (1);
			default:
				throw std::runtime_error ("Unknown option.");
		}
	}

}

// Parse parameter specifications from settings file
void
L1L2::parse_settings_file (FILE* fp)
{
	char buf[BUFSIZ];
	while (fgets (buf, BUFSIZ, fp) != nullptr)
	{
		if (buf[0] == '#') continue;

		const char* p = skip_blanks (buf);
		if (strlen (p) <= 1) continue;

		const char* ptr = strchr (p, ':');
		if (!ptr) continue;
		ptr = skip_blanks (ptr + 1);
		if (strlen (ptr) <= 1) continue;

		switch (p[0])
		{
			// In the program, the x-, y-, and z-axes are defined to
			// eastward, northward, and upward,
			// and are transformed upon reading the settings file.
			case '1':
				sscanf (ptr, "%zu,%zu,%zu", &ny_, &nx_, &nz_);
				break;
			case '2':
				xrange_ = new double[2];
				yrange_ = new double[2];
				zrange_ = new double[2];
				sscanf (ptr, "%lf,%lf,%lf,%lf,%lf,%lf",
					  &yrange_[0], &yrange_[1], &xrange_[0], &xrange_[1], &zrange_[0], &zrange_[1]);
				zrange[0] *= -1.;
				zrange[1] *= -1.;
				break;
			case '3':
				sscanf (ptr, "%lf,%lf,%lf,%lf", &exf_inc_, &exf_dec_, &mgz_inc_, &mgz_dec_);
				break;
			case '4':
				mu_ = atof (ptr);
				break;
			case '5':
				sscanf (ptr, "%lf,%lf,%lf", &nu_, &lower_, &upper_);
				break;
			case '6':
				sscanf (ptr, "%lf,%zu", &tolerance_, &maxiter_);
				tol_maxiter_specified_ = true;
				break;
			default:
				throw std::runtime_error ("Unexpected identifier in settings file.");
		}
	}
}

// Validate inline parameters
void
L1L2::validate_inline_params ()
{
	if (!infile_) throw std::runtime_error ("Input file name is not specified.");
	if (!lambda_specified_) throw std::runtime_error ("log10(lambda) is not specified.");
}

// Validate settings file parameters
void
L1L2::validate_settings ()
{
	if (nx_ <= 0 || ny_ <= 0 || nz_ <= 0)
		throw std::runtime_error ("Invalid number of grid cells specified.");

	if (fabs (xrange_[1] - xrange_[0]) < std::numeric_limits<double>::epsilon () ||
	    fabs (yrange_[1] - yrange_[0]) < std::numeric_limits<double>::epsilon () ||
	    fabs (zrange_[1] - zrange_[0]) < std::numeric_limits<double>::epsilon ())
		throw std::runtime_error ("Invalid x/y/z ranges specified.");

	if (mu_ <= 0.) throw std::runtime_error ("mu must be > 0.");
	if (nu_ < 0.) nu_ = 0.;
	if (nu_ > 0. && upper_ <= lower_)
		throw std::runtime_error ("Lower bound must be less than upper bound.");

	if (!tol_maxiter_specified_) throw std::runtime_error ("Tolerance and max iterations are not specified.");
}

// Export settings to file stream
void
L1L2::export_settings (FILE* stream)
{
	fprintf (stream, "#################################################################\n");
	fprintf (stream, "# *** penalty type: L1L2 ***\n");
	fprintf (stream, "# \n");
	fprintf (stream, "# num of grid cells:nx, ny, nz: %zu, %zu, %zu\n", ny_, nx_, nz_);
	fprintf (stream, "# range of the model space:\n");
	fprintf (stream, "#    x range (south-north):    [%.2f, %.2f]\n", yrange_[0], yrange_[1]);
	fprintf (stream, "#    y range (west-east):      [%.2f, %.2f]\n", xrange_[0], xrange_[1]);
	fprintf (stream, "#    z range (up-down):        [%.2f, %.2f]\n", -zrange_[0], -zrange_[1]);
	fprintf (stream, "# exf/mgz inc, dec:             %.2f, %.2f, %.2f, %.2f\n",
		   exf_inc_, exf_dec_, mgz_inc_, mgz_dec_);
	fprintf (stream, "# tolerance and num of maxiter: %.2e, %zu\n", tolerance_, maxiter_);
	fprintf (stream, "# penalty parameter:mu:         %.2f\n", mu_);
	fprintf (stream, "# penalty parameter:nu:         %.2f\n", nu_);
	if (nu_ > 0.) {
		fprintf (stream, "# upper and lower bounds:       %.2f. %.2f\n", lower_, upper_);
	}
	fprintf (stream, "# \n");
	fprintf (stream, "# input anomaly data file name: %s\n", infile_);
	fprintf (stream, "# settings file name:           %s\n", settings_);
	fprintf (stream, "# alpha:                        %.4f\n", alpha_);
	fprintf (stream, "# log10(lambda):                %.4f\n", log10_lambda_);
	fprintf (stream, "# verbose mode:                 %s\n", (verbose_) ? "true" : "false");
	fprintf (stream, "#################################################################\n");
}

// Skip leading blanks in a C-style string (private utility)
const char *
L1L2::skip_blanks (const char* str) const
{
	if (!str) return nullptr;
	while (*str == ' ' || *str == '\t') ++str;
	return str;
}

/****** Private methods ******/
void
L1L2::init_ ()
{
	strcpy (settings_, "settings.par");
}

