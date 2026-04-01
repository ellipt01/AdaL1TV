#include <iostream>
#include <cstring>
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

static size_t
count_num_delim (char *str, const char delim)
{
	size_t	count = 0;
	char		*p = str;
	while (p) {
		p = strchr (p, delim);
		if (p == nullptr) break;
		count++;
		p++;
	}
	return count;

}

/****** public ******/
AdaL1TV::AdaL1TV (const char* toolname)
{
	// Extract tool name from path
	char buf[BUFFER_SIZE_];
	strncpy (buf, toolname, BUFFER_SIZE_);
	char* p = strrchr (buf, '/');
	if (p) strcpy (toolname_, ++p);
	else strncpy (toolname_, buf, BUFFER_SIZE_);
}

AdaL1TV::~AdaL1TV ()
{
	if (admm_) delete admm_;
}

void AdaL1TV::printUsage ()
{
	fprintf (stderr, "USAGE: %s\n", toolname_);
	fprintf (stderr, "       -f <input_file>\n");
	fprintf (stderr, "       -l <log10(lambda)>   regularization parameter\n");
	fprintf (stderr, "[optional arguments]\n");
	fprintf(stderr, "        -g <guide_model_file>:<sigma>:<c1>:<c2>\n");
	fprintf(stderr, "            guide_model_file : guide model in mm_real format\n");
	fprintf(stderr, "            sigma            : regularization parameter (lambda) used to derive guide model\n");
	fprintf(stderr, "            c1, c2           : coefficients for defining the active set\n");
	fprintf (stderr, "       -c <weight_exponent> (default: 1)\n");
	fprintf (stderr, "       -t <terrain_file>    (default: terrain.in)\n");
	fprintf (stderr, "       -s <settings_file>   (default: settings.par)\n");
	fprintf (stderr, "       -v                   verbose mode\n");
	fprintf (stderr, "       -h                   show this message>\n");
}

// Initialize AdaL1TV object from command-line arguments
void AdaL1TV::initializeFromArgs (int argc, char** argv)
{
	// Parse inline parameters
	parse_command_line_args (argc, argv);
	validate_inline_params ();

	// Parse settings file
	FILE* fp = fopen (settings_, "r");
	if (!fp) throw std::runtime_error ("cannot open specified settings file " + std::string (settings_));
	parse_settings_file (fp);
	fclose (fp);
	validate_settings ();

	// Print settings to stderr
	export_settings (stderr);
}

// Perform ADMM iteration
void AdaL1TV::solve ()
{
	// Read input data file
	FILE* fp = fopen (infile_, "r");
	if (!fp) throw std::runtime_error ("cannot open specified input file " + std::string (infile_));
	data_ = fread_data_array (fp);
	fclose (fp);

	f_ = mm_real_view_array (MM_REAL_DENSE, MM_REAL_GENERAL, data_->n, 1, data_->n, data_->data);

	// Compute sensitivity matrix
	magker_ = new MagKernel (exf_inc_, exf_dec_, mgz_inc_, mgz_dec_);
	magker_->setRange (nx_, ny_, nz_, xrange_, yrange_, zrange_, 1000.);
	if (terrain_fn_) set_terrain (terrain_fn_);
	magker_->setData (data_);
	K_ = magker_->getKernel ();

	// Start ADMM iteration
	start_ADMM (f_, K_);
}

// Export recovered anomaly and model
void AdaL1TV::exportResults (const char* ofn_model, const char* ofn_recovered)
{
	FILE* fp;

	// Export model
	fp = fopen (ofn_model, "w");
	if (!fp)
		throw std::runtime_error ("cannot open file " + std::string (ofn_model) + " to export derived model.");

	mm_real	*beta = admm_->getModel ();
	fwrite_grid_with_data (fp, magker_->getGrid (), beta->data, "%.4f\t%.4f\t%.4f\t%.6e");
	fclose (fp);
	mm_real_free (beta);

	// Export recovered anomaly
	mm_real* h = admm_->recover ();
	fp = fopen (ofn_recovered, "w");
	if (!fp)
		throw std::runtime_error ("cannot open file " + std::string (ofn_recovered) + " to export recovered anomaly.");

	fwrite_data_array_with_data (fp, data_, h->data, "%.4f\t%.4f\t%.4f\t%.6e");
	fclose (fp);
	mm_real_free (h);

	mm_real	*d = admm_->getRegularizationVector ();
	fp = fopen ("regularization.vec", "w");
	if (fp) {
		mm_real_fwrite (fp, d, "%.6e");
		fclose (fp);
	}
}

// Start ADMM iteration until convergence
void AdaL1TV::start_ADMM (mm_real* f, mm_real* K)
{
	size_t m = f->m;
	size_t n = K->n;

	admm_ = new ADMM_AdaL1TV (m, n, log10_lambda_);

	if (verbose_) admm_->setVerbose ();

	admm_->setupLinearSystem (f, K, true);
	admm_->setTVRegularization (mu_, nx_, ny_, nz_);
	if (nu_ > 0.) admm_->setBoundConstraint (nu_, lower_, upper_);
	if (guide_model_file_specified_) {
		FILE* fp = fopen (guide_model_file_, "r");
		if (!fp)
			throw std::runtime_error ("cannot open specified guide model file " + std::string (guide_model_file_));
		mm_real* guide_model = mm_real_fread (fp);
		fclose (fp);
		admm_->setAdaptiveWeighting (sigma_, gamma_, guide_model, c1_, c2_);
		mm_real_free (guide_model);
	}
	size_t niter = admm_->solve (maxiter_, tolerance_);
	fprintf (stderr, "residual[%04zu] = %.4e / %.4e\n", niter, admm_->getResiduals (), tolerance_);
}

/****** protected ******/
// Parse inline parameters
void AdaL1TV::parse_command_line_args (int argc, char** argv)
{
	int	solver_type_int = -1;
	char	opt;
	while ((opt = getopt (argc, argv, ":f:l:g:c:t:s:vh")) != -1) {
		switch (opt) {
			case 'f':
				infile_ = new char[BUFFER_SIZE_];
				strncpy (infile_, optarg, BUFFER_SIZE_);
				break;
			case 'l':
				log10_lambda_ = atof (optarg);
				lambda_specified_ = true;
				break;
			case 'g':
				if (count_num_delim (optarg, ':') != 3)
					throw std::runtime_error ("-g option has to specify guide_model_fn(char):sigma(double):c1:c2");
				guide_model_file_ = new char[256];
				sscanf (optarg, "%256[^:]:%lf:%lf:%lf", guide_model_file_, &sigma_, &c1_, &c2_);
				guide_model_file_specified_ = true;
				break;
			case 'c':
				gamma_ = (double) atof (optarg);
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
				throw std::runtime_error ("unknown option.");
				break;
		}
	}
}

// Parse parameter specifications from settings file
void AdaL1TV::parse_settings_file (FILE* fp)
{
	char buf[BUFSIZ];

	while (fgets (buf, BUFSIZ, fp) != nullptr) {
		if (buf[0] == '#') continue;
		
		const char* p = buf;
		p = skip_blanks (p);
		if (strlen (p) <= 1) continue;

		const char* ptr = strchr (p, ':');
		if (ptr == nullptr) continue;
		else ptr++;
		ptr = skip_blanks (ptr);
		if (strlen (ptr) <= 1) continue;

		switch (p[0]) {
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
				zrange_[0] *= -1.;
				zrange_[1] *= -1.;
				break;
			case '3':
				sscanf (ptr, "%lf,%lf,%lf,%lf", &exf_inc_, &exf_dec_, &mgz_inc_, &mgz_dec_);
				break;
			case '4':
				mu_ = (double) atof (ptr);
				break;
			case '5':
				sscanf (ptr, "%lf,%lf,%lf", &nu_, &lower_, &upper_);
				break;
			case '6':
				sscanf (ptr, "%lf,%zu", &tolerance_, &maxiter_);
				tol_maxiter_specified_ = true;
				break;
			default:
				throw std::runtime_error ("unexpected identifier.");
		}
	}
}

// Export settings to stream
void AdaL1TV::export_settings (FILE* stream)
{
	fprintf (stream, "#################################################################\n");
	fprintf (stream, "# *** penalty type: %s ***\n", (guide_model_file_specified_) ? "AdaL1TV" : "L1TV");
	fprintf (stream, "# \n");
	fprintf (stream, "# num of grid cells:nx, ny, nz: %zu, %zu, %zu\n", ny_, nx_, nz_);
	fprintf (stream, "# range of the model space:\n");
	fprintf (stream, "#    x range (south-north):     [%.2f, %.2f]\n", yrange_[0], yrange_[1]);
	fprintf (stream, "#    y range (west-east):       [%.2f, %.2f]\n", xrange_[0], xrange_[1]);
	fprintf (stream, "#    z range (up-down):         [%.2f, %.2f]\n", -zrange_[0], -zrange_[1]);
	fprintf (stream, "# exf. and mag. inc, dec.:      %.2f, %.2f, %.2f, %.2f\n",
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
	fprintf (stream, "# log10(lambda):                %.4f\n", log10_lambda_);
	fprintf (stream, "# apply adaptive weighting:     %s\n", (guide_model_file_specified_) ? "true" : "false");
	if (guide_model_file_specified_) {
		fprintf (stream, "# guide model file name:        %s\n", guide_model_file_);
		fprintf (stream, "# gamma:                        %.3f\n", gamma_);
	}
	fprintf (stream, "# verbose mode:                 %s\n", (verbose_) ? "true" : "false");
	fprintf (stream, "#################################################################\n");
}

