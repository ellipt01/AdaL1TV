#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include "CLI11.hpp"
#include "AdaL1TVOpts.h"

// Register the options common to all inversion programs.
void
CommonOpts::addTo (CLI::App &app)
{

	app.add_option ("-f,--infile", infn, "Input magnetic anomaly data file")
		->required ();
	app.add_option ("-l,--lambda", log10_lambda, "Regularization parameter: log10(lambda)")
		->required ();
	app.add_option ("-n,--numgrd", numgrd_, "Number of grid cells: nx/ny/nz")
		->delimiter ('/')
		->expected (3)
		->required ();
	app.add_option ("-r,--range", range_, "Model space range: west/east/south/north/top/bottom (km)")
		->delimiter ('/')
		->expected (6)
		->required ();
	app.add_option ("-d,--direction", incdec_,
		"Inclination and declination of external field and magnetization:\n"
		"exf_inc/exf_dec/mgz_inc/mgz_dec")
		->delimiter ('/')
		->expected (4)
		->required ();
	app.add_option ("-p,--penalty", mu, "Penalty parameter: mu")
		->capture_default_str ();

	// Show the default values of the composite options in --help
	std::ostringstream	os;
	os << nu << "/" << lower << "/" << upper;
	app.add_option ("-b,--bounds", bounds_, "Penalty parameter nu, and lower and upper bounds: nu/lower/upper")
		->delimiter ('/')
		->expected (3)
		->default_str (os.str ());
	os.str ("");
	os.clear ();
	os << tolerance << "/" << maxiter;
	app.add_option ("-i,--iteration", iter_, "Tolerance and maximum number of iterations: tol/maxiter")
		->delimiter ('/')
		->default_str (os.str ());
	app.add_option ("-t,--terrain", terrainfn, "Terrain file");

	app.add_flag ("-v,--verbose", verbose, "Enable verbose output")
		->default_str ("false");
}

// Validate the parsed values and convert them into the final member variables.
// Must be called after CLI::App::parse().
void
CommonOpts::finalize ()
{
	// Grid size
	if (numgrd_[0] <= 0 || numgrd_[1] <= 0 || numgrd_[2] <= 0)
		throw std::runtime_error ("Number of grid cells (nx, ny, nz) must be > 0.");
	nx = numgrd_[0];
	ny = numgrd_[1];
	nz = numgrd_[2];

	// Model space range
	xx = {range_[0], range_[1]};
	yy = {range_[2], range_[3]};
	zz = {range_[4], range_[5]};
	if (xx[0] >= xx[1] || yy[0] >= yy[1] || zz[0] >= zz[1])
		throw std::runtime_error ("Invalid range: must satisfy west < east, south < north, and top < bottom.");

	// Directions of the external field and the magnetization
	exf_inc = incdec_[0];
	exf_dec = incdec_[1];
	mgz_inc = incdec_[2];
	mgz_dec = incdec_[3];

	// Bound constraint (active only when nu > 0)
	nu = bounds_[0];
	lower = bounds_[1];
	upper = bounds_[2];
	if (nu > 0. && upper < lower) throw std::runtime_error ("Invalid bounds: lower must not exceed upper.");

	// Convergence criteria
	tolerance = iter_.first;
	maxiter = iter_.second;
}

// Register the common options plus the L1L2-specific option (alpha).
void
L1L2Opts::addTo (CLI::App &app)
{
	CommonOpts::addTo (app);

	app.add_option ("-a,--alpha", alpha, "Mixing ratio of L1 and L2 penalties")
		->capture_default_str ();
}

// Validate and finalize the common values (no L1L2-specific processing is needed yet).
void
L1L2Opts::finalize ()
{
	CommonOpts::finalize ();
}

// Register the common options plus the AdaL1TV-specific options (guide model, sigma, contrast).
void
AdaL1TVOpts::addTo (CLI::App &app)
{
	CommonOpts::addTo (app);

	app.add_option ("-g,--guide", guidefn,
		"Guide model file name.\n"
		"If not specified, the -s option is ignored and plain L1TV is performed.");
	app.add_option ("-s,--sigma", sigmas_,
		"Thresholds for non-zero elements (c1*sigma, c2*sigma): sigma/c1/c2.\n"
		"Used only when -g is specified.")
		->delimiter ('/')
		->expected (3)
		->capture_default_str ();
	app.add_option ("-c,--contrast", contrast, "Contrast ratio between zero and non-zero grid cells\n"
		"If <= 0 is specified, weight exponent gamma is set to 1.")
		->capture_default_str ();
}

// Validate the common values and unpack sigma, c1, and c2.
void
AdaL1TVOpts::finalize ()
{
	CommonOpts::finalize ();
	sigma = sigmas_[0];
	c1 = sigmas_[1];
	c2 = sigmas_[2];
}
