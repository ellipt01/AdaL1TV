#ifndef _L1_L2_H_
#define _L1_L2_H_

#ifndef BUFFER_SIZE_
#define BUFFER_SIZE_ 512
#endif

// L1L2: Class to manage L1-L2 regularized inversion
class L1L2 {

protected:
	char		toolname_[BUFFER_SIZE_]; // Name of the tool/application

	/*** Model space parameters ***/
	size_t	nx_ = 0; // Number of grid points in x
	size_t	ny_ = 0; // Number of grid points in y
	size_t	nz_ = 0; // Number of grid points in z

	double	*xrange_ = nullptr; // X-axis coordinates
	double	*yrange_ = nullptr; // Y-axis coordinates
	double	*zrange_ = nullptr; // Z-axis coordinates

	/*** Magnetic properties and objects ***/
	MagKernel	*magker_ = nullptr; // Magnetic kernel object

	char		*terrain_fn_ = nullptr; // Terrain file name

	// Geomagnetic external field inclination and declination
	double	exf_inc_ = 0.;
	double	exf_dec_ = 0.;
	// Magnetization inclination and declination
	double	mgz_inc_ = 0.;
	double	mgz_dec_ = 0.;

	data_array	*data_ = nullptr; // Observed data array

	mm_real	*f_ = nullptr; // RHS vector
	mm_real	*K_ = nullptr; // Kernel matrix

	/*** Inversion parameters and objects ***/
	ADMM		*admm_ = nullptr; // ADMM solver object

	char		*infile_ = nullptr;				  // Input file
	char		settings_[BUFFER_SIZE_];   // Settings string

	double	alpha_ = 0.9; // Mixing ratio

	bool		lambda_specified_ = false;
	double	log10_lambda_ = 0.; // Regularization parameter (log10)

	bool		tol_maxiter_specified_ = false;
	double	tolerance_ = 1.e-3; // Convergence tolerance
	size_t	maxiter_ = 1000;	// Maximum number of iterations

	double	mu_ = 0.;	// TV regularization parameter
	double	nu_ = 0.;	// Bound constraint parameter
	double	lower_ = 0.; // Lower bound
	double	upper_ = 0.; // Upper bound

	bool		verbose_ = false; // Verbose output

public:
	// Constructors
	L1L2 () { init_ (); }
	L1L2 (const char *toolname);
	
	~L1L2 ();

	L1L2 (const L1L2&) = delete;
	L1L2& operator = (const L1L2&) = delete;

	// Print usage message
	void		printUsage ();

	// Initialize the inversion from command-line arguments
	void		initializeFromArgs (int argc, char **argv);

	// Run the inversion process
	void		solve ();

	// Export model and recovered results
	void		exportResults (const char *ofn_model, const char *ofn_recovered);
	void		exportResults () { exportResults ("model.data", "recovered.data"); }

	// Retrieve model after inversion (optionally remove weights)
	mm_real	*getModel (bool remove_weight = true) { return admm_->getModel (remove_weight); }

protected:
	// Internal argument parsing
	void		parse_command_line_args (int argc, char **argv);
	void		parse_settings_file (FILE *fp);

	// Read terrain file
	void		set_terrain (const char *fn);

	// Start ADMM solver with given data and kernel
	void		start_ADMM (mm_real *f, mm_real *K);

	// Check inline parameters and settings
	void		validate_inline_params ();
	void		validate_settings ();

	// Export settings to file stream
	void		export_settings (FILE *stream);

	// Skip blank characters in a string
	const char*	skip_blanks (const char* str) const;

private:
	void		init_ (); // initializer
};

#endif // _L1_L2_H_

