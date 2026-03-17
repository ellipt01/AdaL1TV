#ifndef _ADA_L1_TV_H_
#define _ADA_L1_TV_H_

// Adaptive L1 Total Variation (TV) inversion class
// Derived from the L1L2 inversion framework
class AdaL1TV : public L1L2 {

	/*** Inversion parameters and objects ***/
	ADMM_AdaL1TV	*admm_ = nullptr;   // ADMM solver for adaptive L1-TV

	// Guide model file for adaptive weighting
	bool			guide_model_file_specified_ = false;
	char			*guide_model_file_ = nullptr;  // File name of the guide model
	// scaling factor s is determined as followings:
	// s1 = median ( |beta_j| > c1 * sigma),
	// s2 = median ( |D*beta_j| > c2 * sigma),
	double		c1_ = 0.1;
	double		c2_ = 0.01;
	double		sigma_ = 1.;		// Scale parameter for adaptive weight
	double		gamma_ = 1.;          // Adaptive weighting parameter
	bool			use_unweighted_L1_guide_ = false;

public:
	AdaL1TV (const char *toolname);  // Constructor with tool name
	~AdaL1TV ();

	void			printUsage ();  // Print usage/help message

	void			initializeFromArgs (int argc, char **argv);  // Initialize from command-line arguments
	void			solve ();  // Run inversion

	void			exportResults (const char *ofn_model, const char *ofn_recovered);  // Export results to files
	void			exportResults () { exportResults ("model.data", "recovered.data"); }  // Default export

	mm_real		*getModel (bool remove_weight) { return admm_->getModel (remove_weight); }  // Get estimated model

protected:
	void			parse_command_line_args (int argc, char **argv);  // Parse command-line arguments
	void			parse_settings_file (FILE *fp);  // Parse settings file

	void			start_ADMM (mm_real *f, mm_real *K);  // Start ADMM solver

	void			export_settings (FILE *stream);  // Export settings to file
};


#endif // _ADA_L1_TV_H_

