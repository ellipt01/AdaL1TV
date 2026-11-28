#include <iostream>
#include <mgcal.h>
#include <mmreal.h>
#include "Kernel.h"
#include "MagKernel.h"
#include "PardisoSolver.h"
#include "DiffOp.h"
#include "ADMM.h"
#include "ADMM_AdaL1TV.h"
#include "L1L2.h"
#include "AdaL1TV.h"

#ifdef ENABLE_TIMING
#include <omp.h>
#endif

int
main (int argc, char **argv)
{
	CLI::App		app;
	AdaL1TVOpts	opts;
	opts.addTo (app);

	try {
		app.parse (argc, argv);
		opts.finalize ();
	} catch (const CLI::ParseError &e) {
		return app.exit (e);
	} catch (const std::exception &e) {
		std::fprintf (stderr, "Error: %s\n", e.what ());
		return EXIT_FAILURE;
	}


	AdaL1TV	inv (opts);

	try {
#ifdef ENABLE_TIMING
		double	start = omp_get_wtime ();
#endif
		inv.solve ();
#ifdef ENABLE_TIMING
		std::cout << "TIME: " << omp_get_wtime () - start << std::endl;
#endif
		inv.exportResults ();
		inv.exportObjective (stderr);
	} catch (const std::exception &e) {
		std::cerr << "ERROR: " << e.what () << std::endl;
		return EXIT_FAILURE;
	}

	// export weighted model in mm_real format
	mm_real	*model = inv.getModel (false);
	FILE		*fp = fopen ("beta_L1TV.vec", "w");
	if (fp) {
		mm_real_fwrite (fp, model, "%.12e");
		fclose (fp);
	}
	mm_real_free (model);
	
	return 0;
}
