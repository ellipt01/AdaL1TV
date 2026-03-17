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
	AdaL1TV	inv (argv[0]);

	try {
		inv.initializeFromArgs (argc, argv);
#ifdef ENABLE_TIMING
		double	start = omp_get_wtime ();
#endif
		inv.solve ();
#ifdef ENABLE_TIMING
		std::cout << "TIME: " << omp_get_wtime () - start << std::endl;
#endif
		inv.exportResults ();
	} catch (const std::exception &e) {
		std::cerr << "ERROR: " << e.what () << std::endl;
		inv.printUsage ();
		exit (1);
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
