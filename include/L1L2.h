#ifndef L1_L2_H_
#define L1_L2_H_

#include <cstdio>
#include "Kernel.h"
#include "MagKernel.h"
#include "ADMM.h"
#include "AdaL1TVOpts.h"

// L1L2: Class to manage L1-L2 regularized inversion
class L1L2 {

protected:
	const CommonOpts	&opts_;			// L1L2Opts -> CommonOpts

	data_array	*data_ = nullptr;		// Observed data array
	MagKernel	*magker_ = nullptr;

	mm_real		*f_ = nullptr;			// RHS vector
	mm_real		*K_ = nullptr;			// Kernel matrix

	ADMM			*admm_ = nullptr;		// ADMM solver object

public:
	explicit L1L2 (const CommonOpts &opts) : opts_ (opts) {}
	virtual ~L1L2 ();					// virtual

	L1L2 (const L1L2 &) = delete;
	L1L2 &operator= (const L1L2 &) = delete;

	virtual void	solve ();

	virtual void	exportResults (const char *ofn_model, const char *ofn_recovered);
	void			exportResults () { exportResults ("model.data", "recovered.data"); }

	mm_real		*getModel (bool remove_weight = true) { return admm_->getModel (remove_weight); }

protected:
	void			set_terrain (const char *fn);
	virtual void	start_ADMM (mm_real *f, mm_real *K);
	virtual void	export_settings (FILE *stream);
};

#endif // L1_L2_H_
