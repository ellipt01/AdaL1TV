#ifndef ADA_L1_TV_H_
#define ADA_L1_TV_H_

#include <cstdio>
#include "Kernel.h"
#include "MagKernel.h"
#include "ADMM.h"
#include "ADMM_AdaL1TV.h"
#include "AdaL1TVOpts.h"
#include "L1L2.h"

// Adaptive L1 Total Variation (TV) inversion class
// Derived from the L1L2 inversion framework
class AdaL1TV : public L1L2 {

	const AdaL1TVOpts	&aopts_;   // 型付きの設定参照(基底の opts_ とは別名)

public:
	explicit AdaL1TV (const AdaL1TVOpts &opts) : L1L2 (opts), aopts_ (opts) {}
	~AdaL1TV () override = default;   // admm_ は基底が delete

	AdaL1TV (const AdaL1TV &) = delete;
	AdaL1TV &operator= (const AdaL1TV &) = delete;

	// regularization.vec も出力する
	void		exportResults (const char *ofn_model, const char *ofn_recovered) override;
	using L1L2::exportResults;        // 引数なし版を隠さない

	// Export the evaluation of the objective function at the current model
	void		exportObjective (FILE *stream);

protected:
	void		start_ADMM (mm_real *f, mm_real *K) override;
	void		export_settings (FILE *stream) override;
};

#endif // ADA_L1_TV_H_
