#ifndef __DIFF_OP_H__
#define __DIFF_OP_H__

/*
	Class for computing discrete differential operators in 3D grids.

	This class generates sparse or dense derivative operators
	along x, y, z axes.
	It supports constructing the full differential operator D as either:
		- D0 = [I; Dx; Dy; Dz]  // including identity matrix I
		- D1 = [Dx; Dy; Dz]	  // without identity

	Usage Example:
		DiffOp diffOp;
	auto D0 = diffOp.calc(nx, ny, nz, true);  // D0: includes identity
 */

class DiffOp
{
public:
	DiffOp () = default;

	mm_real	*build (size_t nx, size_t ny, size_t nz, bool includeIdentity = false);

	mm_real	*createDiffX (size_t nx, size_t ny, size_t nz);
	mm_real	*createDiffY (size_t nx, size_t ny, size_t nz);
	mm_real	*createDiffZ (size_t nx, size_t ny, size_t nz);

	mm_real	*createLaplacianX (size_t nx, size_t ny, size_t nz);
	mm_real	*createLaplacianY (size_t nx, size_t ny, size_t nz);
	mm_real	*createLaplacianZ (size_t nx, size_t ny, size_t nz);

	mm_real	*createColWeightedLaplacianX (size_t nx, size_t ny, size_t nz, const double *weights);
	mm_real	*createColWeightedLaplacianY (size_t nx, size_t ny, size_t nz, const double *weights);
	mm_real	*createColWeightedLaplacianZ (size_t nx, size_t ny, size_t nz, const double *weights);

	mm_real	*createRowWeightedLaplacianX (size_t nx, size_t ny, size_t nz, const double *weights);
	mm_real	*createRowWeightedLaplacianY (size_t nx, size_t ny, size_t nz, const double *weights);
	mm_real	*createRowWeightedLaplacianZ (size_t nx, size_t ny, size_t nz, const double *weights);

	mm_real	*addLaplacians (const mm_real* Lx, const mm_real* Ly, const mm_real* Lz);

protected:
	// D = [ I; Dx; Dy; Dz ]
	mm_real	*buildD0_ (size_t nx, size_t ny, size_t nz);
	// D = [ Dx; Dy; Dz ]
	mm_real	*buildD1_ (size_t nx, size_t ny, size_t nz);
};

#endif

