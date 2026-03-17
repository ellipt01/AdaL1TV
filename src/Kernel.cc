#include <iostream>
#include <cstdio>
#include <stdexcept>
#include <mgcal.h>
#include <mmreal.h>
#include "Kernel.h"

/*** Public methods implementation ***/

// Destructor: release allocated memory and resources
Kernel::~Kernel ()
{
	if (grd_) grid_free (grd_);
}

// Set the coordinate ranges and create a grid for the model space
// nx, ny, nz : number of grid points in each dimension
// xx, yy, zz : coordinate arrays
// ll : optional stretching factor at the grid edges (default = 0)
void
Kernel::setRange (size_t nx, size_t ny, size_t nz, double *xx, double *yy, double *zz, const double ll)
{
	nx_ = nx;
	ny_ = ny;
	nz_ = nz;

	// Create grid
	if (grd_) grid_free (grd_);
	grd_ = grid_new (nx_, ny_, nz_, xx, yy, zz);
	n_ = grd_->n;

	// Apply stretching at the grid edges if requested
	if (ll > 0.) grid_stretch_at_edge (grd_, ll);
}

// Assign observed data to the Kernel
void
Kernel::setData (data_array *data)
{
	if (!data) throw std::invalid_argument ("Null pointer passed to set_data().");
	data_ = data;
	m_ = data_->n; // number of observations
}

// Write the model values to a stream
// The default format prints x, y, z, model value
void
Kernel::fwrite (FILE *stream, mm_real *model, const char *format = nullptr)
{
	if (!stream || !model)
		throw std::invalid_argument ("Invalid argument in fwrite().");

	if (format)
		fwrite_grid_with_data (stream, grd_, model->data, format);
	else
		fwrite_grid_with_data (stream, grd_, model->data, "%.4e\t%.4e\t%.4e\t%.8e");
}

