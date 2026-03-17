#ifndef _KERNEL_H_
#define _KERNEL_H_

/*** Superclass for kernel computations ***/
class Kernel {

protected:

	// Number of observed data points
	size_t	m_ = 0;

	// Number of model elements (grid points)
	size_t	n_ = 0;

	// Grid dimensions in each direction
	size_t	nx_ = 0;
	size_t	ny_ = 0;
	size_t	nz_ = 0;

	// Grid object representing the 3D model space
	grid		*grd_ = nullptr;

	// Observed data array
	// data_ is not owned by Kernel (external ownership)
	data_array	*data_ = nullptr;

public:
	Kernel () { }
	virtual ~Kernel ();

	Kernel (const Kernel&) = delete;
	Kernel& operator = (const Kernel&) = delete;

	// Set the coordinate ranges and create a grid
	// nx, ny, nz: number of grid points in x, y, z
	// xx, yy, zz: coordinate arrays
	// boundary_extension: additional margin added to the model space
	//                     to extend the edges of the computational domain.
	void		setRange (size_t nx, size_t ny, size_t nz,
				     double *xx, double *yy, double *zz,
				     const double boundary_extension = 0.);

	// Set the surface topography of the model
	void		setSurface (double *zsurf) { grid_set_surface (grd_, zsurf); }

	// Assign observed data array to the Kernel
	void		setData (data_array *array);

	// Access the model grid object
	grid		*getGrid () { return grd_; }

	// Pure virtual function to compute and return the kernel matrix
	virtual mm_real	*getKernel () = 0;

	// Write the model to a file or stream
	void		fwrite (FILE *stream, mm_real *model, const char *format);
};

#endif // _KERNEL_H_

