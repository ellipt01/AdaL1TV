#include <iostream>
#include <cstdio>
#include <stdexcept>
#include <mgcal.h>
#include <mmreal.h>
#include "Kernel.h"
#include "MagKernel.h"

/*** Public methods implementation ***/

// Destructor: free allocated vector3d objects
MagKernel::~MagKernel ()
{
	if (exf_) vector3d_free (exf_);
	if (mgz_) vector3d_free (mgz_);
	if (K_) mm_real_free (K_);
	if (func_) mgcal_func_free (func_);
}

// Constructor: external field and magnetization aligned
MagKernel::MagKernel (double inc, double dec)
{
	exf_ = vector3d_new_with_geodesic_poler (1., inc, dec); // external field vector
	mgz_ = vector3d_new_with_geodesic_poler (1., inc, dec); // magnetization vector
	func_ = mgcal_func_new (total_force_prism, nullptr);	   // kernel evaluation function
}

// Constructor: separate external field and magnetization directions
MagKernel::MagKernel (double exf_inc, double exf_dec, double mgz_inc, double mgz_dec)
{
	exf_ = vector3d_new_with_geodesic_poler (1., exf_inc, exf_dec);
	mgz_ = vector3d_new_with_geodesic_poler (1., mgz_inc, mgz_dec);
	func_ = mgcal_func_new (total_force_prism, nullptr);
}

// Set the unit vector representing the direction of the external geomagnetic field.
void
MagKernel::setExternalFieldDirection (double inc, double dec)
{
	if (exf_) vector3d_free (exf_);
	exf_ = vector3d_new_with_geodesic_poler (1., inc, dec);
}

// Set the unit vector representing the direction of the magnetization.
void
MagKernel::setMagnetizationDirection (double inc, double dec)
{
	if (mgz_) vector3d_free (mgz_);
	mgz_ = vector3d_new_with_geodesic_poler (1., inc, dec);
}

// Compute and return the magnetic kernel matrix
mm_real *
MagKernel::getKernel ()
{
	if (K_ == nullptr) compute_kernel_matrix_ ();
	return K_;
}

/*** private methods ***/

// Compute the magnetic kernel matrix internally
// This function is called by get() to initialize K_ if it is not yet computed
void
MagKernel::compute_kernel_matrix_ ()
{
	if (data_ == nullptr) throw std::invalid_argument ("Data array has not been set.");
	if (grd_ == nullptr)   throw std::invalid_argument ("Grid definition has not been set.");

	size_t	m = data_->n;
	size_t	n = grd_->n;

	// Allocate dense kernel matrix (m × n)
	if (K_) mm_real_free (K_);
	K_ = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m, n, m * n);

	// Fill kernel matrix using mgcal function
	kernel_matrix_set (K_->data, data_, grd_, mgz_, exf_, func_);
}

