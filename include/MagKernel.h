#ifndef _MAG_KERNEL_H_
#define _MAG_KERNEL_H_

/*** Magnetic kernel subclass of Kernel ***/
class MagKernel : public Kernel
{
	// Unit vectors parallel to the external field and magnetization
	vector3d	*exf_ = nullptr; // External magnetic field direction
	vector3d	*mgz_ = nullptr; // Magnetization vector direction

	// Kernel matrix representing the forward operator
	mm_real	*K_ = nullptr;

	// mgcal_func handle for kernel computation
	mgcal_func	*func_ = nullptr;

public:
	MagKernel () { }

	// Constructor with external field inclination and declination
	// exf_inc, exf_dec: inclination and declination of the external field
	MagKernel (double inc, double dec);

	// Constructor with separate external field and magnetization directions
	// exf_inc, exf_dec: external field direction (inclination and declination)
	// mgz_inc, mgz_dec: magnetization direction
	MagKernel (double exf_inc, double exf_dec, double mgz_inc, double mgz_dec);

	~MagKernel ();

	// Set the unit vector representing the direction of the external geomagnetic field.
	// inc, dec: inclination and declination in degrees
	void setExternalFieldDirection (double inc, double dec);

	// Set the unit vector representing the direction of the magnetization.
	// inc, dec: inclination and declination in degrees
	void setMagnetizationDirection (double inc, double dec);

	// Compute and return the magnetic kernel matrix
	// Returns: pointer to mm_real representing the kernel
	mm_real *getKernel ();

private:
	// Internal function to evaluate the kernel matrix
	// Called when the kernel matrix has not been computed yet
	void compute_kernel_matrix_ ();
};

#endif // _MAG_KERNEL_H_

