#ifndef ADAL1TV_OPTS_H_
#define ADAL1TV_OPTS_H_

#include <string>
#include <vector>
#include <utility>
#include "CLI11.hpp"

class CommonOpts
{
public:
	std::string	infn;
	std::string	terrainfn;

	int			nx = 0;
	int			ny = 0;
	int			nz = 0;

	std::vector <double>	xx;
	std::vector <double>	yy;
	std::vector <double>	zz;

	double		exf_inc = 0.;
	double		exf_dec = 0.;
	double		mgz_inc = 0.;
	double		mgz_dec = 0.;

	double		mu = 1.;
	double		nu = -1.;
	double		upper = 0.;
	double		lower = 0.;

	double		tolerance = 1.e-3;
	size_t		maxiter = 1000;

	bool			verbose = false;	

protected:
	std::vector <double>	range_;
	std::vector <int>	numgrd_;
	std::vector <double>	incdec_;
	std::vector <double>	bounds_ = {-1., 0., 0.};
	std::pair <double, size_t>	iter_ = {1.e-3, 1000};

public:
	CommonOpts () = default;
	CommonOpts(const CommonOpts&) = delete;
    CommonOpts& operator=(const CommonOpts&) = delete;
    virtual ~CommonOpts() = default;
    
	virtual void		addTo (CLI::App &app);
	virtual void		finalize ();

};

class L1L2Opts : public CommonOpts {

public:
	double	lambda1;
	double	lambda2;
	double	alpha = -1.;
	std::vector <double>	log10_lambdas_;

public:
	L1L2Opts () = default;

	void		addTo (CLI::App &app) override;
	void		finalize () override;

};

class AdaL1TVOpts : public CommonOpts {

public:
	double		log10_lambda;
	double		lambda;
	std::string	guidefn;
	double		sigma = 1.;
	double		c1 = 0.1;
	double		c2 = 0.1;
	
	double		contrast = -1.;
public:
	AdaL1TVOpts () = default;

	void		addTo (CLI::App &app) override;
	void		finalize () override;

protected:
	std::vector <double>	sigmas_ = {1., 0.1, 0.1};
};


#endif // ADAL1TV_OPTS_H_
