#!/usr/bin/env python3
# coding: utf-8

import glob
import re
import numpy as np

from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt

def lcurve_curvature(lambda_vals, misfit, regularization):
	"""
	Compute curvature of L-curve.

	Parameters
	----------
	lambda_vals : array
		regularization parameters
	misfit : array
		||Aβ - d||_2
	regularization : array
		||Lβ||_2

	Returns
	-------
	curvature : array
		curvature of the L-curve
	"""

	# assumes inputs are already in log scale
	x = misfit
	y = regularization

	t = lambda_vals

	# spline interpolation
	sx = CubicSpline(t, x)
	sy = CubicSpline(t, y)

	# derivatives
	x1 = sx(t, 1)
	x2 = sx(t, 2)

	y1 = sy(t, 1)
	y2 = sy(t, 2)

	# curvature
	eps = 1.e-12
	curvature = np.abs(x1 * y2 - y1 * x2) / (x1**2 + y1**2 + eps)**1.5

	return curvature


def lcurve_L1TV():
	'''
	L-curve for L1TV inversion
	'''
	# L-curve for L1TV inversion
	res = np.loadtxt('input_mag.in')
	f = res[:, 3]

	fn_list = sorted(glob.glob('model_L1TV_*.data'))

	nrm = np.empty(0)
	rss = np.empty(0)
	for fn in fn_list:

		idx = re.split('[_.]', fn)[2]
		try:
			lam = float(idx)
		except ValueError:
			continue

		d = np.loadtxt('regularization_L1TV_{:s}.vec'.format(idx), skiprows=2)
		nrmi = np.abs(d).sum()

		res = np.loadtxt('recovered_L1TV_{:s}.data'.format(idx))
		r = res[:, 3]
		rssi = np.linalg.norm(f - r) / 2.

		nrm = np.append(nrm, nrmi)
		rss = np.append(rss, rssi)
	return nrm, rss

def main():

	nrm, rss = lcurve_L1TV()
	res = np.loadtxt('lambdas_L1TV.data')
	lambda_vals = res[:, 1]

	eps = 1.e-20
	lrss = np.log10(rss + eps)
	lnrm = np.log10(nrm + eps)

	c = lcurve_curvature(lambda_vals, lrss, lnrm)
	opt = np.argmax(c)

	return opt

if __name__ == '__main__':
	opt = main()
	print(opt)

