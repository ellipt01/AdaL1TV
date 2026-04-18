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


def lcurve_L1L2():
	'''
	L-curve for L1-L2 inversion
	'''
	res = np.loadtxt('input_mag.in')
	f = res[:, 3]
	w = np.loadtxt('w.vec', skiprows=2)

	alpha = 0.9

	fn_list = sorted(glob.glob('model_L1L2_*.data'))

	nrm = np.empty(0)
	rss = np.empty(0)
	for fn in fn_list:

		idx = re.split('[_.]', fn)[2]
		try:
			float(idx)
		except ValueError:
			continue

		res = np.loadtxt('model_L1L2_{:s}.data'.format(idx))
		b = res[:, 3]
		beta = b * w

		nrmi = (1. - alpha) * np.linalg.norm(beta)**2 / 2. + alpha * sum(np.abs(beta))

		res = np.loadtxt('recovered_L1L2_{:s}.data'.format(idx))
		r = res[:, 3]
		rssi = np.linalg.norm(f - r) / 2.

		nrm = np.append(nrm, nrmi)
		rss = np.append(rss, rssi)
	return nrm, rss

def main():

	nrm, rss = lcurve_L1L2()
	res = np.loadtxt('lambdas_L1L2.data')
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

