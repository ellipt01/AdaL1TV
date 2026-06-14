/*
 * calc.h
 *
 *  Created on: 2015/03/14
 *      Author: utsugi
 *
 *  Minimal version: prism (3D) only.
 */

#ifndef CALC_H_
#define CALC_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	MGCAL_X_COMPONENT,	// Hx
	MGCAL_Y_COMPONENT,	// Hy
	MGCAL_Z_COMPONENT,	// Hz
	MGCAL_TOTAL_FORCE	// f
} MgcalComponent;

vector3d	*prism (const vector3d *obs, const source *s);

double		total_force (const vector3d *exf, const vector3d *f);

double		x_component_prism (const vector3d *obs, const source *src, void *data);
double		y_component_prism (const vector3d *obs, const source *src, void *data);
double		z_component_prism (const vector3d *obs, const source *src, void *data);
double		total_force_prism (const vector3d *obs, const source *src, void *data);

#ifdef __cplusplus
}
#endif

#endif /* CALC_H_ */
