/*
 * source.h
 *
 *  Created on: 2015/03/14
 *      Author: utsugi
 *
 *  Minimal version: set_position / set_dimension / set_magnetization removed.
 */

#ifndef SOURCE_H_
#define SOURCE_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct s_source_item	source_item;
typedef struct s_source			source;

struct s_source_item {
	vector3d	*mgz;	// magnetization
	vector3d	*pos;	// center of the magnetized body
	vector3d	*dim;	// dimension of source
	source_item	*next;
};

struct s_source {
	vector3d	*exf;	// external field
	source_item	*item;
	source_item	*begin;
	source_item	*end;
};

int		source_append_item (source *src);
source	*source_new (void);
void	source_free (source *src);
void	source_set_external (source *src, const double inc, const double dec);

#ifdef __cplusplus
}
#endif

#endif /* SOURCE_H_ */
