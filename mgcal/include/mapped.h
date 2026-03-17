/*
 * mapped.h
 *
 *  Created on: 2020/07/04
 *      Author: utsugi
 */

#ifndef MAPPED_H_
#define MAPPED_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct s_mapped  mapped;

struct s_mapped
{
	char	fn[80];
	int		fd;
	int		m;
	int		n;
	long	nnz;
	long	size;
	double	*data;
};

mapped	*mapped_new (int id, int nobs, int nsrc, const char *dir);
void	mapped_free (mapped *map);
mapped	*mapped_fread (const char *fn, size_t m, size_t n, size_t nnz);

mapped	*kernel_matrix_mapped (int id, const data_array *array, const grid *g, const vector3d *mgz, const vector3d *exf, const mgcal_func *f,
	const char *dir);

#ifdef __cplusplus
}
#endif

#endif /* MAPPED_H_ */
