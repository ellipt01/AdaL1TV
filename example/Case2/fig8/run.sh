#!/usr/bin/env bash

root="../../../"
log_fn="log_case2_fig8"

if [ ! -e "$root"/bin/l1l2inv ] || [ ! -e "$root"/bin/l1tvinv ]; then
	make TIMING=1 -C "$root"
	make install -C "$root"
fi

schedule=(0.03125 0.0625 0.125 0.25 0.5 1)
lm_opt=`cat ../opt_lambda_L1L2.data`

rm -f log_case2_fig8

echo -n >| gammas_AdaL1TV.data
k=0
for gamma in "${schedule[@]}"; do

	echo "$k $gamma" >> gammas_AdaL1TV.data

	printf -v idx "%03d" "$k"

	if [ ! -e model_AdaL1TV_"$idx".data ]; then
		"$root"/bin/l1tvinv -f ../input_mag.in -l 0.25\
			-g ../beta_L1L2_opt.vec:"$lm_opt":0.1:0.1 -c "$gamma" -s ../settings.par -v 2>&1 | tee -a "$log_fn"

		mv -f model.data model_AdaL1TV_"$idx".data
		mv -f recovered.data recovered_AdaL1TV_"$idx".data
		mv -f beta_L1TV.vec beta_AdaL1TV_"$idx".vec
		mv -f regularization.vec regularization_AdaL1TV_"$idx".vec
	fi

	k=$((k+1))
done

