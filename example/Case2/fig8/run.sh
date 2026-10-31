#!/usr/bin/env bash

root="../../../"
log_fn="log_case2_fig8"

if [ ! -e "$root"/bin/l1l2inv ] || [ ! -e "$root"/bin/l1tvinv ]; then
	make TIMING=1 -C "$root"
	make install -C "$root"
fi

lm_opt=`cat ../opt_lambda_AdaL1TV.data | gawk '{print log($1)/log(10)}'`
echo $lm_opt
#schedule=(0.03125 0.0625 0.125 0.25 0.5 1)
#               1/7         1/6      1/5  1/4       1/3    1/2
#schedule=(0.142857 0.166667 0.2 0.25 0.333333 0.5)
#               1/9        1/8         1/7         1/6         1/5         1/4       1/3         1/2         1
schedule=(0.111111 0.125000 0.142857 0.166667 0.200000 0.250000 0.333333 0.500000 1.000000)
sigma=`cat ../opt_lambda_L1L2.data | gawk '{printf("%.4e", $2*$1/(1+(1-$2)*$1))}'`
echo "$sigma"

rm -f log_case2_fig8

echo -n >| gammas_AdaL1TV.data
k=0
for gamma in "${schedule[@]}"; do

	echo "$k $gamma" >> gammas_AdaL1TV.data

	printf -v idx "%03d" "$k"

	if [ ! -e model_AdaL1TV_"$idx".data ]; then
		"$root"/bin/l1tvinv -f ../input_mag.in -l "$lm_opt"\
			-g ../beta_L1L2_opt.vec:"$sigma":0.1:0.1 -c "$gamma" -s ../settings.par -v 2>&1 | tee -a "$log_fn"

		mv -f model.data model_AdaL1TV_"$idx".data
		mv -f recovered.data recovered_AdaL1TV_"$idx".data
		mv -f beta_L1TV.vec beta_AdaL1TV_"$idx".vec
		mv -f regularization.vec regularization_AdaL1TV_"$idx".vec
		for p in 0 x y z; do
			mv -f weight_d"$p".vec weight_d"$p"_"$idx".vec
		done
	fi

	k=$((k+1))
done

