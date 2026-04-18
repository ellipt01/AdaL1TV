#!/usr/bin/env bash

root="../../"

log_fn="log_case3"

if [ ! -e "$root"/bin/l1l2inv ] || [ ! -e "$root"/bin/l1tvinv ]; then
	make TIMING=1 -C "$root"
	make install -C "$root"
fi

schedule=(0.0 0.25 1.5)

rm -f log_case3

### conventional L1TV
echo -n >| lambdas_L1TV.data
k=0
for log_lambda in $(seq "${schedule[@]}"); do

	echo "$k $log_lambda" >> lambdas_L1TV.data

	printf -v idx "%03d" "$k"

	if [ ! -e model_L1TV_"$idx".data ]; then
		"$root"/bin/l1tvinv -f input_mag.in -l "$log_lambda" -s settings.par -v 2>&1 | tee -a "$log_fn"

		mv -f model.data model_L1TV_"$idx".data
		mv -f recovered.data recovered_L1TV_"$idx".data
		mv -f beta_L1TV.vec beta_L1TV_"$idx".vec
		mv -f regularization.vec regularization_L1TV_"$idx".vec
	fi

	k=$((k+1))
done

opt_idx=`../bin/opt_lambda_lcurve_L1TV.py | gawk '{printf("%03d",$1)}'`
ln -sf model_L1TV_"$opt_idx".data model_L1TV_opt.data
ln -sf recovered_L1TV_"$opt_idx".data recovered_L1TV_opt.data
ln -sf beta_L1TV_"$opt_idx".vec beta_L1TV_opt.vec

### two-stage strategy
### preliminary L1L2 inversion
echo -n >| lambdas_L1L2.data
k=0
for log_lambda in $(seq "${schedule[@]}"); do

	echo "$k $log_lambda" >> lambdas_L1L2.data

	printf -v idx "%03d" "$k"

	if [ ! -e model_L1L2_"$idx".data ]; then
		"$root"/bin/l1l2inv -f input_mag.in -l "$log_lambda" -a 0.9 -s settings.par -v 2>&1 | tee -a "$log_fn"

		mv -f model.data model_L1L2_"$idx".data
		mv -f recovered.data recovered_L1L2_"$idx".data
		mv -f beta_L1L2.vec beta_L1L2_"$idx".vec
	fi
	
	k=$((k+1))
done

opt_idx=`../bin/opt_lambda_lcurve_L1L2.py | gawk '{printf("%03d",$1)}'`
ln -sf model_L1L2_"$opt_idx".data model_L1L2_opt.data
ln -sf recovered_L1L2_"$opt_idx".data recovered_L1L2_opt.data
ln -sf beta_L1L2_"$opt_idx".vec beta_L1L2_opt.vec

lm_opt=`cat lambdas_L1L2.data | gawk '{if($1=='$opt_idx'){printf("%.4f",10**$2)}}'`
echo "$lm_opt" >| opt_lambda_L1L2.data

### AdaL1TV: L1TV inversion with guide model
echo -n >| lambdas_AdaL1TV.data
k=0
for log_lambda in $(seq "${schedule[@]}"); do

	echo "$k $log_lambda" >> lambdas_AdaL1TV.data

	printf -v idx "%03d" "$k"

	if [ ! -e model_AdaL1TV_"$idx".data ]; then
		"$root"/bin/l1tvinv -f input_mag.in -l "$log_lambda"\
			-g beta_L1L2_opt.vec:"$lm_opt":0.1:0.1 -c 0.125 -s settings.par -v 2>&1 | tee -a log_case2

		mv -f model.data model_AdaL1TV_"$idx".data
		mv -f recovered.data recovered_AdaL1TV_"$idx".data
		mv -f beta_L1TV.vec beta_AdaL1TV_"$idx".vec
		mv -f regularization.vec regularization_AdaL1TV_"$idx".vec
	fi

	k=$((k+1))
done

opt_idx=`../bin/opt_lambda_lcurve_AdaL1TV.py | gawk '{printf("%03d",$1)}'`
ln -sf model_AdaL1TV_"$opt_idx".data model_AdaL1TV_opt.data
ln -sf recovered_AdaL1TV_"$opt_idx".data recovered_AdaL1TV_opt.data
ln -sf beta_AdaL1TV_"$opt_idx".vec beta_AdaL1TV_opt.vec

