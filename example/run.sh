#!/usr/bin/env bash

root="../"

if [ ! -e $root/bin/l1l2inv -o ! -e $root/bin/l1tvinv ]; then
	make TIMING=1 -C $root
	make install -C $root
fi

$root/bin/l1l2inv -f input_mag.in -l -0.25 -a 0.9 -s settings.par -s settings.par -v 2>&1 | tee log_l1l2
$root/bin/l1tvinv -f input_mag.in -l 0. -g beta_L1L2.vec:0.5623:0.1:0.1 -c 0.125 -s settings.par -v 2>&1 | tee log_AdaL1TV
