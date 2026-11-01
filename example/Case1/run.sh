#!/usr/bin/env bash

root="../../"
fn_log="log_case1"

if [ ! -e "$root"/bin/l1tvinv ]; then
        make TIMING=1 -C "$root"
        make install -C "$root"
fi

rm -f log_case1
"$root"/bin/l1tvinv -f input_mag.in -l -1. -n 50/50/25 -r -2.5/2.5/-2.5/2.5/0./2.5\
	-d 45./-7./45./-7. -b 1./0.0/100. -i 1.0e-4/100000\
	-g true.guide -s 1.e-4/1./1. -c -1. -v 2>&1 | tee -a "$fn_log"
