#!/usr/bin/env bash

root="../../"

if [ ! -e "$root"/bin/l1tvinv ]; then
        make TIMING=1 -C "$root"
        make install -C "$root"
fi

rm -f log_case1
"$root"/bin/l1tvinv -f input_mag.in -l -2. -g true.vec:1.e-4:1.:1. -c 1. -s settings.par -v 2>&1 | tee -a log_case1
