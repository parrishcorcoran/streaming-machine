#!/bin/zsh
# all twelve builds, one at a time (smallest first)
cd ~/streaming-machine/blankgo
for s in "64 2" "64 4" "128 2" "128 4"; do
  set -- ${=s}
  for r in A B C; do
    python3 build.py width=$1 layers=$2 recipe=$r 2>&1 | grep -v -i warn > build_w$1_L$2_$r.log
  done
done
echo ALL_DONE >> build_w128_L4_C.log
