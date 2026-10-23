#!/bin/zsh
# dodge_exact.py on the other 11 builds, one at a time
cd ~/streaming-machine/blankgo
for s in "64 2" "64 4" "128 2" "128 4"; do
  set -- ${=s}
  for r in A B C; do
    n=w$1_L$2_$r
    [ "$n" = "w64_L2_A" ] && continue
    python3 dodge_exact.py width=$1 layers=$2 recipe=$r > dodge_exact_$n.log 2>&1
  done
done
echo DODGE_DONE > dodge_done.flag
