#!/bin/zsh
# work.py on every build, one process at a time (128 held-out windows, every snapshot)
cd ~/streaming-machine/blankgo
for s in "64 2" "64 4" "128 2" "128 4"; do
  set -- ${=s}
  for r in A B C; do
    python3 work.py width=$1 layers=$2 recipe=$r > work_w$1_L$2_$r.log 2>&1
  done
done
echo WORK_DONE > work_done.flag
