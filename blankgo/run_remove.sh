#!/bin/zsh
# remove.py then workcheck.py for every build, each after that build's work.py is through; one process in this chain at a time
cd ~/streaming-machine/blankgo
for s in "64 2" "64 4" "128 2" "128 4"; do
  set -- ${=s}
  for r in A B C; do
    n=w$1_L$2_$r
    until grep -q "^done" work_$n.log 2>/dev/null; do sleep 30; done
    python3 remove.py width=$1 layers=$2 recipe=$r > remove_$n.log 2>&1
    python3 workcheck.py width=$1 layers=$2 recipe=$r 2>&1 | grep -v -i warn > workcheck_$n.log
  done
done
echo REMOVE_DONE > remove_done.flag
