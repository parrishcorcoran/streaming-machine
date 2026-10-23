#!/bin/zsh
# remove.py then workcheck.py, split into two chains (the first chain, run_remove.sh, was stopped after it started w64_L4_A)
cd ~/streaming-machine/blankgo
for n in w128_L4_A w128_L4_B w128_L4_C; do
  w=${n#w}; d=${w%%_*}; rest=${n#*_L}; L=${rest%%_*}; r=${n##*_}
  until grep -q "^done" work_$n.log 2>/dev/null; do sleep 30; done
  python3 remove.py width=$d layers=$L recipe=$r > remove_$n.log 2>&1
  python3 workcheck.py width=$d layers=$L recipe=$r 2>&1 | grep -v -i warn > workcheck_$n.log
done
echo DONE > remove2_done.flag
