#!/bin/zsh
# the per-place blank removal for every build (after that build's chain is through), then the w64 L2 checkers again
# with the blank-out and scrambled rebuild (added to checker.py after their first run)
cd ~/streaming-machine/blankgo
for r in A B C; do
  python3 checker.py width=64 layers=2 recipe=$r 2>&1 | grep -v -i warn > checker_w64_L2_$r.log
done
for s in "64 2" "64 4" "128 2" "128 4"; do
  set -- ${=s}
  for r in A B C; do
    n=w$1_L$2_$r
    until grep -q "^worst" checker_$n.log 2>/dev/null; do sleep 30; done
    python3 where.py width=$1 layers=$2 recipe=$r 2>&1 | grep -v -i warn > where_$n.log
  done
done
echo WHERE_DONE > where_done.flag
