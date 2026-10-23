#!/bin/zsh
# after every build: the float side, the conversion, the machine (plain, then the order-free and int64 checks at
# step 3000), and the independent checker; one process at a time
cd ~/streaming-machine/blankgo
for s in "64 2" "64 4" "128 2" "128 4"; do
  set -- ${=s}
  for r in A B C; do
    n=w$1_L$2_$r
    until grep -q "^done" build_$n.log 2>/dev/null; do sleep 30; done
    python3 eval.py width=$1 layers=$2 recipe=$r 2>&1 | grep -v -i warn > eval_$n.log
    python3 convert.py width=$1 layers=$2 recipe=$r 2>&1 | grep -v -i warn > convert_$n.log
    rm -f words/words_$n.json words/words_$n-shuffle.json words/words_$n-matmulint.json
    python3 machine.py width=$1 layers=$2 recipe=$r 2>&1 | grep -v -i warn > machine_$n.log
    python3 machine.py width=$1 layers=$2 recipe=$r steps=3000 plates=finished,pushes,scr1 order=shuffle 2>&1 | grep -v -i warn > machine_$n-shuffle.log
    python3 machine.py width=$1 layers=$2 recipe=$r steps=3000 plates=finished matmul=int 2>&1 | grep -v -i warn > machine_$n-matmulint.log
    python3 checker.py width=$1 layers=$2 recipe=$r 2>&1 | grep -v -i warn > checker_$n.log
  done
done
echo REST_DONE > rest_done.flag
