#!/bin/bash
# 10 rounds, the machines alternating in each round (the order turned each round); one process at a time; all built -O
cd ~/streaming-machine
for r in 1 2 3 4 5 6 7 8 9 10; do
  if [ $((r % 2)) -eq 1 ]; then order="orig fr3 fr2nofuse fr2"; else order="fr2 fr2nofuse fr3 orig"; fi
  for m in $order; do
    case $m in
      orig)      ./gpu_fr2/base2_stream all=1 quiet=1 > gpu_fr2/s_orig_$r.log 2>&1 ;;
      fr3)       ./gpu_fr2/base2_stream_fr all=1 quiet=1 planes=3 out=gpu_fr2/s_fr3_$r.json > gpu_fr2/s_fr3_$r.log 2>&1 ;;
      fr2nofuse) ./gpu_fr2/base2_stream_fr2 all=1 quiet=1 fuse=0 out=gpu_fr2/s_fr2nofuse_$r.json > gpu_fr2/s_fr2nofuse_$r.log 2>&1 ;;
      fr2)       ./gpu_fr2/base2_stream_fr2 all=1 quiet=1 out=gpu_fr2/s_fr2_$r.json > gpu_fr2/s_fr2_$r.log 2>&1 ;;
    esac
  done
done
