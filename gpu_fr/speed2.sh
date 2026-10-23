#!/bin/bash
# 5 rounds, the six machines interleaved in each round; one process at a time; each run saves its words and speeds
cd ~/streaming-machine
for r in 1 2 3 4 5; do
  ./gpu_fr/base2_stream_orig all=1 quiet=1 > gpu_fr/t_orig_$r.log 2>&1
  ./gpu_fr/base2_stream_fr all=1 quiet=1 top=plain out=gpu_fr/t_plain_$r.json > gpu_fr/t_plain_$r.log 2>&1
  ./gpu_fr/base2_stream_fr all=1 quiet=1 out=gpu_fr/t_fr2_$r.json > gpu_fr/t_fr2_$r.log 2>&1
  ./gpu_fr/base2_stream_fr all=1 quiet=1 entry=1 out=gpu_fr/t_fr2e_$r.json > gpu_fr/t_fr2e_$r.log 2>&1
  ./gpu_fr/base2_stream_fr all=1 quiet=1 planes=3 out=gpu_fr/t_fr3_$r.json > gpu_fr/t_fr3_$r.log 2>&1
  ./gpu_fr/base2_stream_fr all=1 quiet=1 planes=3 entry=1 out=gpu_fr/t_fr3e_$r.json > gpu_fr/t_fr3e_$r.log 2>&1
done
