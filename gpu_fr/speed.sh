#!/bin/bash
# 5 rounds, the four machines interleaved in each round; one process at a time
cd ~/streaming-machine
for r in 1 2 3 4 5; do
  ./gpu_fr/base2_stream_orig all=1 quiet=1 > gpu_fr/speed_orig_$r.log 2>&1
  ./gpu_fr/base2_stream_fr all=1 quiet=1 top=plain > gpu_fr/speed_plain_$r.log 2>&1
  ./gpu_fr/base2_stream_fr all=1 quiet=1 > gpu_fr/speed_fr_$r.log 2>&1
  ./gpu_fr/base2_stream_fr all=1 quiet=1 entry=1 > gpu_fr/speed_fr_entry_$r.log 2>&1
done
