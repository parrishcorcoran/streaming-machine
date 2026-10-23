#!/bin/zsh
# the per-place choice, after the rebuild queue, one process at a time
cd /Users/abundancemachine/streaming-machine
until grep -q finished groups2/out/run_confirm.err 2>/dev/null; do sleep 60; done
for m in gpt2 qwen; do for g in 8 12; do
  while ps -axo rss=,comm= | awk '$2 ~ /[Pp]ython/ && $1 > 1500000 {f=1} END {exit !f}'; do sleep 60; done
  python3 groups2/uh_perplace.py model=$m g=$g > /dev/null 2>> groups2/out/run_perplace.err
done; done
echo finished >> groups2/out/run_perplace.err
