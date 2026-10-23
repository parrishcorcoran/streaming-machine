#!/bin/zsh
# the up/hole rebuild, one process at a time: GPT-2 then Qwen3-0.6B, g = 8 then 12, all four runs each
cd /Users/abundancemachine/streaming-machine
for m in gpt2 qwen; do for g in 8 12; do
  while ps -axo rss=,comm= | awk '$2 ~ /[Pp]ython/ && $1 > 1500000 {f=1} END {exit !f}'; do sleep 60; done
  python3 groups2/uh_rebuild.py model=$m g=$g > /dev/null 2>> groups2/out/run_confirm.err
done; done
echo finished >> groups2/out/run_confirm.err
