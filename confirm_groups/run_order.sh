#!/bin/zsh
# the order-free pair on refs texts 0-2 (after the check-path fix for relabelled words), one heavy process at a time
cd /Users/abundancemachine/streaming-machine
wait_free() { while ps -axo rss,comm | awk '$1 > 1500000 && $2 ~ /[Pp]ython/' | grep -q .; do sleep 60; done; }
for args in "mode=refs seed=1 texts=0,1,2" "mode=refs seed=1 texts=0,1,2 reorder=5"; do
  wait_free
  echo "=== $args $(date)"
  python3 confirm_groups/rebuild_gpt2.py ${=args} 2>&1 | grep -v "Warning\|warnings.warn\|part = xf"
done
python3 confirm_groups/orderfree_compare.py > confirm_groups/orderfree_compare.log 2>&1
echo "=== all done $(date)"
