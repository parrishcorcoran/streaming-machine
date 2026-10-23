#!/bin/zsh
# the checker's runs, one heavy process at a time (each about 2 GB)
cd /Users/abundancemachine/streaming-machine
wait_free() { while ps -axo rss,comm | awk '$1 > 1500000 && $2 ~ /[Pp]ython/' | grep -q .; do sleep 60; done; }
for args in "mode=refs seed=1" "mode=refs seed=7" "mode=refs seed=1 texts=0,1,2 reorder=5" "mode=windows a=0 b=300"; do
  wait_free
  echo "=== $args $(date)"
  python3 confirm_groups/rebuild_gpt2.py ${=args} 2>&1 | grep -v "Warning\|warnings.warn\|part = xf"
done
echo "=== all done $(date)"
