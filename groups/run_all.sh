#!/bin/zsh
# Every run of the group machine, one after another (one heavy process at a time). Before each run: wait (60 s polls)
# while any other python process holds over 1.5 GB.
cd /Users/abundancemachine/streaming-machine
wait_clear() {
  while ps -axo pid=,rss=,command= | awk -v me=$$ '$2 > 1500000 && $3 ~ /[Pp]ython/ {print}' | grep -q .; do
    echo "  waiting: another python process over 1.5 GB ($(date +%H:%M:%S))"; sleep 60
  done
}
for spec in "$@"; do
  wait_clear
  echo "=== $spec ($(date +%H:%M:%S))"
  python3 groups/groupmachine.py ${=spec} || { echo "FAILED: $spec"; exit 1; }
done
echo "=== all done ($(date +%H:%M:%S))"
