#!/bin/zsh
# flat build 1: every run of base2_groups_readers.py, one process at a time; each waits (polling 60 s) while another
# python process over 1.5 GB runs. Stops at the first failure (every run asserts its own exactness).
cd /Users/abundancemachine/streaming-machine
Q=flat_build/1/machine/queue.log; E=flat_build/1/machine/queue.err
run() {
  while ps -axo rss=,comm= | awk '$2 ~ /[Pp]ython/ && $1 > 1500000 {f=1} END {exit !f}'; do sleep 60; done
  echo "$(date +%H:%M:%S) START $*" >> $Q
  if python3 flat_build/1/base2_groups_readers.py "$@" > /dev/null 2>> $E; then echo "$(date +%H:%M:%S) END $*" >> $Q
  else echo "$(date +%H:%M:%S) FAILED $*" >> $Q; exit 1; fi
}
run refs
run refs order=5
run top set=refs
run top set=orderfree order=1
run top set=refs light=shuffled
for ab in "0 40" "40 80" "80 120" "120 160" "160 200" "200 240" "240 280" "280 300"; do
  a=${ab% *}; b=${ab#* }; run windows a=$a b=$b
done
run join
for i in 1 2 3 4 5 6 7 8; do run top set=windows part=$i/8; done
run judge
run report
echo "$(date +%H:%M:%S) QUEUE DONE" >> $Q
