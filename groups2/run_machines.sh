#!/bin/zsh
# the whole machines end to end, one process at a time; before each, wait (60 s polls) while another python holds over 1.5 GB
cd /Users/abundancemachine/streaming-machine
L=groups2/machine/queue.log
run() {
  while ps -axo rss=,comm= | awk '$2 ~ /[Pp]ython/ && $1 > 1500000 {f=1} END {exit !f}'; do sleep 60; done
  echo "$(date +%H:%M:%S) START $*" >> $L
  "$@" > /dev/null 2>> groups2/machine/queue.err || echo "$(date +%H:%M:%S) FAILED $*" >> $L
  echo "$(date +%H:%M:%S) END $*" >> $L
}
run python3 base2_groups.py top set=refs
run python3 base2_groups.py refs texts=0,1,2 shuffle=0
run python3 base2_groups.py refs texts=0,1,2 order=7
run python3 base2_groups.py top set=orderfree order=1
run python3 base2_groups.py top set=refs light=shuffled
for a in 0 40 80 120 160 200 240 280; do run python3 base2_groups.py windows a=$a b=$((a + 40 > 300 ? 300 : a + 40)); done
run python3 base2_groups.py join
run python3 base2_groups.py top set=windows part=1/2
run python3 base2_groups.py top set=windows part=2/2
run python3 base2_groups.py judge
run python3 qwen_groups.py machine
run python3 qwen_groups.py top set=machine
run python3 qwen_groups.py words
run python3 qwen_groups.py top set=words
run python3 qwen_groups.py top set=orderfree
run python3 qwen_groups.py top set=machine light=shuffled
echo "$(date +%H:%M:%S) QUEUE DONE" >> $L
