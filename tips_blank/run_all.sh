#!/bin/zsh
# Every run of tips_blank/, one after another (one heavy process at a time). Before each run: wait (60 s polls) while
# any other python process holds over 1.5 GB.
cd /Users/abundancemachine/streaming-machine
wait_clear() {
  while ps -axo pid=,rss=,command= | awk '$2 > 1500000 && $3 ~ /[Pp]ython/ {print}' | grep -q .; do
    echo "  waiting: another python process over 1.5 GB ($(date +%H:%M:%S))"; sleep 60
  done
}
run() { wait_clear; echo "=== $* ($(date +%H:%M:%S))"; python3 "$@" || { echo "FAILED: $*"; exit 1; }; }
run tips_blank/pythia_tips.py size=70m      > tips_blank/pythia_70m.log 2>&1
run tips_blank/pythia_tips.py size=160m     > tips_blank/pythia_160m.log 2>&1
for r in A B C; do run tips_blank/tiny_tips.py recipe=$r > tips_blank/tiny_w128_L4_$r.log 2>&1; done
run tips_blank/gpt2_tips.py                 > tips_blank/gpt2.log 2>&1
run tips_blank/qwen_tips.py                 > tips_blank/qwen.log 2>&1
echo "=== all done ($(date +%H:%M:%S))"
