#!/bin/bash
# groups2/run_top.sh gpt2|qwen -- the top-table group runs, one heavy process at a time. Before each run it waits, in
# 60 s polls, while any other python process holds over 1.5 GB (tips_blank/ or anything else; nothing is stopped).
# Qwen's runs hold one or two forms' patterns at a time (uh,pp then sd) to stay well under 3 GB.
cd /Users/abundancemachine/streaming-machine
ERR=groups2/out/run_top_$1.err
wait_turn() {
  while ps -axo rss=,command= | awk '$0 ~ /[Pp]ython/ && $1 > 1500000 {f = 1} END {exit !f}'; do
    echo "$(date +%T) waiting: another python process over 1.5 GB"; sleep 60
  done
}
run() { wait_turn; echo "$(date +%T) START $*"; python3 groups2/topg.py "$@" > /dev/null 2>> $ERR || echo "$(date +%T) FAILED $*"; echo "$(date +%T) END $*"; }
if [ "$1" = gpt2 ]; then
  run model=gpt2 set=refs g=16 run=main
  run model=gpt2 set=windows g=16 run=main
  run model=gpt2 set=refs g=20 run=main
  run model=gpt2 set=windows g=20 run=main
  run model=gpt2 set=refs g=16 run=seed2
  run model=gpt2 set=refs g=16 run=shuffle
  run model=gpt2 set=refs g=16 run=order
else
  for F in uh,pp sd; do run model=qwen set=machine g=16 run=main forms=$F; done
  for F in uh,pp sd; do run model=qwen set=words g=16 run=main forms=$F; done
  for F in uh,pp sd; do run model=qwen set=machine g=20 run=main forms=$F; done
  for R in seed2 shuffle order; do for F in uh,pp sd; do run model=qwen set=machine g=16 run=$R forms=$F; done; done
fi
echo "$(date +%T) QUEUE DONE"
