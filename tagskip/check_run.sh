#!/bin/zsh
# the checker's runs, one at a time; waits (60 s polls) while any other python process is over 1 GB (onlyextra swings to 1.56 GB)
cd /Users/abundancemachine/streaming-machine
waitfree() { while pgrep -f onlyextra.py > /dev/null || ps -axo rss=,command= | awk '$1 > 1000000 && /[Pp]ython/' | grep -v check_tagskip | grep -q .; do echo "waiting $(date +%H:%M:%S)"; sleep 60; done; }
for p in C D A B; do
  waitfree
  echo "=== part=$p $(date +%H:%M:%S)"
  /usr/bin/time -l python3 tagskip/check_tagskip.py part=$p > /dev/null 2> tagskip/check_err_$p.txt || echo "FAILED part=$p"
  grep "maximum resident" tagskip/check_err_$p.txt
done
echo "=== done $(date +%H:%M:%S)"
