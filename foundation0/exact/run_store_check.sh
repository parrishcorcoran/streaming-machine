#!/bin/zsh
# after run_words.sh: storage and the independent checker for 70m and 160m, one heavy process at a time
cd ~/streaming-machine
busy() { ps -axo rss=,command= | awk '/[Pp]ython/ && !/remove_exact|store_exact|checker_exact/ { if ($1 > 1572864) big = 1; tot += $1 } END { exit !(big || tot > 2621440) }'; }
waitclear() { while pgrep -f run_words.sh > /dev/null || busy; do echo "$(date +%T) waiting"; sleep 60; done; }
for size in 70m 160m; do
  waitclear; echo "$(date +%T) store $size"; /usr/bin/time -l python3 foundation0/exact/store_exact.py size=$size > foundation0/exact/run_store_$size.out 2> foundation0/exact/run_store_$size.err
  waitclear; echo "$(date +%T) checker $size"; /usr/bin/time -l python3 foundation0/exact/checker_exact.py size=$size > foundation0/exact/run_checker_$size.out 2> foundation0/exact/run_checker_$size.err
done
echo "$(date +%T) all done"
