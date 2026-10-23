#!/bin/zsh
# one heavy process at a time: waits (60 s polls) while any other python process over 1.5 GB runs, then runs the next
cd ~/streaming-machine
# (also waits while the other python processes together hold over 2.5 GB: more careful than the rule, never less)
busy() { ps -axo rss=,command= | awk '/[Pp]ython/ && !/remove_exact|store_exact|checker_exact/ { if ($1 > 1572864) big = 1; tot += $1 } END { exit !(big || tot > 2621440) }'; }
waitclear() { while busy; do echo "$(date +%T) waiting: $(ps -axo pid=,rss=,command= | awk '/[Pp]ython/ && $2 > 1572864 {print $1, $2/1048576 " GB", $4, $5}')"; sleep 60; done; }
for size in 70m 160m; do   # 14m (plain and order-free) was run by hand first
  if true; then waitclear; echo "$(date +%T) start $size"; /usr/bin/time -l python3 foundation0/exact/remove_exact.py size=$size > foundation0/exact/run_words_$size.out 2> foundation0/exact/run_words_$size.err; fi   # 14m plain was run by hand first
  waitclear; echo "$(date +%T) start $size order-free"; /usr/bin/time -l python3 foundation0/exact/remove_exact.py size=$size order=shuffle cases=fin,k,p > foundation0/exact/run_words_${size}_shuffle.out 2> foundation0/exact/run_words_${size}_shuffle.err
done
echo "$(date +%T) all done"
