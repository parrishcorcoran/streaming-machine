#!/bin/zsh
cd /Users/abundancemachine/streaming-machine
reach/when_free.sh /usr/bin/time -l python3 reach/reach_check.py > reach/reach_check.out 2>&1
reach/when_free.sh /usr/bin/time -l python3 reach/answer.py cache=1 > reach/answer_cache.out 2>&1
rm -f reach/answer/qkv.11_b1_selftest.json
reach/when_free.sh /usr/bin/time -l python3 reach/answer.py plates=proj.10 selftest=1 > reach/answer_selftest_run2.out 2>&1
echo chain1 done
