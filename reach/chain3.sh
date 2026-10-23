#!/bin/zsh
cd /Users/abundancemachine/streaming-machine
rm -f reach/answer/head_b1_selftest.json
reach/when_free.sh /usr/bin/time -l python3 reach/answer.py plates=head selftest=1 > reach/answer_selftest_head.out 2>&1 || { echo "head selftest FAILED"; exit 1; }
reach/when_free.sh /usr/bin/time -l python3 reach/answer.py > reach/answer_run.out 2>&1
reach/when_free.sh /usr/bin/time -l python3 reach/reach_check.py > reach/reach_check.out 2>&1
reach/when_free.sh python3 reach/answer_report.py > /dev/null 2>&1
echo chain3 done
