#!/bin/zsh
cd /Users/abundancemachine/streaming-machine
rm -f reach/answer/proj.10_b1_selftest.json
reach/when_free.sh /usr/bin/time -l python3 reach/answer.py plates=proj.10 selftest=1 > reach/answer_selftest_run3.out 2>&1 || { echo "selftest FAILED"; exit 1; }
reach/when_free.sh /usr/bin/time -l python3 reach/reach_check.py > reach/reach_check.out 2>&1
reach/when_free.sh /usr/bin/time -l python3 reach/answer.py > reach/answer_run.out 2>&1
python3 reach/answer_report.py > /dev/null 2>&1
echo chain2 done
