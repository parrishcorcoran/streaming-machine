#!/bin/zsh
cd /Users/abundancemachine/streaming-machine
reach/when_free.sh /usr/bin/time -l python3 reach/answer.py > reach/answer_run3.out 2>&1
reach/when_free.sh python3 reach/answer_report.py > /dev/null 2>&1
echo chain5 done
