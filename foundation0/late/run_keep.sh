#!/bin/zsh
# waits (60 s polls) while any other python process over 1.0 GB runs (stricter than the 1.5 GB rule), then runs one
# heavy process at a time
cd ~/streaming-machine/foundation0/late
waitclear() { while ps -axo rss=,command= | awk '/[Pp]ython/ && $1 > 1048576' | grep -q .; do echo "$(date +%T) waiting: $(ps -axo pid=,rss=,command= | awk '/[Pp]ython/ && $2 > 1048576 {print $1, $2/1048576 " GB", $4, $5, $6}')"; sleep 60; done; }
step() { local out=$1; shift; waitclear; echo "$(date +%T) start $*"; /usr/bin/time -l python3 "$@" > $out.out 2> $out.err; local rc=$?; echo "$(date +%T) end $* (exit $rc)"; }
if [ "$1" = smoke ]; then step run_keep_smoke keep_late.py size=70m texts=2 greedy=0 cases=b,k tag=smoke; exit; fi
step run_keep_160m keep_late.py size=160m
step run_keep_70m keep_late.py size=70m
step run_keep_160m_shuffle keep_late.py size=160m order=shuffle cases=b,k greedy=0
step run_keep_70m_shuffle keep_late.py size=70m order=shuffle cases=b,k greedy=0
step run_checker_70m keep_checker.py size=70m
step run_checker_160m keep_checker.py size=160m
echo "$(date +%T) all done"
