#!/bin/zsh
# waits (60 s polls) while any other python process over 1.5 GB runs, then runs one heavy process at a time
cd ~/streaming-machine/foundation0/late
waitclear() { while ps -axo rss=,command= | awk '/[Pp]ython/ && $1 > 1572864' | grep -q .; do echo "$(date +%T) waiting: $(ps -axo pid=,rss=,command= | awk '/[Pp]ython/ && $2 > 1572864 {print $1, $2/1048576 " GB", $4, $5}')"; sleep 60; done; }
waitclear; echo "$(date +%T) start 160m"; /usr/bin/time -l python3 plate_look.py size=160m > run_160m.out 2> run_160m.err
waitclear; echo "$(date +%T) start 70m";  /usr/bin/time -l python3 plate_look.py size=70m > run_70m.out 2> run_70m.err
waitclear; echo "$(date +%T) start order-free check"; /usr/bin/time -l python3 plate_look.py size=160m order=shuffle cases=b:qkv.11,cb:qkv.11@20261006,b:fc.0,b:K=qkv > run_160m_shuffle.out 2> run_160m_shuffle.err
echo "$(date +%T) all done"
