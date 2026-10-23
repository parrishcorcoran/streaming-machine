#!/bin/zsh
# the backward runs, one heavy process at a time
cd /Users/abundancemachine/streaming-machine
python3 backward/req.py set=refs layers=10,9 points=QKV:q,QKV:k,QKV:v > backward/out/req_refs_L10-9_qkv.stdout 2>&1
python3 backward/diag.py > backward/out/diag_refs.stdout 2>&1
python3 backward/stops.py set=refs verify=0,5,10,11 > backward/out/stops_refs.stdout 2>&1
python3 backward/top.py set=windows > backward/out/top_windows.stdout 2>&1
python3 backward/top.py set=refs light=shuffled > backward/out/top_refs_shuffled.stdout 2>&1
python3 backward/top.py set=refs order=7 > backward/out/top_refs_order7.stdout 2>&1
python3 backward/req.py set=windows layers=11 points=QKV:q,A,H,Mw > backward/out/req_windows_L11.stdout 2>&1
python3 backward/stops.py set=windows layers=11 verify=11 > backward/out/stops_windows.stdout 2>&1
python3 backward/stops.py set=refs layers=0,5,11 light=shuffled verify= > backward/out/stops_refs_shuffled.stdout 2>&1
echo ALLDONE > backward/out/run_all.done
