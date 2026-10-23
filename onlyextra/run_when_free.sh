#!/bin/zsh
# one heavy process at a time: start only after two polls in a row (60 s apart) find no python process over 1.5 GB
cd /Users/abundancemachine/streaming-machine
clean=0
while [ $clean -lt 2 ]; do
  big=$(ps -axo pid=,rss=,comm= | awk '/ython/ && $2 > 1500000 {print $1}' | head -1)
  if [ -z "$big" ]; then clean=$((clean+1)); else clean=0; echo "waiting on $big $(date +%H:%M:%S)"; fi
  [ $clean -lt 2 ] && sleep 60
done
echo "=== start $(date +%H:%M:%S): $*"
python3 onlyextra/onlyextra.py "$@"
echo "=== exit $? $(date +%H:%M:%S)"
