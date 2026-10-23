#!/bin/zsh
# the remaining runs, one heavy process at a time (each waits while any other python process holds over 1.5 GB)
cd ~/streaming-machine
waitfree() { while ps -axo rss,command | grep -i python | grep -v grep | grep -v run_rest | awk '$1>1500000 {f=1} END {exit !f}'; do echo "waiting: another python process over 1.5 GB"; sleep 60; done; }
for job in "machine set=refs seed=7" "top set=refs seed=7" "machine set=words seed=1" "top set=words seed=1" "machine set=orderfree seed=1 reorder=4242" "top set=orderfree seed=1 reorder=4242 wordperm=1"; do
  waitfree; echo "START $job $(date +%T)"
  python3 confirm_groups/cg_machine.py ${=job} > "confirm_groups/out_${job// /_}.stdout" 2>&1 || echo "FAILED $job"
  tail -3 "confirm_groups/out_${job// /_}.stdout" | cut -c1-400
  echo "END $job $(date +%T)"
done
echo ALLDONE
