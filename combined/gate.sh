#!/bin/zsh
# Before each run: list python processes over 1.5 GB (resident). Exit 1 if any is running (then wait and look again).
big=$(ps -axo pid=,rss=,command= | awk '/[Pp]ython/ && $2 > 1572864 {print}')
if [[ -n "$big" ]]; then echo "WAIT: python over 1.5 GB running:"; echo "$big"; exit 1; fi
echo "clear: no python process over 1.5 GB"; ps -axo pid=,rss=,command= | awk '/[Pp]ython/ && $2 > 200000 {printf "  running: pid %s, %d MB, %s\n", $1, $2/1024, substr($0, index($0,$3), 90)}'
