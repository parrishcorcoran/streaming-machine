#!/bin/zsh
# run the given command only when no python process over 1.5 GB is running (checked every 60 s)
while [ -n "$(ps -axo rss=,command= | awk '$1>1500000 && /[Pp]ython/')" ]; do echo "waiting: $(ps -axo rss=,command= | awk '$1>1500000 && /[Pp]ython/' | head -1 | cut -c1-120)"; sleep 60; done
echo "free at $(date +%H:%M:%S); running: $*"
exec "$@"
