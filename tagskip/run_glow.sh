#!/bin/zsh
cd /Users/abundancemachine/streaming-machine
for args in "mode=work skip=all glow=stored" "mode=work skip=L6-11 glow=stored" "mode=work skip=L11 glow=stored" \
            "mode=judge skip=all glow=stored" "mode=judge skip=L6-11 glow=stored" "mode=judge skip=L11 glow=stored" \
            "mode=judge skip=all tags=shuffledtags glow=stored"; do
  echo "=== $args  $(date +%H:%M:%S)"
  python3 tagskip/tagskip.py ${=args} > /dev/null 2> tagskip/err_$(echo $args | tr ' =' '__').txt || echo "FAILED: $args"
done
echo "=== done $(date +%H:%M:%S)"
