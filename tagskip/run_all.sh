#!/bin/zsh
# one heavy process at a time, in this order
cd /Users/abundancemachine/streaming-machine
for args in "mode=work skip=none" "mode=work skip=all shufmeas=1" "mode=work skip=all" "mode=work skip=all order=1" \
            "mode=work skip=L11 shufmeas=1" "mode=work skip=L6-11 shufmeas=1" \
            "mode=judge skip=none" "mode=judge skip=all" "mode=judge skip=L11" "mode=judge skip=L6-11" \
            "mode=judge skip=all tags=shuffledtags" "mode=judge skip=all tags=shuffledlight"; do
  echo "=== $args  $(date +%H:%M:%S)"
  python3 tagskip/tagskip.py ${=args} > /dev/null 2> tagskip/err_$(echo $args | tr ' =' '__').txt || echo "FAILED: $args"
done
echo "=== done $(date +%H:%M:%S)"
