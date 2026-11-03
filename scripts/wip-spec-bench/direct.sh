#!/bin/bash
# usage: direct.sh LOG [extra args...]: the child server with the captured production args on :18900, in the background
S=$(dirname "$(readlink -f "$0")")
LOG=$1; shift
P2P=/home/piggidragon/Services/llama-configs/p2p-bench
loaded=$(cat /sys/module/nvidia/version 2>/dev/null)
if [ -n "$loaded" ] && [ "$(cat "$P2P/lib/VERSION" 2>/dev/null)" = "$loaded" ] && nvidia-smi topo -p2p r 2>/dev/null | grep -q OK; then
  export LD_LIBRARY_PATH="$P2P/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" NCCL_P2P_LEVEL=SYS
fi
mapfile -t ARGS < <(python3 -c "
import json,sys
a=json.load(open('$S/child-args.json'))
a[0]='${LLAMA_BIN:-/home/piggidragon/Services/llama.cpp-mtp/build/bin/llama-server}'
i=a.index('--port'); a[i+1]='18900'
print('\n'.join(a))")
"${ARGS[@]}" "$@" > "$LOG" 2>&1 &
echo $!
