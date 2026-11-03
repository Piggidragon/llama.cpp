#!/bin/bash
# usage: start-test.sh PRESET [PORT]  (LLAMA_BIN overrides the binary)
PRESET=$1; PORT=${2:-18900}
BIN=${LLAMA_BIN:-/home/piggidragon/Services/llama.cpp-mtp/build/bin/llama-server}
P2P=/home/piggidragon/Services/llama-configs/p2p-bench
loaded=$(cat /sys/module/nvidia/version 2>/dev/null)
p2p_ok() {
  nvidia-smi topo -p2p r 2>/dev/null | awk '/^ *GPU[0-9]+[ \t]/ { for (i = 2; i <= NF; i++) { if ($i == "OK") ok++; else if ($i != "X") bad++ } } END { exit !(ok > 0 && bad == 0) }'
}
if [ -n "$loaded" ] && [ "$(cat "$P2P/lib/VERSION" 2>/dev/null)" = "$loaded" ] && p2p_ok; then
  export LD_LIBRARY_PATH="$P2P/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" NCCL_P2P_LEVEL=SYS
  echo "start-test.sh: P2P on" >&2
else
  echo "start-test.sh: P2P off" >&2
fi
exec "$BIN" --host 127.0.0.1 --port "$PORT" --models-preset "$PRESET" --models-max 1
