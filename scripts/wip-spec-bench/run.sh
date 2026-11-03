#!/bin/bash
# usage: run.sh TAG REPS TASKS PAR key=value ...  (preset overrides; trace into trace-TAG.txt)
S=$(dirname "$(readlink -f "$0")")
TAG=$1 REPS=$2 TASKS=$3 PAR=$4; shift 4
python3 $S/mkpreset.py $S/base.ini $S/p-$TAG.ini "$@"
rm -f $S/trace-$TAG.txt
if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; then echo "GPU busy"; exit 1; fi
LLAMA_SPEC_TRACE=$S/trace-$TAG.txt $S/start-test.sh $S/p-$TAG.ini 18900 > $S/srv-$TAG.log 2>&1 &
PID=$!
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:18900/health && break; sleep 1; done
curl -s -X POST http://127.0.0.1:18900/models/load -H 'Content-Type: application/json' -d '{"model":"Qwen3.8-27b"}' > /dev/null
for i in $(seq 1 180); do st=$(curl -s http://127.0.0.1:18900/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["status"]["value"])' 2>/dev/null); [ "$st" = loaded ] && break; [ "$st" = unloaded ] && [ $i -gt 10 ] && break; sleep 2; done
echo "model: $st"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
if [ "$st" = loaded ]; then python3 $S/bench.py 18900 $TAG $REPS "$TASKS" $PAR; fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
kill $PID; wait $PID 2>/dev/null
for i in $(seq 1 60); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
