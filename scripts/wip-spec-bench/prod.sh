#!/bin/bash
# usage: prod.sh unload|load  (production Qwen3.8-27b in the router on :5803)
case $1 in
unload) curl -s -X POST http://127.0.0.1:5803/models/unload -H 'Content-Type: application/json' -d '{"model":"Qwen3.8-27b"}'; echo
        for i in $(seq 1 60); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 3; done ;;
load)   [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo "GPU busy"; exit 1; }
        curl -s -X POST http://127.0.0.1:5803/models/load -H 'Content-Type: application/json' -d '{"model":"Qwen3.8-27b"}'; echo
        for i in $(seq 1 60); do st=$(curl -s http://127.0.0.1:5803/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["status"]["value"])' 2>/dev/null); [ "$st" = loaded ] && break; sleep 2; done; echo "production: $st" ;;
esac
