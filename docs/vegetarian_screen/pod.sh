#!/usr/bin/env bash
# Minimal RunPod REST helper. Usage: pod.sh create | info ID | ssh ID CMD | delete ID
set -euo pipefail
set -a; source ~/j-lens/.env; set +a
API=https://rest.runpod.io/v1
H=(-H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json")
case "$1" in
create)
  body=$(python3 - <<PY
import json; print(json.dumps({"name":"traitscreen","imageName":"runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
 "gpuTypeIds":["${GPU:-NVIDIA A40}"],"gpuCount":1,"cloudType":"${CLOUD:-SECURE}","containerDiskInGb":80,"volumeInGb":0,
 "ports":["22/tcp"],
 "env":{"PUBLIC_KEY":open("$HOME/.ssh/id_ed25519.pub").read().strip()}}))
PY
)
  curl -s -X POST "${H[@]}" -d "$body" $API/pods ;;
info) curl -s "${H[@]}" $API/pods/$2 ;;
delete) curl -s -X DELETE "${H[@]}" $API/pods/$2 -w "%{http_code}\n" ;;
list) curl -s "${H[@]}" $API/pods ;;
esac
