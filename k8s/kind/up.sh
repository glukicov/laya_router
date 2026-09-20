#!/usr/bin/env bash
# Build the CPU image, create a local kind cluster, and serve /route on http://127.0.0.1:8080.
#
# Every kubectl call names the context explicitly and writes to a dedicated kubeconfig, so whatever context your
# shell has selected is never touched.
set -euo pipefail

export KUBECONFIG="${LAYA_KUBECONFIG:-$HOME/.kube/laya-router}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
CTX=kind-laya
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

if [ ! -d "$HF_CACHE_DIR" ]; then
  echo "No Hugging Face cache at $HF_CACHE_DIR." >&2
  echo "Run 'uv run laya-router classify \"hello\"' once to download the checkpoint, then retry." >&2
  exit 1
fi

echo "==> Building laya-router:cpu"
docker build -f "$ROOT/docker/service.Dockerfile" -t laya-router:cpu "$ROOT"

if ! kind get clusters | grep -qx laya; then
  echo "==> Creating the kind cluster"
  envsubst < "$HERE/cluster.yaml" | kind create cluster --config - --wait 120s
fi

echo "==> Loading the image into the cluster"
kind load docker-image laya-router:cpu --name laya

echo "==> Applying manifests"
kubectl --context "$CTX" apply -f "$HERE/deployment.yaml"

echo "==> Waiting for the model to load and warm (CPU, so this is slower than the native server)"
kubectl --context "$CTX" -n laya rollout status deployment/laya-router --timeout=600s

cat <<'MSG'

Ready. Try it:

  curl -s localhost:8080/health | python3 -m json.tool
  curl -s localhost:8080/route -H 'content-type: application/json' \
    -d '{"message":"The API is returning 500 errors after today deploy."}' | python3 -m json.tool

Tear down with k8s/kind/down.sh
MSG
