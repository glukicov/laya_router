#!/usr/bin/env bash
# Delete the local cluster. Nothing else on the machine is touched.
set -euo pipefail
export KUBECONFIG="${LAYA_KUBECONFIG:-$HOME/.kube/laya-router}"
kind delete cluster --name laya
