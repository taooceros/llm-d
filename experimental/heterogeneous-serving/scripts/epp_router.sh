#!/usr/bin/env bash
# Roll the llm-d EPP router (deploy/k8s/omp-epp-router.yaml) onto one arm config.
# Usage: OMP_EPP_IMAGE=<registry>/llm-d-router-epp:<tag> scripts/epp_router.sh <plain|reserve|latebinding>
#        scripts/epp_router.sh --down
# Requires the deployed layout's Python gateway (gateway-up) so engine pods carry
# the omp.llm-d.ai/instance=true, role, tp and max-model-len labels.
set -euo pipefail
cd "$(dirname "$0")/.."
NS=${NS:-llm-d-optimized-baseline}
if [ "${1:-}" = "--down" ]; then
  kubectl -n "$NS" delete -f deploy/k8s/omp-epp-router.yaml --ignore-not-found
  kubectl -n "$NS" delete configmap omp-epp-config --ignore-not-found
  exit 0
fi
arm=${1:?usage: scripts/epp_router.sh <plain|reserve|latebinding>}
image=${OMP_EPP_IMAGE:?set OMP_EPP_IMAGE to the pushed forked EPP image}
config=deploy/epp/$arm.yaml
[ -f "$config" ] || { echo "no EPP arm config $config" >&2; exit 1; }
sha=$(sha256sum "$config" | cut -d' ' -f1)
rendered=$(sed -e "s|OMP_EPP_CONFIG_SHA256|$sha|" -e "s|OMP_EPP_CONFIG_NAME|$arm|g" \
               -e "s|OMP_EPP_IMAGE|$image|" deploy/k8s/omp-epp-router.yaml)
if grep -q "OMP_EPP_" <<<"$rendered"; then echo "unrendered placeholder" >&2; exit 1; fi
kubectl -n "$NS" create configmap omp-epp-config --from-file=deploy/epp/ \
  --dry-run=client -o yaml | kubectl -n "$NS" apply -f -
kubectl -n "$NS" apply -f - <<<"$rendered"
kubectl -n "$NS" rollout status deployment/omp-epp --timeout=300s
kubectl -n "$NS" rollout status deployment/omp-epp-router --timeout=120s
echo "EPP router serving $config ($sha)"
