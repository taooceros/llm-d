{{/* Keep the original Helm naming semantics for existing releases. */}}
{{- define "vllm-tpu.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end }}

{{/* Preserve the kuberay app label required by the GKE TPU webhook. */}}
{{- define "vllm-tpu.labels" -}}
app.kubernetes.io/name: kuberay
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
llm-d.ai/accelerator-vendor: google
llm-d.ai/accelerator-variant: tpu
{{- end }}

{{- define "vllm-tpu.acceleratorLabel" -}}
{{- $accelerators := dict "v5e" "tpu-v5-lite-podslice" "v5p" "tpu-v5p-slice" "v6e" "tpu-v6e-slice" "v7" "tpu-v7-slice" -}}
{{- if not (hasKey $accelerators .Values.tpu.generation) -}}
{{- fail "tpu.generation must be v5e, v5p, v6e, or v7" -}}
{{- end -}}
{{- get $accelerators .Values.tpu.generation -}}
{{- end }}

{{- define "vllm-tpu.validate" -}}
{{- if not (regexMatch "^[1-9][0-9]*x[1-9][0-9]*(x[1-9][0-9]*)?$" .Values.tpu.topology) -}}
{{- fail "tpu.topology must contain two or three positive dimensions (for example 4x8)" -}}
{{- end -}}
{{- range $name := list "nodeCount" "chipsPerPod" -}}
{{- if not (regexMatch "^[1-9][0-9]*$" (get $.Values.tpu $name | toString)) -}}
{{- fail (printf "tpu.%s must be a positive integer" $name) -}}
{{- end -}}
{{- end -}}
{{- $chips := 1 -}}
{{- range splitList "x" .Values.tpu.topology -}}
{{- $chips = mul $chips (int .) -}}
{{- end -}}
{{- if ne $chips (mul .Values.tpu.nodeCount .Values.tpu.chipsPerPod) -}}
{{- fail "tpu.topology chip product must equal tpu.nodeCount * tpu.chipsPerPod" -}}
{{- end -}}
{{- if not (has .Values.storage.type (list "none" "gcsFuse")) -}}
{{- fail "storage.type must be none or gcsFuse" -}}
{{- end -}}
{{- if eq .Values.storage.type "gcsFuse" -}}
{{- if not (regexMatch "^[a-z0-9][a-z0-9._-]+[a-z0-9]$" .Values.storage.gcsBucket) -}}
{{- fail "storage.gcsBucket must be a bucket name, not an empty value, URI, or placeholder" -}}
{{- end -}}
{{- if not (hasPrefix "/" .Values.storage.gcsMountPath) -}}
{{- fail "storage.gcsMountPath must be an absolute path" -}}
{{- end -}}
{{- end -}}
{{- end }}
