$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    throw "kubectl is not installed."
}

if (-not (Test-Path -LiteralPath ".env")) {
    throw ".env is missing. Copy .env.example and fill in FINNHUB_API_KEY, MONGODB_URI, and MONGODB_DATABASE."
}

kubectl create namespace mlops-repe --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic mlops-repe-env `
    --from-env-file=.env `
    --namespace mlops-repe `
    --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap prometheus-config `
    --from-file=prometheus.yml=monitoring/prometheus/prometheus.yml `
    --namespace mlops-repe `
    --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap grafana-datasource `
    --from-file=prometheus.yml=monitoring/grafana/provisioning/datasources/prometheus.yml `
    --namespace mlops-repe `
    --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap grafana-dashboard-provider `
    --from-file=backend.yml=monitoring/grafana/provisioning/dashboards/backend.yml `
    --namespace mlops-repe `
    --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap grafana-dashboards `
    --from-file=backend.json=monitoring/grafana/dashboards/backend.json `
    --namespace mlops-repe `
    --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -k k8s

function Convert-ToDockerDesktopPath([string]$windowsPath) {
    $full = [System.IO.Path]::GetFullPath($windowsPath)
    if ($full -match "^([A-Za-z]):\\(.*)$") {
        $drive = $Matches[1].ToLower()
        $rest = ($Matches[2] -replace "\\", "/")
        return "/run/desktop/mnt/host/$drive/$rest"
    }
    throw "Cannot map $full onto the Docker Desktop Kubernetes VM."
}

$models = Join-Path $repoRoot "models"
$outputs = Join-Path $repoRoot "outputs"
if ((Test-Path -LiteralPath $models) -and (Test-Path -LiteralPath $outputs)) {
    $patch = @(
        @{
            op = "replace"
            path = "/spec/template/spec/volumes/0"
            value = @{
                name = "models"
                hostPath = @{
                    path = (Convert-ToDockerDesktopPath $models)
                    type = "Directory"
                }
            }
        },
        @{
            op = "replace"
            path = "/spec/template/spec/volumes/1"
            value = @{
                name = "outputs"
                hostPath = @{
                    path = (Convert-ToDockerDesktopPath $outputs)
                    type = "Directory"
                }
            }
        }
    ) | ConvertTo-Json -Depth 6
    $patchFile = Join-Path $env:TEMP "mlops-repe-backend-volumes.json"
    Set-Content -LiteralPath $patchFile -Value $patch -Encoding ascii
    kubectl patch deployment backend --namespace mlops-repe --type json --patch-file $patchFile
}

kubectl rollout status deployment/backend --namespace mlops-repe --timeout 180s
kubectl rollout status deployment/frontend --namespace mlops-repe --timeout 180s
kubectl rollout status deployment/prometheus --namespace mlops-repe --timeout 180s
kubectl rollout status deployment/grafana --namespace mlops-repe --timeout 180s
kubectl get pods --namespace mlops-repe
