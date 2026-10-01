# Fylit RAG - Kubernetes Deployment Validation
#
# Validates the Kubernetes environment before application-level checks are run.
# The script stops early when kubectl is unavailable or the cluster cannot
# be reached.

$ErrorActionPreference = "Stop"

$Namespace = "fylit-rag"

# Displays Kubernetes troubleshooting information when a deployment
# validation check fails. This includes pod, deployment, service,
# endpoint, and recent event details to help identify the cause.
function Show-Diagnostics {
    param(
        [string]$Reason
    )

    Write-Host ""
    Write-Host "=========================================="
    Write-Host " Deployment Diagnostics"
    Write-Host "=========================================="

    Write-Host ""
    Write-Host "Failure reason: $Reason"

    Write-Host ""
    Write-Host "--- Pods ---"
    kubectl get pods -n $Namespace -o wide

    Write-Host ""
    Write-Host "--- Deployments ---"
    kubectl get deployments -n $Namespace

    Write-Host ""
    Write-Host "--- Services ---"
    kubectl get services -n $Namespace

    Write-Host ""
    Write-Host "--- Endpoints ---"
    kubectl get endpoints -n $Namespace

    Write-Host ""
    Write-Host "--- Recent Kubernetes Events ---"
    kubectl get events -n $Namespace --sort-by='.lastTimestamp' |
        Select-Object -Last 20

    Write-Host ""
    Write-Host "For detailed API logs run:"
    Write-Host "kubectl logs -n $Namespace -l app=fylit-rag-api --tail=100"
    Write-Host ""
}

Write-Host ""
Write-Host "=========================================="
Write-Host " Fylit RAG Kubernetes Deployment Validator"
Write-Host "=========================================="
Write-Host ""


# 1. Check kubectl

Write-Host "[1] Checking kubectl..."

if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    Write-Host "[FAIL] kubectl is not installed or not available in PATH."
    exit 1
}

Write-Host "[PASS] kubectl is available."



# 2. Check cluster connectivity

Write-Host ""
Write-Host "[2] Checking Kubernetes cluster connectivity..."

try {
    kubectl cluster-info *> $null

    if ($LASTEXITCODE -ne 0) {
        throw "kubectl could not connect to the cluster."
    }

    Write-Host "[PASS] Kubernetes cluster is reachable."
}
catch {
    Write-Host "[FAIL] Unable to connect to the Kubernetes cluster."
    exit 1
}



# 3. Check namespace


Write-Host ""
Write-Host "[3] Checking namespace '$Namespace'..."

kubectl get namespace $Namespace *> $null

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Namespace '$Namespace' was not found."
    exit 1
}

Write-Host "[PASS] Namespace '$Namespace' exists."


# 4. Check API deployment

Write-Host ""
Write-Host "[4] Checking API deployment..."

$Deployment = "fylit-rag-api"

kubectl get deployment $Deployment -n $Namespace *> $null

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Deployment '$Deployment' was not found."
    exit 1
}

Write-Host "[PASS] Deployment '$Deployment' exists."



# 5. Check deployment rollout

Write-Host ""
Write-Host "[5] Checking API deployment rollout..."

kubectl rollout status deployment/$Deployment `
    -n $Namespace `
    --timeout=30s

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Deployment rollout did not complete successfully."
    Show-Diagnostics "API deployment rollout failed."
    exit 1
}

Write-Host "[PASS] API deployment rollout is successful."



# 6. Check API replicas

Write-Host ""
Write-Host "[6] Checking API replicas..."

$DesiredReplicas = kubectl get deployment $Deployment `
    -n $Namespace `
    -o jsonpath='{.spec.replicas}'

$ReadyReplicas = kubectl get deployment $Deployment `
    -n $Namespace `
    -o jsonpath='{.status.readyReplicas}'

if (-not $ReadyReplicas) {
    $ReadyReplicas = 0
}

Write-Host "Desired replicas: $DesiredReplicas"
Write-Host "Ready replicas:   $ReadyReplicas"

if ([int]$ReadyReplicas -lt [int]$DesiredReplicas) {
    Write-Host "[FAIL] Not all API replicas are ready."
    Show-Diagnostics "API replicas are not fully ready."
    exit 1
}

Write-Host "[PASS] All API replicas are ready."



# 7. Check API pods


Write-Host ""
Write-Host "[7] Checking API pods..."

$ApiPods = kubectl get pods `
    -n $Namespace `
    -l app=fylit-rag-api `
    --no-headers

if (-not $ApiPods) {
    Write-Host "[FAIL] No API pods were found."
    exit 1
}

Write-Host $ApiPods
Write-Host "[PASS] API pods were found."


# 8. Check PostgreSQL resources


Write-Host ""
Write-Host "[8] Checking PostgreSQL..."

$PostgresPods = kubectl get pods `
    -n $Namespace `
    -l app=postgres `
    --no-headers

if (-not $PostgresPods) {
    Write-Host "[FAIL] No PostgreSQL pods were found."
    exit 1
}

Write-Host $PostgresPods

$PostgresReady = kubectl get pods `
    -n $Namespace `
    -l app=postgres `
    -o jsonpath='{.items[*].status.containerStatuses[*].ready}'

if (-not $PostgresReady -or $PostgresReady -contains "false") {
    Write-Host "[FAIL] PostgreSQL is not ready."
    Show-Diagnostics "PostgreSQL pod is not ready."
    exit 1
}

Write-Host "[PASS] PostgreSQL pod is ready."



# 9. Check API service

Write-Host ""
Write-Host "[9] Checking Kubernetes API service..."

$Service = "fylit-rag-api"

kubectl get service $Service -n $Namespace *> $null

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Service '$Service' was not found."
    exit 1
}

Write-Host "[PASS] Service '$Service' exists."



# 10. Check service endpoints

Write-Host ""
Write-Host "[10] Checking service endpoints..."

$Endpoints = kubectl get endpoints $Service `
    -n $Namespace `
    -o jsonpath='{.subsets[*].addresses[*].ip}'

if (-not $Endpoints) {
    Write-Host "[FAIL] Service '$Service' has no ready endpoints."
    Show-Diagnostics "API service has no ready endpoints."
    exit 1
}


Write-Host "Ready endpoint(s): $Endpoints"
Write-Host "[PASS] Service has ready endpoints."

# 11. Check /health endpoint

Write-Host ""
Write-Host "[11] Checking API /health endpoint..."

$HealthResult = kubectl exec `
    -n $Namespace `
    deploy/$Deployment `
    -- python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health', timeout=5).read().decode())"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] API /health endpoint could not be reached."
    Show-Diagnostics "API health check failed."
    exit 1
}

Write-Host "Response: $HealthResult"

if ($HealthResult -notmatch '"status"\s*:\s*"ok"') {
    Write-Host "[FAIL] API /health returned an unexpected response."
    Show-Diagnostics "API health endpoint returned an unexpected response."
    exit 1
}

Write-Host "[PASS] API /health reports healthy."


# 12. Check /ready endpoint

Write-Host ""
Write-Host "[12] Checking API /ready endpoint..."

$ReadyResult = kubectl exec `
    -n $Namespace `
    deploy/$Deployment `
    -- python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/ready', timeout=10).read().decode())"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] API /ready endpoint could not be reached."
    Show-Diagnostics "API readiness check failed."
    exit 1
}

Write-Host "Response: $ReadyResult"

if ($ReadyResult -notmatch '"status"\s*:\s*"ready"') {
    Write-Host "[FAIL] API /ready reports that the application is not ready."
    exit 1
}

Write-Host "[PASS] API /ready reports ready."


# Final  Validation summary


Write-Host ""
Write-Host "=========================================="
Write-Host " Kubernetes deployment validation PASSED"
Write-Host "=========================================="
Write-Host ""
Write-Host "Validated:"
Write-Host "  - kubectl availability"
Write-Host "  - cluster connectivity"
Write-Host "  - namespace"
Write-Host "  - API deployment"
Write-Host "  - deployment rollout"
Write-Host "  - API replicas"
Write-Host "  - API pods"
Write-Host "  - PostgreSQL readiness"
Write-Host "  - API service"
Write-Host "  - service endpoints"
Write-Host "  - /health"
Write-Host "  - /ready"
Write-Host ""

exit 0