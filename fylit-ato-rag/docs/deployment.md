# Fylit RAG Kubernetes Deployment

## Overview

The Fylit RAG API is deployed on Kubernetes with:

- 2 FastAPI replicas
- PostgreSQL with pgvector
- Kubernetes Service
- Health and readiness endpoints
- PowerShell deployment validation script

The ATO retrieval index is stored in PostgreSQL and is managed separately from the API image.

---

## Deploy / Update API

Build the API image:

```powershell
docker build -f docker/api.Dockerfile -t fylit-rag-api:local .
```

Restart the Kubernetes deployment:

```powershell
kubectl rollout restart deployment/fylit-rag-api -n fylit-rag
```

Check rollout status:

```powershell
kubectl rollout status deployment/fylit-rag-api -n fylit-rag
```

Check pods:

```powershell
kubectl get pods -n fylit-rag
```

Expected:

- 2 API pods: `1/1 Running`
- PostgreSQL: `1/1 Running`

---

## Validate Deployment

Run the PowerShell deployment validation script:

```powershell
deploy\k8s\validate-deployment.ps1
```

The script checks:

- Kubernetes cluster connectivity
- namespace
- API deployment and rollout
- API replicas and pods
- PostgreSQL readiness
- Kubernetes service and endpoints
- `/health`
- `/ready`

Successful validation ends with:

```text
Kubernetes deployment validation PASSED
```

---

## Access the API

Port-forward the Kubernetes service:

```powershell
kubectl port-forward -n fylit-rag service/fylit-rag-api 8080:80
```

Keep the terminal running while accessing the API.

Swagger UI:

```text
http://localhost:8080/docs
```

Check API health:

```powershell
curl http://localhost:8080/health
```

Check application readiness:

```powershell
curl http://localhost:8080/ready
```

---

## Useful Commands

Check Kubernetes pods:

```powershell
kubectl get pods -n fylit-rag
```

Check Kubernetes services:

```powershell
kubectl get svc -n fylit-rag
```

Check API logs:

```powershell
kubectl logs -n fylit-rag deployment/fylit-rag-api --tail=50
```

Check the PostgreSQL index:

```powershell
kubectl exec -n fylit-rag postgres-0 -- psql -U fylit -d fylit -c "SELECT count(*) AS chunks, count(embedding) AS embedded FROM chunks;"
```

---

## Important Note

Updating the API image does not automatically update the ATO retrieval index.

The API application and PostgreSQL retrieval index are updated separately.

After any API or index update, rerun the deployment validation script:

```powershell
deploy\k8s\validate-deployment.ps1
```