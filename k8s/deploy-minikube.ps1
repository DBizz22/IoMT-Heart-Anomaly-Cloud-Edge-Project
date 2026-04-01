$ErrorActionPreference = 'Stop'

Write-Host "[1/6] Starting Minikube (if not already running)..."
minikube status | Out-Null
if ($LASTEXITCODE -ne 0) {
  minikube start --cpus=6 --memory=12288
}

Write-Host "[2/6] Building image inside Minikube Docker daemon..."
minikube -p minikube docker-env --shell powershell | Invoke-Expression
docker build -t iomt-ray-ml:latest .

Write-Host "[3/6] Installing KubeRay operator..."
kubectl get namespace kuberay-system 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
  kubectl create namespace kuberay-system
}
kubectl apply -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.2.2"

Write-Host "[4/6] Waiting for KubeRay operator..."
kubectl wait --for=condition=Available deployment/kuberay-operator -n kuberay-system --timeout=180s

Write-Host "[5/6] Deploying observability + ray workloads..."
kubectl apply -k k8s

Write-Host "[6/6] Current status summary"
kubectl get pods -n observability
kubectl get rayjobs.ray.io -n iomt-ray
kubectl get rayservices.ray.io -n iomt-ray

Write-Host "Done."
Write-Host "Use: kubectl port-forward -n observability svc/grafana 3000:3000"
Write-Host "Use: kubectl port-forward -n iomt-ray svc/iomt-inference-service-serve-svc 8000:8000"
