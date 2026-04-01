# Minikube deployment: KubeRay + Ray Jobs + Ray Serve + Grafana LGTM

This setup deploys:
- **Training** with `RayJob` (`trainer_API.py`)
- **Inference** with `RayService` + Ray Serve (`prediction_API.py`)
- **Observability** stack: **Loki** (logs), **Tempo** (traces), **Mimir** (metrics), **Grafana** (visualization)
- **Telemetry plumbing**: Promtail -> Loki, OTEL Collector -> Tempo, Prometheus -> Mimir

## Prerequisites

- Minikube installed
- `kubectl` and `kustomize` support (`kubectl apply -k`)
- Docker available

## 1) Start Minikube

Use enough resources for Ray + LGTM:

- CPUs: 6+
- RAM: 12GB+

## 2) Build image inside Minikube Docker

Use Minikube Docker environment and build image:

- image: `iomt-ray-ml:latest`

## 3) Install KubeRay operator

Install CRDs/operator before applying Ray manifests.

Recommended install:
- `kubectl create namespace kuberay-system`
- `kubectl apply -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.2.2"`

Wait until operator is ready:
- `kubectl get pods -n kuberay-system`

## 4) Deploy namespaces + observability + ray workloads

From repository root:

- `kubectl apply -k k8s`

Check rollout:
- `kubectl get pods -n observability`
- `kubectl get rayservices.ray.io -n iomt-ray`
- `kubectl get rayjobs.ray.io -n iomt-ray`

## 5) Trigger training job (if needed)

`RayJob` is created by default. You can re-run by deleting and re-applying:

- `kubectl delete rayjob iomt-trainer-job -n iomt-ray`
- `kubectl apply -f k8s/ray/rayjob-trainer.yaml`

Get logs:
- `kubectl logs -n iomt-ray job/iomt-trainer-job-submitter`

## 6) Access inference API and Grafana

Port-forward commands:

- Inference (Ray Serve):
  - `kubectl port-forward -n iomt-ray svc/iomt-inference-service-serve-svc 8000:8000`
- Grafana:
  - `kubectl port-forward -n observability svc/grafana 3000:3000`

Test endpoints:
- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/predict`

Grafana login:
- user: `admin`
- password: `admin`

## 7) Data sources in Grafana

Provisioned automatically:
- Loki: `http://loki.observability.svc.cluster.local:3100`
- Tempo: `http://tempo.observability.svc.cluster.local:3200`
- Mimir (Prometheus API): `http://mimir.observability.svc.cluster.local:9009/prometheus`

## Notes

- `prediction_API.py` and `trainer_API.py` now emit OTEL traces when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
- `prometheus.yaml` scrapes Ray head service metrics on port name `metrics` and pushes to Mimir.
- For production, replace default credentials, add persistence classes, and enable TLS/auth.
