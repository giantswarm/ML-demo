## ML Example Demo (Flask) with KEDA Autoscaling

This is a small Flask app that simulates a simple ML inference workload and exposes endpoints for health, prediction, and generating artificial CPU load. A Kubernetes `Deployment` and `Service` are provided, along with a KEDA `ScaledObject` that creates an HPA to scale the app based on Prometheus (Mimir) metrics.

### Endpoints

- **GET /health**: basic health with GPU usage
- **POST /predict**: accepts JSON `{ "data": [100 floats] }`, returns a mock prediction
- **GET /load?duration=10**: generates GPU load for testing autoscaling

### Run locally

Requirements: Python 3.9+

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
# then in another shell
curl http://localhost:8080/health
```

Example predict request (array must have 100 numbers):

```bash
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"data": ['$(python - <<'PY'
print(','.join(['1']*100))
PY
  )']}'
```

Generate load locally (default: 10s):

```bash
curl "http://localhost:8080/load?duration=15"
```


### Deploy to Kubernetes

The manifests are in `deploy/manifests/`.

```bash
kubectl create namespace ml-workloads
kubectl apply -n ml-workloads -f deploy/manifests/deployment.yaml
kubectl apply -f deploy/manifests/keda.yaml
```

Port-forward to test:

```bash
kubectl -n ml-workloads port-forward svc/ml-demo 8080:80
curl http://localhost:8080/health
curl "http://localhost:8080/load?duration=30"
```

### Autoscaling with KEDA

- The KEDA `ScaledObject` (`deploy/manifests/keda.yaml`) targets the `ml-demo` `Deployment` and defines Prometheus queries against Mimir to drive scaling.
- `minReplicaCount` and `maxReplicaCount` bound the replicas; `advanced.horizontalPodAutoscalerConfig.behavior` tunes HPA scale up/down.
- The Mimir server URL and basic auth are configured via a `Secret` and `TriggerAuthentication` in the same file. Replace the default credentials and server URL for your environment before applying.

### Clean up

```bash
kubectl delete -n ml-workloads -f deploy/manifests/deployment.yaml
kubectl delete -f deploy/manifests/keda.yaml
kubectl delete namespace ml-workloads
```