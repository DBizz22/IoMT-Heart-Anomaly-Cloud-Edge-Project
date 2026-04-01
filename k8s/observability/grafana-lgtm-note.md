# LGTM wiring summary

- **Logs**: Promtail DaemonSet tails pod logs and pushes to Loki.
- **Traces**: Apps send OTLP traces to OTEL Collector; collector exports to Tempo.
- **Metrics**: Prometheus scrapes Ray head metrics and remote_writes to Mimir.
- **Visualization**: Grafana has Loki, Tempo, and Mimir datasources pre-provisioned.
