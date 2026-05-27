# SMSML_Ika-Nurfitriani
Default of Credit Card Clients Monitoring System is a comprehensive MLOps (Machine Learning Operations) engineering project that implements production-grade model serving, real-time system monitoring, logging, and automated alerting. This solution is designed to track a credit default classification model, ensuring system reliability, data integrity, and operational transparency in financial or banking environments.

## 🎯 Objective
The primary goals of this project are:
- To build and train an accurate machine learning classifier utilizing the Default of Credit Card Clients dataset from the UCI Machine Learning Repository.
- To serve the trained model as a production-ready API endpoint using MLflow Model Serving architecture.
- To establish a robust, real-time observability pipeline utilizing a custom Prometheus Exporter to track model predictions, data throughput, inference latencies, hardware resource consumption, and system errors.
- To create interactive monitoring visualization via Grafana Dashboards and configure automated alerting rules linked to communication channels (Email) for active incident management.

## 🛠 Methodology (MLOps Monitoring Pipeline)
This project utilizes a highly integrated MLOps pipeline to handle model deployment and live observability:
1. 🧪 Experimentation & Version Control (DagsHub Cloud)
    - Manual Experimentation: Performs detailed data exploratory analysis (EDA), target feature isolation, and automated hyperparameter search utilizing Scikit-Learn's `RandomizedSearchCV`.
    - Cloud Integration: Tracks model experiments, evaluation metrics, and logs visual validation artifacts (Confusion Matrix, ROC Curve, and Precision-Recall Display) remotely to the DagsHub Cloud platform via unified MLflow Manual Logging.
2. 🚀 Model Serving (MLflow & Docker Registry)
    - Architecture: Serves the optimized credit card default model through the MLflow serving local network framework mapped at port `8050` (`/invocations`).
    - Containerization: Packages the entire prediction stack ready for production deployment using Docker Hub registries for platform-independent replication.
3. 📊 Custom Metrics Collection (Prometheus Exporter)
    - Script Integration: A robust custom Prometheus Exporter (`prometheus_exporter.py`) built with Flask running at port `8000` to serve the `/metrics` endpoint.
    - Actual Data Metrics: Collects real, dynamic data feeds routed directly from the actual active inference loop instead of simulated inputs. Mapped metrics include counters for total requests, specific credit default outcomes (Default vs. Non-Default classes), HTTP errors, execution latency distributions, request payload byte lengths, and physical computer resource tracking (CPU & Memory footprints handled via `psutil`).
4. 📉 Real-Time Inference Simulation
    - Actual Inference Data: Runs a continuous prediction program (`inference.py`) parsing a pristine, separate testing matrix partition (`credit-card-test_clean.csv`) row-by-row to represent live incoming customer banking traffic.
    - Operational Feedback: Captures production performance metrics and reports runtime data payloads back into the collector gateway instantaneously.

## 📂 Directory Structure
```bash
MSML_Ika-Nurfitriani/
│
├── Membangun_Model/              
│   ├── default-credit-card_preprocessing/ 
│   │   ├── credit-card-test_clean.csv
│   │   ├── credit-card-train_clean.csv
│   │   └── credit-card_preprocessing.csv
│   ├── DagsHub.txt
│   ├── modelling.py                        
│   ├── modelling_tuning.py                
│   ├── requirements.txt                   
│   ├── screenshoot_artifak(dagshub).png            
│   ├── screenshoot_artifak(mlflow-lokal).jpg
│   ├── screenshoot_dashboard(dagshub).png            
│   └── screenshoot_dashboard(mlflow-lokal).jpg
│
├── Monitoring dan Logging/                       
│   ├── 1.bukti_serving/ 
│   │   ├── 1.bukti_serving(pull-images).png
│   │   ├── 2.bukti_serving(docker).png
│   │   ├── 3.bukti_serving(docker-images).png
│   │   └── 4.bukti_serving(mlflow).png
│   ├── 4.bukti monitoring Prometheus/ 
│   │   ├── 1.monitoring_model_prediction_requests_total.png
│   │   ├── 2.monitoring_model_prediction_default_total.png
│   │   ├── 3.monitoring_model_prediction_non_default_total.png
│   │   ├── 4.monitoring_model_prediction_errors_total.png
│   │   ├── 5.monitoring_model_serving_cpu_usage_percent.png
│   │   └── 6.monitoring_all-metrics.png
│   ├── 5.bukti monitoring Grafana/ 
│   │   ├── 1.monitoring_model_prediction_requests_total.png
│   │   ├── 2.monitoring_model_prediction_default_total.png
│   │   ├── 3.monitoring_model_prediction_non_default_total.png
│   │   ├── 4.monitoring_model_prediction_errors_total.png
│   │   ├── 5.monitoring_model_serving_cpu_usage_percent.png
│   │   ├── 6.monitoring_model_serving_memory_bytes.png
│   │   ├── 7.monitoring_model_active_version.png
│   │   ├── 8.monitoring_model_inference_latency_seconds_sum.png
│   │   ├── 9.monitoring_model_inference_latency_seconds_count.png
│   │   ├── 10.monitoring_model_request_payload_size_bytes_sum.png
│   │   ├── 11.monitoring_model_inference_latency_seconds_bucket.png
│   │   ├── 12.monitoring_model_request_payload_size_bytes_count.png
│   │   └── 13.monitoring_all-metrics.png
│   ├── 6.bukti alerting Grafana/ 
│   │   ├── 1.rules_model_prediction_errors_total.png
│   │   ├── 2.notifikasi_model_prediction_errors_total.png
│   │   ├── 3.rules_model_serving_cpu_usage_percent.png
│   │   ├── 4.notifikasi_model_serving_cpu_usage_percent.png
│   │   ├── 5.rules_model_inference_latency_seconds_sumcount.png
│   │   └── 6.notifikasi_model_inference_latency_seconds_sumcount.png
│   ├── 2.prometheus.yml                        
│   ├── 3.prometheus_exporter.py                
│   └── 7.inference.py          
│
├── Eksperimen_SML_Ika-Nurfitriani.txt                         
├── LICENSE.txt                          
├── Workflow-CI.txt                 
└── README.md                       
```

## 👩🏻‍💻 Author
[@Ika Nurfitriani](https://github.com/ikanurfitriani)