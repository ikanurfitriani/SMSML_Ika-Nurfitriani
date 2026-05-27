# MIT License

# Copyright (c) 2026 Ika Nurfitriani

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import time
import sys
import pandas as pd
import requests

MLFLOW_URL = "http://localhost:8050/invocations"
EXPORTER_URL = "http://localhost:8000/track"
HEADERS = {"Content-Type": "application/json"}

try:
    df = pd.read_csv("credit-card-test_clean.csv") 
    
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    elif "id" in df.columns:
        df = df.drop(columns=["id"])
        
    if "default payment next month" in df.columns:
        df = df.drop(columns=["default payment next month"])
        
    print(f"Berhasil memuat dataset aktual. Ditemukan {len(df)} baris sampel data untuk pengujian.")
except Exception as e:
    print(f"Gagal memuat credit-card-test_clean.csv! Pastikan file berada di folder yang sama. Eror: {e}")
    sys.exit(1)

print("==================================================================")
print(" Memulai simulasi inferensi MLOps menggunakan DATASET AKTUAL      ")
print(" Seluruh metrik dikirim secara RIIL ke Prometheus Exporter        ")
print("==================================================================")

try:
    request_count = 0
    for index, row in df.iterrows():
        request_count += 1
        
        actual_features = row.tolist()
        
        payload = {
            "dataframe_records": [actual_features]
        }
        payload_string = json.dumps(payload)
        payload_bytes = len(payload_string.encode('utf-8'))
        
        metrics_to_report = {
            "success": False,
            "latency": 0.0,
            "payload_bytes": payload_bytes,
            "prediction": None
        }
        
        start_time = time.perf_counter()
        try:
            response = requests.post(MLFLOW_URL, data=payload_string, headers=HEADERS, timeout=5)
            end_time = time.perf_counter()
            actual_latency = end_time - start_time
            
            metrics_to_report["latency"] = actual_latency
            
            if response.status_code == 200:
                metrics_to_report["success"] = True
                
                response_data = response.json()
                if isinstance(response_data, dict) and "predictions" in response_data:
                    pred_val = response_data["predictions"][0]
                elif isinstance(response_data, list):
                    pred_val = response_data[0]
                else:
                    pred_val = int(response.text.strip("[] \n"))
                
                metrics_to_report["prediction"] = pred_val
                
                print(f"[{request_count}] Sukses | Latensi: {actual_latency:.4f}s | Ukuran: {payload_bytes} Bytes | Hasil Prediksi Model: {pred_val}")
            else:
                print(f"[{request_count}] Gagal HTTP {response.status_code} | Pesan: {response.text.strip()}")
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            end_time = time.perf_counter()
            metrics_to_report["latency"] = end_time - start_time
            print(f"[{request_count}] Gagal | Server MLflow (Port 8050) mati atau timeout.")
            
        try:
            requests.post(EXPORTER_URL, json=metrics_to_report, headers=HEADERS, timeout=2)
        except requests.exceptions.ConnectionError:
            print("⚠️ Peringatan: Gagal terhubung ke Prometheus Exporter (Port 8000).")

        time.sleep(1.0) 
        
except KeyboardInterrupt:
    print("\nSimulasi inferensi dihentikan.")