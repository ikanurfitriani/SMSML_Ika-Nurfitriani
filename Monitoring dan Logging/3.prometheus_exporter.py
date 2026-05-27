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

import time
import psutil
from flask import Flask, request, jsonify
from prometheus_client import make_wsgi_app, Counter, Gauge, Histogram, Summary
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)

REQUEST_TOTAL = Counter('model_prediction_requests_total', 'Total jumlah request inferensi')
PRED_DEFAULT_TOTAL = Counter('model_prediction_default_total', 'Total nasabah diprediksi gagal bayar')
PRED_NON_DEFAULT_TOTAL = Counter('model_prediction_non_default_total', 'Total nasabah diprediksi aman')
ERROR_TOTAL = Counter('model_prediction_errors_total', 'Total request yang gagal/error')

CPU_USAGE = Gauge('model_serving_cpu_usage_percent', 'Persentase penggunaan CPU server model')
MEMORY_USAGE = Gauge('model_serving_memory_bytes', 'Penggunaan memori dalam bytes')
MODEL_VERSION = Gauge('model_active_version', 'Versi model yang sedang aktif', ['version'])

LATENCY_HISTOGRAM = Histogram('model_inference_latency_seconds', 'Durasi inferensi dalam detik (Histogram)')
PAYLOAD_SIZE = Summary('model_request_payload_size_bytes', 'Ukuran data masuk dalam bytes')

MODEL_VERSION.labels(version='1.0.0').set(1)

@app.route('/track', methods=['POST'])
def track_metrics():
    data = request.json
    
    success = data.get('success', False)
    latency = data.get('latency', 0.0)
    payload_bytes = data.get('payload_bytes', 0)
    prediction = data.get('prediction', None)
    
    REQUEST_TOTAL.inc()
    
    if success:
        LATENCY_HISTOGRAM.observe(latency)
        PAYLOAD_SIZE.observe(payload_bytes)
        
        if prediction == 1:
            PRED_DEFAULT_TOTAL.inc()
        elif prediction == 0:
            PRED_NON_DEFAULT_TOTAL.inc()
    else:
        ERROR_TOTAL.inc()
        
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().used)
    
    return jsonify({"status": "metrics_updated"}), 200

app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

if __name__ == '__main__':
    print("Prometheus Custom Exporter (Flask) berjalan di http://localhost:8000/metrics")
    print("Menerima log aktivitas riil di http://localhost:8000/track")
    app.run(host='0.0.0.0', port=8000, debug=False)