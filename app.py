from flask import Flask, request, jsonify
import numpy as np
import time
import os
import psutil

# Prefer GPU via CuPy; fail fast if unavailable to ensure GPU-backed workload
try:
    import cupy as cp
except Exception as e:
    cp = None

app = Flask(__name__)

# Simulate a simple ML model
class SimpleMLModel:
    def __init__(self):
        # Simulate model loading time
        time.sleep(2)
        if cp is None:
            raise RuntimeError("CuPy not available. Ensure a CUDA-enabled CuPy wheel is installed and GPU is present.")
        # Initialize weights on GPU
        self.weights = cp.random.rand(100, 100, dtype=cp.float32)
        # Pre-allocate some buffers for heavier GPU work during predict
        self.heavy_a = cp.random.rand(4096, 4096, dtype=cp.float32)
        self.heavy_b = cp.random.rand(4096, 4096, dtype=cp.float32)

    def predict(self, data):
        # Copy input to GPU and perform dot product
        input_vec = cp.asarray(data, dtype=cp.float32)
        result = self.weights.dot(input_vec)
        # Additional GPU-heavy work to generate load
        tmp = self.heavy_a @ self.heavy_b
        # Force compute completion
        cp.cuda.Stream.null.synchronize()
        # Transfer a chunk back to host to exercise copy engines
        _ = cp.asnumpy(tmp[:256, :256])
        return cp.asnumpy(result).tolist()

model = SimpleMLModel()

@app.route('/health')
def health():
    payload = {"status": "healthy", "cpu_percent": psutil.cpu_percent()}
    try:
        if cp is not None:
            # Basic GPU memory info
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            payload.update({
                "gpu_memory_free_bytes": int(free_bytes),
                "gpu_memory_total_bytes": int(total_bytes)
            })
    except Exception:
        # If GPU query fails, keep health minimal
        pass
    return jsonify(payload)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('data', [])
        if len(data) != 100:
            return jsonify({"error": "Input data must have 100 features"}), 400

        # Make prediction on GPU
        prediction = model.predict(data)

        return jsonify({
            "prediction": prediction[:10],  # Return first 10 values
            "status": "success",
            "pod_name": os.environ.get('HOSTNAME', 'unknown')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/load')
def generate_load():
    """Endpoint to generate GPU load for testing autoscaling.

    Performs repeated large matrix multiplications on the GPU and host<->device
    transfers to exercise both SM and copy engines.
    """
    if cp is None:
        return jsonify({"error": "GPU not available (CuPy not installed)."}), 500

    duration = int(request.args.get('duration', 10))
    start_time = time.time()

    # Pre-allocate big arrays to avoid time spent in allocations
    a = cp.random.rand(4096, 4096, dtype=cp.float32)
    b = cp.random.rand(4096, 4096, dtype=cp.float32)
    host_buf = np.random.rand(4096, 4096).astype(np.float32)

    iters = 0
    while time.time() - start_time < duration:
        # Compute-heavy workload
        c = a @ b
        # Copy to host and back to stress copy engine
        h = cp.asnumpy(c)
        _ = cp.asarray(host_buf)
        cp.cuda.Stream.null.synchronize()
        iters += 1

    return jsonify({
        "message": f"Generated GPU load for {duration} seconds",
        "iters": iters,
        "pod_name": os.environ.get('HOSTNAME', 'unknown')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
