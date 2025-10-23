from flask import Flask, request, jsonify
import numpy as np
import time
import os
import psutil

app = Flask(__name__)

# Simulate a simple ML model
class SimpleMLModel:
    def __init__(self):
        # Simulate model loading time
        time.sleep(2)
        self.weights = np.random.rand(100, 100)
    
    def predict(self, data):
        # Simulate CPU-intensive ML inference
        result = np.dot(self.weights, data)
        # Add some CPU load for testing autoscaling
        for i in range(1000000):
            _ = i ** 2
        return result.tolist()

model = SimpleMLModel()

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "cpu_percent": psutil.cpu_percent()})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('data', [])
        if len(data) != 100:
            return jsonify({"error": "Input data must have 100 features"}), 400
        
        # Convert to numpy array
        input_data = np.array(data)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return jsonify({
            "prediction": prediction[:10],  # Return first 10 values
            "status": "success",
            "pod_name": os.environ.get('HOSTNAME', 'unknown')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/load')
def generate_load():
    """Endpoint to generate CPU load for testing HPA"""
    duration = int(request.args.get('duration', 10))
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Generate CPU load
        for i in range(100000):
            _ = i ** 2
    
    return jsonify({
        "message": f"Generated load for {duration} seconds",
        "pod_name": os.environ.get('HOSTNAME', 'unknown')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
