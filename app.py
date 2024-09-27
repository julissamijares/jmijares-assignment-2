from flask import Flask, request, jsonify
import numpy as np
from kmeans import KMeans

app = Flask(__name__)

@app.route('/api/cluster', methods=['POST'])
def cluster():
    data = np.array(request.json['data'])
    k = request.json['k']
    init_method = request.json['init_method']

    kmeans = KMeans(k=k, init=init_method)
    centroids, clusters = kmeans.fit(data)

    return jsonify({'centroids': centroids.tolist(), 'clusters': clusters.tolist()})

if __name__ == "__main__":
    app.run(port=3000)
