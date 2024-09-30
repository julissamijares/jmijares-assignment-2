from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/step', methods=['POST'])
def step():
    data = request.json['data']
    centroids = request.json['centroids']
    n_clusters = int(request.json['n_clusters'])  # Convert to integer

    # Convert the data to numpy array for KMeans processing
    data_array = np.array(data)

    # Perform KMeans step
    kmeans = KMeans(n_clusters=n_clusters, init=np.array(centroids), n_init=1, max_iter=1)
    kmeans.fit(data_array)

    new_centroids = kmeans.cluster_centers_
    labels = kmeans.labels_.tolist()

    return jsonify({
        'centroids': new_centroids.tolist(),
        'labels': labels
    })

if __name__ == '__main__':
    app.run(debug=True)
