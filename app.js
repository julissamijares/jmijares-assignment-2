document.getElementById('runKMeans').addEventListener('click', async () => {
    const response = await fetch('/api/cluster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: generatedData, k: 3, init_method: 'random' })
    });
    const result = await response.json();
    drawClusters(result.centroids, result.clusters);
});

function generateData() {
    // Code to generate random data points
}

function drawClusters(centroids, clusters) {
    // Code to visualize centroids and clusters on the canvas
}
