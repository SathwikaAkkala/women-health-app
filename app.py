from flask import Flask, request, jsonify
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

app = Flask(__name__)

@app.route('/cluster', methods=['POST'])
def cluster_data():
    data = request.json['data']
    df = pd.DataFrame(data)
    
    # Apply Hierarchical Clustering
    hierarchical_model = AgglomerativeClustering(n_clusters=3)
    df['cluster_h'] = hierarchical_model.fit_predict(df)

    # Apply Gaussian Mixture Model
    gmm = GaussianMixture(n_components=3)
    df['cluster_gmm'] = gmm.fit_predict(df)

    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
