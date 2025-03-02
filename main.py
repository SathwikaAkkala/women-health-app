from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import json
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster

app = FastAPI()

# Load dataset (Ensure 'dataset.csv' exists in backend/)
@app.post("women_health_dataset.csv")
async def upload_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # Normalize data
    df_normalized = (df - df.min()) / (df.max() - df.min())

    # Hierarchical Clustering
    linkage_matrix = linkage(df_normalized, method="ward")
    df["Hierarchical_Cluster"] = fcluster(linkage_matrix, t=3, criterion="maxclust")

    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=3)
    df["GMM_Cluster"] = gmm.fit_predict(df_normalized)

    # Convert to JSON
    response = df.to_json(orient="records")

    return {"clusters": json.loads(response)}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

