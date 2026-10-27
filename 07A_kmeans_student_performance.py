# Practical 7 - Version 1: K-Means on Student Performance (K=5)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("StudentsPerformance.csv")
X = df[["math score", "reading score", "writing score"]]

scaled_X = StandardScaler().fit_transform(X)
model = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(scaled_X)

print(df[["math score", "reading score", "writing score", "Cluster"]].head())
print("\nStudents in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# PCA helps display the three score columns in a 2D graph.
points = PCA(n_components=2).fit_transform(scaled_X)
plt.scatter(points[:, 0], points[:, 1], c=df["Cluster"])
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Student Clusters (K=5)")
plt.show()
