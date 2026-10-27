# Practical 7 - Version 2: K-Means on Mall Customers (K=5)
# Convert the Mall Customers PDF table to Mall_Customers.csv first.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaled_X = StandardScaler().fit_transform(X)
model = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(scaled_X)

print(df[["Annual Income (k$)", "Spending Score (1-100)", "Cluster"]].head())
print("\nCluster sizes:")
print(df["Cluster"].value_counts().sort_index())

plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"],
            c=df["Cluster"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Clusters")
plt.show()
