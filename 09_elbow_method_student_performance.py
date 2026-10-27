# Practical 9: Elbow Method on Student Performance Dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("StudentsPerformance.csv")
X = df[["math score", "reading score", "writing score"]]
X = StandardScaler().fit_transform(X)

inertia_values = []
k_values = range(1, 11)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertia_values.append(model.inertia_)

plt.plot(list(k_values), inertia_values, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia / WCSS")
plt.title("Elbow Method")
plt.xticks(list(k_values))
plt.show()

print("Inertia values:", inertia_values)
print("Choose K near the bend (elbow) in the graph.")
