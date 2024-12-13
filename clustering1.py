import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, Scrollbar
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


#1
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from tkinter import Tk, Frame
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# Objective: To identify if there is a link between sales and scores
# Secondary Objective : Is it better to rely on the scores of the press or users?


# Load the CSV file
file_path = "VideoGameSales.csv"
df = pd.read_csv(file_path, encoding="latin1")


# Preprocessing
# Handle missing values
numeric_cols_with_na = ['Critic_Score', 'User_Score', 'Critic_Count', 'User_Count']
for col in numeric_cols_with_na:
    df[col] = df[col].fillna(df[col].mean())

categorical_cols_with_na = ['Developer', 'Rating']
for col in categorical_cols_with_na:
    df[col] = df[col].fillna('Unknown')

df['Year_of_Release'] = df['Year_of_Release'].fillna(df['Year_of_Release'].median())
df['Name'] = df['Name'].fillna('Unknown')

# Normalization of column
scaler = MinMaxScaler()
numeric_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales',
                'Critic_Score', 'User_Score', 'Critic_Count', 'User_Count']

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Encoding categorical columns
categorical_cols = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])







# 1 : Divide data in train and test
X_critics = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score']]
X_train, X_test = train_test_split(X_critics, test_size=0.3, random_state=42)



# 2 : Calculate inertia to optimize the number of clusters
inertias = []

print("\nCalculating inertia for different numbers of clusters :")
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)
    inertias.append(kmeans.inertia_)
    if n_clusters > 1:
        print(f"Numbre of clusters : {n_clusters}, Inertia : {kmeans.inertia_:.2f}")



# 3 : Visualize the elbow method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertias, marker='o', linestyle='-')
plt.title("The elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("Intra-cluster inertia")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


for n_clusters in range(3, 8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)
    train_silhouette = silhouette_score(X_train, kmeans.labels_)
    test_silhouette = silhouette_score(X_test, kmeans.predict(X_test))
    print(f"Number of clusters : {n_clusters}")
    print(f"Silhouette score (train) : {train_silhouette:.2f}")
    print(f"Silhouette score (test) : {test_silhouette:.2f}\n")



# 4 : Apply K-Means with a fixed number of clusters
n_clusters = 4
kmeans_users = KMeans(n_clusters=n_clusters, random_state=42)
train_clusters = kmeans_users.fit_predict(X_train)
test_clusters = kmeans_users.predict(X_test)










# 5 : Cluster visualization
pca_users = PCA(n_components=2)
reduced_features_train = pca_users.fit_transform(X_train)
reduced_features_test = pca_users.transform(X_test)

# Visualization on the training set
plt.figure(figsize=(8, 6))
plt.scatter(reduced_features_train[:, 0], reduced_features_train[:, 1], c=train_clusters, cmap='viridis', s=10)
plt.title("Sales & Review Press Clusters (train)")
plt.xlabel("Press Reviews")
plt.ylabel("Number of Sales")
plt.colorbar(label="Cluster")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Visualization on the test set
plt.figure(figsize=(8, 6))
plt.scatter(reduced_features_test[:, 0], reduced_features_test[:, 1], c=test_clusters, cmap='viridis', s=10)
plt.title("Sales & Review Press Clusters (test)")
plt.xlabel("Press Reviews")
plt.ylabel("Number of Sales")
plt.colorbar(label="Cluster")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()







# 1 : Divide data in train and test
X_users = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'User_Score']]
X_train, X_test = train_test_split(X_users, test_size=0.3, random_state=42)


# 2 : Calculate inertia to optimize the number of clusters
inertias = []

print("\nCalculating inertia for different numbers of clusters :")
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)
    inertias.append(kmeans.inertia_)
    if n_clusters > 1:
        print(f"Numbre of clusters : {n_clusters}, Inertia : {kmeans.inertia_:.2f}")

# 3 : Visualize the elbow method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertias, marker='o', linestyle='-')
plt.title("The elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("Intra-cluster inertia")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


for n_clusters in range(3, 8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)
    train_silhouette = silhouette_score(X_train, kmeans.labels_)
    test_silhouette = silhouette_score(X_test, kmeans.predict(X_test))
    print(f"Number of clusters : {n_clusters}")
    print(f"Silhouette score (train) : {train_silhouette:.2f}")
    print(f"Silhouette score (test) : {test_silhouette:.2f}\n")

# 4 : Apply K-Means with a fixed number of clusters
n_clusters = 4
kmeans_users = KMeans(n_clusters=n_clusters, random_state=42)
train_clusters = kmeans_users.fit_predict(X_train)
test_clusters = kmeans_users.predict(X_test)


# 5 : Cluster visualization
pca_users = PCA(n_components=2)
reduced_features_train = pca_users.fit_transform(X_train)
reduced_features_test = pca_users.transform(X_test)


# Visualization on the training set
plt.figure(figsize=(8, 6))
plt.scatter(reduced_features_train[:, 0], reduced_features_train[:, 1], c=train_clusters, cmap='viridis', s=10)
plt.title("Sales & Review Users Clusters (train)")
plt.xlabel("User Reviews")
plt.ylabel("Number of Sales")
plt.colorbar(label="Cluster")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Visualization on the test set
plt.figure(figsize=(8, 6))
plt.scatter(reduced_features_test[:, 0], reduced_features_test[:, 1], c=test_clusters, cmap='viridis', s=10)
plt.title("Sales & Review Users Clusters (test)")
plt.xlabel("User Reviews")
plt.ylabel("Number of Sales")
plt.colorbar(label="Cluster")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()



