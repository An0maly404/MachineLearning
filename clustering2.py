import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage




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




# Divide data in train and test
X_critics = df[['Critic_Score', 'Global_Sales']]
X_train, X_test = train_test_split(X_critics, test_size=0.3, random_state=42)

# Linkage
linked = linkage(X_train, method='ward')

# Dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=12.)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.axhline(y=100, color='r', linestyle='--')
plt.show()

# Silhouette Score for different cluster counts
print("Calculating Silhouette Scores for different cluster counts (Train Data - Press Critics):")
for n_clusters in range(3, 7):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    clusters = clustering.fit_predict(X_train)
    score = silhouette_score(X_train, clusters)
    print(f"Number of Clusters: {n_clusters}, Silhouette Score: {score:.2f}")

# Apply Agglomerative Clustering with 3 clusters
n_clusters = 3
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
clusters = clustering.fit_predict(X_train)

# Create a DataFrame for results
train_results = X_train.copy()
train_results['Cluster'] = clusters

# Visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X_train)
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='rainbow', alpha=0.7)
plt.title('Clusters Visualization (PCA 2D)')
plt.xlabel('Global Sales')
plt.ylabel('Critic Score')
plt.show()





# B. Test Part for Press Critics

# Linkage for Test Data
linked_test = linkage(X_test, method='ward')

# Dendrogram for Test Data
plt.figure(figsize=(12, 8))
dendrogram(linked_test, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=12.)
plt.title('Dendrogram for Agglomerative Clustering (Test Data - Press Critics)')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.axhline(y=100, color='r', linestyle='--')
plt.show()

# Silhouette Score for different cluster counts (Test Data)
print("Calculating Silhouette Scores for different cluster counts (Test Data - Press Critics):")
for n_clusters in range(3, 7):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    clusters_test = clustering.fit_predict(X_test)
    score_test = silhouette_score(X_test, clusters_test)
    print(f"Number of Clusters: {n_clusters}, Silhouette Score: {score_test:.2f}")

# Apply Agglomerative Clustering with 3 clusters (Test Data)
n_clusters = 3
clustering_test = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
clusters_test = clustering_test.fit_predict(X_test)

# Create a DataFrame for test results
press_test_results = X_test.copy()
press_test_results['Cluster'] = clusters_test

# Visualization of Test Data
pca_test = PCA(n_components=2)
reduced_data_test = pca_test.fit_transform(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data_test[:, 0], reduced_data_test[:, 1], c=clusters_test, cmap='rainbow', alpha=0.7)
plt.title('Clusters Visualization (Test Data - Press Critics, PCA 2D)')
plt.xlabel('Global Sales')
plt.ylabel('Critic Score')
plt.show()






#2 User critics
# A Test
X_critics = df[['User_Score', 'Global_Sales']]
X_train, X_test = train_test_split(X_critics, test_size=0.3, random_state=42)



# Linkage
linked = linkage(X_train, method='ward')

# Dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=12.)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.axhline(y=100, color='r', linestyle='--')
plt.show()

# Silhouette Score for different cluster counts
print("Calculating Silhouette Scores for different cluster counts (Train Data - Users Critics):")
for n_clusters in range(3, 7):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    clusters = clustering.fit_predict(X_train)
    score = silhouette_score(X_train, clusters)
    print(f"Number of Clusters: {n_clusters}, Silhouette Score: {score:.2f}")

# Apply Agglomerative Clustering with 3 clusters
n_clusters = 3
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
clusters = clustering.fit_predict(X_train)

# Create a DataFrame for results
train_results = X_train.copy()
train_results['Cluster'] = clusters

# Visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X_train)
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='rainbow', alpha=0.7)
plt.title('Clusters Visualization (PCA 2D)')
plt.xlabel('Global Sales')
plt.ylabel('Critic Score')
plt.show()





# B. Test Part for Press Critics

# Linkage for Test Data
linked_test = linkage(X_test, method='ward')

# Dendrogram for Test Data
plt.figure(figsize=(12, 8))
dendrogram(linked_test, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=12.)
plt.title('Dendrogram for Agglomerative Clustering (Test Data - Press Critics)')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.axhline(y=100, color='r', linestyle='--')
plt.show()

# Silhouette Score for different cluster counts (Test Data)
print("Calculating Silhouette Scores for different cluster counts (Test Data - User Critics):")
for n_clusters in range(3, 7):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    clusters_test = clustering.fit_predict(X_test)
    score_test = silhouette_score(X_test, clusters_test)
    print(f"Number of Clusters: {n_clusters}, Silhouette Score: {score_test:.2f}")

# Apply Agglomerative Clustering with 3 clusters (Test Data)
n_clusters = 3
clustering_test = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
clusters_test = clustering_test.fit_predict(X_test)

# Create a DataFrame for test results
press_test_results = X_test.copy()
press_test_results['Cluster'] = clusters_test

# Visualization of Test Data
pca_test = PCA(n_components=2)
reduced_data_test = pca_test.fit_transform(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data_test[:, 0], reduced_data_test[:, 1], c=clusters_test, cmap='rainbow', alpha=0.7)
plt.title('Clusters Visualization (Test Data - Press Critics, PCA 2D)')
plt.xlabel('Global Sales')
plt.ylabel('User Score')
plt.show()