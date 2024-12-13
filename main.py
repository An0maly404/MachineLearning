import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import Scrollbar
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tkinter import ttk, Scrollbar
from tkinter import Tk, Frame

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#aggromerative clustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the CSV file
file_path = "VideoGameSales.csv"  # Replace with your file path
df = pd.read_csv(file_path, encoding="latin1")



choice = 0




# For summary stats, such as the count, the mean, the standard deviation, and the minimum and maximum values.
if choice == 1:
    summary_stats = df.describe(include='all').transpose()  # Transpose for better readability
    root = tk.Tk()
    root.title("Summary Statistics")
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)
    vsb = Scrollbar(frame, orient="vertical")
    vsb.pack(side="right", fill="y")
    hsb = Scrollbar(frame, orient="horizontal")
    hsb.pack(side="bottom", fill="x")
    tree = ttk.Treeview(frame, columns=list(summary_stats.columns), show="headings", yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    vsb.config(command=tree.yview)
    hsb.config(command=tree.xview)
    tree["columns"] = ["Index"] + list(summary_stats.columns)
    tree.heading("Index", text="Index")
    tree.column("Index", anchor="center", width=100)
    for col in summary_stats.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=120)
    for index, row in summary_stats.iterrows():
        tree.insert("", "end", values=[index] + list(row))
    tree.pack(expand=True, fill="both")
    root.mainloop()


if choice == 2:
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ["Column", "Missing Values"]
    print("\nData types:")
    print(df.dtypes)
    missing_values["Data Type"] = df.dtypes.values
    root = tk.Tk()
    root.title("Missing Values and Data Types")
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)
    vsb = Scrollbar(frame, orient="vertical")
    vsb.pack(side="right", fill="y")
    tree = ttk.Treeview(frame, columns=["Column", "Missing Values", "Data Type"], show="headings", yscrollcommand=vsb.set)
    vsb.config(command=tree.yview)
    for col in missing_values.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=200)
    for index, row in missing_values.iterrows():
        tree.insert("", "end", values=list(row))
    tree.pack(expand=True, fill="both")
    root.mainloop()

#To have data that is useful for choice 3 and 4.
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
exclude_cols = ['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', "User_Count"]
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]


if choice == 3:
    if numeric_cols:
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True, bins=30, color='blue')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()



if choice == 4:
    if len(numeric_cols) >= 1:
        sns.pairplot(df[numeric_cols].dropna(), diag_kind='kde', corner=True)
        plt.suptitle("Pairplot of Numeric Features", y=1.02)
        plt.show()

        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()


if choice == 5:
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    unique_value_threshold = 50
    if not categorical_cols.empty:
        for col in categorical_cols:
            unique_values = df[col].nunique()
            if unique_values > unique_value_threshold:
                continue
            temp_col = df[col].fillna('Missing')
            plt.figure(figsize=(8, 4))
            sns.countplot(y=temp_col, order=temp_col.value_counts().index)
            plt.title(f'Distribution of {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
            plt.tight_layout()
            plt.show()



if choice == 6:
    
    # 1. Gérer les valeurs manquantes
    # Remplir les colonnes numériques avec la moyenne
    numeric_cols_with_na = ['Critic_Score', 'User_Score', 'Critic_Count', 'User_Count']
    for col in numeric_cols_with_na:
        df[col] = df[col].fillna(df[col].mean())

    # Remplir les colonnes catégoriques avec 'Unknown'
    categorical_cols_with_na = ['Developer', 'Rating']
    for col in categorical_cols_with_na:
        df[col] = df[col].fillna('Unknown')

    # Remplir Year_of_Release avec la médiane
    df['Year_of_Release'] = df['Year_of_Release'].fillna(df['Year_of_Release'].median())

    df['Name'] = df['Name'].fillna('Unknown')

    # 2. Normalisation des colonnes numériques
    scaler = MinMaxScaler()
    numeric_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales',
                'Critic_Score', 'User_Score', 'Critic_Count', 'User_Count']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 3. Encodage des colonnes catégoriques
    categorical_cols = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

   # Fonction pour afficher un DataFrame dans une fenêtre Tkinter
    def show_dataframe(dataframe, title):
        root = Tk()
        root.title(title)
        frame = Frame(root)
        frame.pack(fill="both", expand=True)

        # Scrollbars
        vsb = Scrollbar(frame, orient="vertical")
        vsb.pack(side="right", fill="y")
        hsb = Scrollbar(frame, orient="horizontal")
        hsb.pack(side="bottom", fill="x")

        # Table Treeview
        tree = ttk.Treeview(frame, columns=list(dataframe.columns), show="headings", yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)

        # Ajouter les colonnes
        for col in dataframe.columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=100)

        # Ajouter les lignes
        for index, row in dataframe.iterrows():
            tree.insert("", "end", values=list(row))

        tree.pack(expand=True, fill="both")
        root.mainloop()

    # Afficher les tableaux dans des fenêtres Tkinter
    show_dataframe(df.head(10), "Top 10 Rows")

    show_dataframe(df.describe().transpose(), "Summary Statistics")

    show_dataframe(df.isnull().sum().reset_index(name="Missing Values"), "Missing Values")

    categorical_data = df[['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']].head(10)
    show_dataframe(categorical_data, "Encoded Categorical Columns")



    
    
    #Etape 2 :
    X = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score', 'User_Score']]

    # Clustering 1 : Basé sur Ventes + Avis Utilisateurs
    X_users = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'User_Score']]

    kmeans_users = KMeans(n_clusters=3, random_state=42)
    df['Cluster_Users'] = kmeans_users.fit_predict(X_users)

    # Affichage des moyennes des colonnes numériques par cluster
    print("\nMoyennes par cluster (Ventes + Avis Utilisateurs) :")
    print(df[['NA_Sales', 'EU_Sales', 'User_Score', 'Cluster_Users']].groupby('Cluster_Users').mean())

    # Afficher des exemples de jeux par cluster avec les noms
    print("\nExemples pour Clustering basé sur Ventes + Avis Utilisateurs :")
    for cluster in df['Cluster_Users'].unique():
        print(f"\nExemples pour le cluster {cluster} :")
        print(df[df['Cluster_Users'] == cluster][['Name', 'NA_Sales', 'EU_Sales', 'User_Score']].head(10))

    # Visualisation pour Ventes + Avis Utilisateurs
    pca_users = PCA(n_components=2)
    reduced_features_users = pca_users.fit_transform(X_users)
    plt.scatter(reduced_features_users[:, 0], reduced_features_users[:, 1], c=df['Cluster_Users'], cmap='viridis', s=10)
    plt.title("Clusters pour Ventes et Avis Utilisateurs (K-Means)")
    plt.xlabel("Score des utilisateurs")
    plt.ylabel("Variations des ventes régionales")
    plt.colorbar(label="Cluster")
    plt.show()


    # Clustering 2 : Basé sur Ventes + Critiques
    X_critics = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score']]

    kmeans_critics = KMeans(n_clusters=3, random_state=42)
    df['Cluster_Critics'] = kmeans_critics.fit_predict(X_critics)

    # Affichage des moyennes des colonnes numériques par cluster
    print("\nMoyennes par cluster (Ventes + Critiques) :")
    print(df[['NA_Sales', 'EU_Sales', 'Critic_Score', 'Cluster_Critics']].groupby('Cluster_Critics').mean())

    # Afficher des exemples de jeux par cluster avec les noms
    print("\nExemples pour Clustering basé sur Ventes + Critiques :")
    for cluster in df['Cluster_Critics'].unique():
        print(f"\nExemples pour le cluster {cluster} :")
        print(df[df['Cluster_Critics'] == cluster][['Name', 'NA_Sales', 'EU_Sales', 'Critic_Score']].head(10))

    # Visualisation pour Ventes + Critiques
    pca_critics = PCA(n_components=2)
    reduced_features_critics = pca_critics.fit_transform(X_critics)
    plt.scatter(reduced_features_critics[:, 0], reduced_features_critics[:, 1], c=df['Cluster_Critics'], cmap='plasma', s=10)
    plt.title("Clusters pour Ventes et Critiques (K-Means)")
    plt.xlabel("Score des critique")
    plt.ylabel("Variations des ventes régionales")
    plt.colorbar(label="Cluster")
    plt.show()



    """

    # Pour Ventes + Avis Utilisateurs
    print("Loadings pour Ventes + Avis Utilisateurs :")
    print(pd.DataFrame(pca_users.components_, columns=X_users.columns, index=['Composante 1', 'Composante 2']))

    # Pour Ventes + Critiques
    print("Loadings pour Ventes + Critiques :")
    print(pd.DataFrame(pca_critics.components_, columns=X_critics.columns, index=['Composante 1', 'Composante 2']))



    # Afficher des exemples de jeux dans chaque cluster
    # Afficher des exemples de jeux dans chaque cluster pour Ventes + Critiques
    print("Exemples de Cluster 0 pour Ventes + Critiques :")
    print(df[df['Cluster_Critics'] == 0][['Name', 'NA_Sales', 'EU_Sales', 'Critic_Score']].head(10))

    print("Exemples de Cluster 1 pour Ventes + Critiques :")
    print(df[df['Cluster_Critics'] == 1][['Name', 'NA_Sales', 'EU_Sales', 'Critic_Score']].head(10))

    print("Exemples de Cluster 2 pour Ventes + Critiques :")
    print(df[df['Cluster_Critics'] == 2][['Name', 'NA_Sales', 'EU_Sales', 'Critic_Score']].head(10))

    # Afficher des exemples de jeux dans chaque cluster pour Ventes + Avis Utilisateurs
    print("Exemples de Cluster 0 pour Ventes + Avis Utilisateurs :")
    print(df[df['Cluster_Users'] == 0][['Name', 'NA_Sales', 'EU_Sales', 'User_Score']].head(10))

    print("Exemples de Cluster 1 pour Ventes + Avis Utilisateurs :")
    print(df[df['Cluster_Users'] == 1][['Name', 'NA_Sales', 'EU_Sales', 'User_Score']].head(10))

    print("Exemples de Cluster 2 pour Ventes + Avis Utilisateurs :")
    print(df[df['Cluster_Users'] == 2][['Name', 'NA_Sales', 'EU_Sales', 'User_Score']].head(10))


    """
    




    """"

    #Agglomerative Clustering

    # Créer une matrice de liaison pour le dendrogramme
    linkage_matrix = linkage(X, method='ward')  # Méthode 'ward' minimise la variance intra-cluster

    # Tracer le dendrogramme
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title("Dendrogramme - Clustering Hiérarchique")
    plt.xlabel("Points de données")
    plt.ylabel("Distance")
    plt.show()


    # Appliquer Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
    df['Cluster_Agglo'] = agglo.fit_predict(X)

    # Afficher les premiers résultats
    print(df[['Name', 'Cluster_Agglo']].head(10))




    # Réduction de dimension avec PCA pour visualisation
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X)

    # Visualiser les clusters
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=df['Cluster_Agglo'], cmap='plasma', s=50)
    plt.title("Clusters - Agglomerative Clustering")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.colorbar(label="Cluster")
    plt.show()

    # Afficher les clusters après correction
    print(df[['Name', 'Cluster_Agglo']].head(10))

    # Calculer le Silhouette Score
    score_agglo = silhouette_score(X, df['Cluster_Agglo'])
    print("Silhouette Score pour Agglomerative Clustering:", score_agglo)

    
    """