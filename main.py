import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import Scrollbar
import seaborn as sns

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