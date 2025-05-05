# Deliveroo Data Analysis - Final Visualisation Assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set consistent font sizes for all plots to improve readability
sns.set_context("talk")

# Load the dataset containing restaurant delivery data from Deliveroo
# Ensure the file path is correct for local usage
df = pd.read_csv(r"C:\Data Handling and Visualization\deliveroo.csv")

# Initial data exploration to understand the structure and content of the dataset
print("Top 5 Records:")
print(df.head())

print("\nDataset Shape:")  # Displays number of rows and columns
print(df.shape)

print("\nData Types and Non-Null Counts:")  # Provides overview of each column's data type and missing values
df.info()

print("\nDescriptive Statistics:")  # Summary of statistics for numeric columns
print(df.describe())

print("\nNumber of Columns with Missing Values:")  # Count of columns with any missing data
print(df.isnull().any().sum())

print("\nNumber of Duplicate Rows:")  # Checks for exact duplicate rows
print(df.duplicated().sum())

# Clean and preprocess the dataset
# Additional Step: Remove unrealistic or extreme outlier values to improve visual clarity
# (Optional cutoff, can be adjusted as needed)
df = df[df['delivery_time'] < 120]  # Remove extremely long delivery times

# Convert review_rating and review_count to numeric, handling special entries like '500+'
df['review_rating'] = pd.to_numeric(df['review_rating'], errors='coerce')
df['review_count'] = df['review_count'].replace('500+', 500)
df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce')

# Remove rows with missing values in key columns required for analysis
df = df.dropna(subset=['delivery_fee', 'delivery_time', 'review_rating', 'review_count', 'distance'])

# Create categorical bins for rating and distance to group data meaningfully
df['rating_bin'] = pd.cut(df['review_rating'], bins=[0, 3, 3.5, 4, 4.5, 5], labels=['<3', '3–3.5', '3.5–4', '4–4.5', '>4.5'])
df['distance_bin'] = pd.cut(df['distance'], bins=[0, 1, 2, 3, 5, 10], labels=['<1km', '1–2km', '2–3km', '3–5km', '>5km'])

# Plot 1: Boxplot showing delivery time distribution across rating brackets
plt.figure(figsize=(12, 7))
sns.boxplot(data=df, x='rating_bin', y='delivery_time', palette='coolwarm')
plt.title('Delivery Time Distribution by Review Rating Brackets', fontsize=24)
plt.xlabel('Review Rating Range', fontsize=20)
plt.ylabel('Delivery Time (minutes)', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# Add annotation to highlight outliers that may affect service experience
plt.annotate('Outliers may indicate occasional delays', xy=(2, 60), xytext=(0.5, 70),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=16)
plt.tight_layout()
plt.show()

# Plot 2: Correlation heatmap showing relationships between numeric features
# Additional: Sort the correlation matrix to group strongly correlated features
numerical_cols = ['delivery_fee', 'delivery_time', 'review_rating', 'review_count', 'distance']
corr_matrix = df[numerical_cols].corr().sort_values(by='delivery_time', ascending=False)
plt.figure(figsize=(10, 7))
numerical_cols = ['delivery_fee', 'delivery_time', 'review_rating', 'review_count', 'distance']
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu', fmt=".2f", linewidths=0.5, annot_kws={"size": 12})
plt.title('Correlation Matrix of Key Variables', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# Plot 3: Bubble plot - visualizes delivery time vs distance, with color = rating and size = review count
plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(
    data=df,
    x='distance',
    y='delivery_time',
    hue='review_rating',
    size='review_count',
    sizes=(40, 400),
    palette='viridis',
    alpha=0.7,
    edgecolor='gray',
    legend='full'
)
plt.title('Delivery Time vs Distance (Bubble Size = Review Count)', fontsize=24)
plt.xlabel('Distance (km)', fontsize=20)
plt.ylabel('Delivery Time (minutes)', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# Highlight observation of cluster formation
plt.annotate('Clusters at moderate distances and times', xy=(1.5, 40), xytext=(0.5, 60),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=16)

# Customize the legend to reduce clutter and improve readability
handles, labels = plt.gca().get_legend_handles_labels()
from collections import OrderedDict
filtered = list(OrderedDict.fromkeys(zip(labels, handles)))[:10]  # Limit to 10 entries
filtered_labels, filtered_handles = zip(*filtered)
plt.legend(filtered_handles, filtered_labels,
           title='Rating & Review Size', fontsize=16, title_fontsize=16,
           loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, ncol=1)
plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.show()

# Generate a summary statistics table grouped by delivery category (e.g. cities or zones)
summary = df.groupby('searched_category').agg({
    'delivery_fee': ['mean', 'median'],
    'delivery_time': ['mean', 'median'],
    'review_rating': ['mean'],
    'review_count': ['mean']
}).round(2).reset_index()

# Rename columns to be more descriptive
summary.columns = ['Category', 'Avg Fee', 'Median Fee', 'Avg Time', 'Median Time', 'Avg Rating', 'Avg Review Count']

# Additional Insight: Sort summary table by Avg Rating to highlight top-rated categories
summary = summary.sort_values(by='Avg Rating', ascending=False)

# Export the summary table for reporting
summary.to_csv('deliveroo_summary_table.csv', index=False)
