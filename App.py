# ============================================
# Student Performance Analyzer (Medical Lab Equipment)
# ============================================

# 👩‍🔬 Description:
# This program analyzes biomedical device readings taken by students in a lab.
# It compares student measurements with standard (true) biomedical values,
# calculates accuracy/error, and clusters students based on performance.

# 📦 Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================================
# 1️⃣ Generate Sample Data (You can replace with real lab data)
# ============================================

np.random.seed(42)

# Suppose we have readings from 3 different biomedical devices
num_students = 50
data = {
    'Student_ID': [f'ST_{i+1}' for i in range(num_students)],
    'Device_A_Reading': np.random.normal(100, 5, num_students),  # e.g., Blood Pressure Sensor
    'Device_B_Reading': np.random.normal(37, 0.5, num_students),  # e.g., Temperature Sensor
    'Device_C_Reading': np.random.normal(5.5, 0.3, num_students)  # e.g., pH Meter
}

df = pd.DataFrame(data)

# True standard (reference) values
true_values = {'Device_A_True': 100, 'Device_B_True': 37, 'Device_C_True': 5.5}

# ============================================
# 2️⃣ Compute Accuracy and Error Rates
# ============================================

for device in ['A', 'B', 'C']:
    df[f'Device_{device}_Error'] = abs(df[f'Device_{device}_Reading'] - true_values[f'Device_{device}_True'])
    df[f'Device_{device}_Accuracy'] = 100 - (df[f'Device_{device}_Error'] / true_values[f'Device_{device}_True'] * 100)

# Overall performance
df['Mean_Accuracy'] = df[[f'Device_{d}_Accuracy' for d in ['A', 'B', 'C']]].mean(axis=1)
df['Mean_Error'] = df[[f'Device_{d}_Error' for d in ['A', 'B', 'C']]].mean(axis=1)

# ============================================
# 3️⃣ Visual Analysis
# ============================================

sns.set(style="whitegrid")

# Plot accuracy distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Mean_Accuracy'], kde=True, bins=10, color='skyblue')
plt.title("Distribution of Student Accuracy (%)")
plt.xlabel("Accuracy (%)")
plt.ylabel("Count")
plt.show()

# ============================================
# 4️⃣ Clustering Students by Performance
# ============================================

X = df[['Mean_Accuracy', 'Mean_Error']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ============================================
# 5️⃣ Cluster Visualization
# ============================================

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x='Mean_Accuracy', y='Mean_Error',
    hue='Cluster', data=df, palette='deep', s=80
)
plt.title("Student Clustering Based on Performance")
plt.xlabel("Mean Accuracy (%)")
plt.ylabel("Mean Error")
plt.show()

# ============================================
# 6️⃣ Summary Report
# ============================================

cluster_summary = df.groupby('Cluster')[['Mean_Accuracy', 'Mean_Error']].mean().reset_index()
print("=== Cluster Summary ===")
print(cluster_summary)

print("\n=== Sample Student Performance Data ===")
print(df.head())

# Save results
df.to_csv("student_performance_analysis.csv", index=False)
print("\n✅ Report saved as 'student_performance_analysis.csv'")
