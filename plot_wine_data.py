import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
import seaborn as sns

# Load your exact dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['cultivar'] = wine.target

print("Dataset loaded!")
print(f"Shape: {df.shape}")
print("Classes:", wine.target_names)

# ----------------------------
# 1. SINGLE FEATURE HISTOGRAM
# ----------------------------
plt.figure(figsize=(12, 8))

# Plot alcohol distribution for each class
plt.subplot(2, 2, 1)
for class_num in [0, 1, 2]:
    class_data = df[df['cultivar'] == class_num]['alcohol']
    plt.hist(class_data, alpha=0.7, label=wine.target_names[class_num], bins=15)

plt.title('Alcohol Distribution by Wine Cultivar')
plt.xlabel('Alcohol (%)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# ----------------------------
# 2. MULTIPLE FEATURE HISTOGRAMS (Subplots)
# ----------------------------
features = ['alcohol', 'malic_acid', 'flavanoids', 'proline']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, feature in enumerate(features):
    row, col = i // 2, i % 2
    for class_num in [0, 1, 2]:
        class_data = df[df['cultivar'] == class_num][feature]
        axes[row, col].hist(class_data, alpha=0.6, 
                           label=wine.target_names[class_num], bins=12)
    axes[row, col].set_title(f'{feature.title()} Distribution')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------
# 3. OVERLAYED HISTOGRAMS (Best for comparison)
# ----------------------------
plt.figure(figsize=(12, 8))

# Proline (best separator)
plt.subplot(1, 2, 1)
for class_num in [0, 1, 2]:
    class_data = df[df['cultivar'] == class_num]['proline']
    plt.hist(class_data, alpha=0.7, bins=20, 
             label=f'{wine.target_names[class_num]} (n={len(class_data)})',
             density=True)  # Normalize to density

plt.title('Proline Distribution by Cultivar (Normalized)')
plt.xlabel('Proline (mg/L)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Flavanoids
plt.subplot(1, 2, 2)
for class_num in [0, 1, 2]:
    class_data = df[df['cultivar'] == class_num]['flavanoids']
    plt.hist(class_data, alpha=0.7, bins=20, 
             label=f'{wine.target_names[class_num]} (n={len(class_data)})',
             density=True)

plt.title('Flavanoids Distribution by Cultivar (Normalized)')
plt.xlabel('Flavanoids (g/L)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------
# 4. SAVE PLOTS (for your report)
# ----------------------------
plt.figure(figsize=(10, 6))
for class_num in [0, 1, 2]:
    class_data = df[df['cultivar'] == class_num]['proline']
    plt.hist(class_data, alpha=0.7, bins=20, 
             label=f'{wine.target_names[class_num]} (n={len(class_data)})',
             density=True)

plt.title('Proline Distribution by Wine Cultivar\n(Your Model Input Feature)', fontsize=14, fontweight='bold')
plt.xlabel('Proline (mg/L)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save for your report
plt.savefig('wine_proline_histogram.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Histogram saved as 'wine_proline_histogram.png'")
