import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "city_day.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'AQI']

plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

for i, col in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col], color="#3498db", flierprops=dict(marker='o', markersize=4, markerfacecolor='#e74c3c'))
    plt.title(f"Box Plot: {col}", fontsize=11, fontweight='bold')
    plt.ylabel("")
    plt.tight_layout()

output_path = os.path.join(FIGURES_DIR, "outlier_boxplots.png")
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved outlier boxplots to: {output_path}")
