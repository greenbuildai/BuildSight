import matplotlib.pyplot as plt
import numpy as np
import os

# Data for Detection Models (mAP@0.50)
conditions = ['S1 Normal', 'S2 Dusty', 'S3 Low-Light', 'S4 Crowded']
yolo11 = [0.7085, 0.8608, 0.7122, 0.6709]
yolo26 = [0.6941, 0.8475, 0.7069, 0.6470]
yolact = [0.5133, 0.7994, 0.5850, 0.4762]

# Data for SAMURAI Tracker (Temporal Consistency Score)
samurai = [-0.2628, -0.2234, -0.1840, -0.2628] # Negative scores, closer to 0 is better

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(conditions))
width = 0.25

# Plot 1: Detection Models Performance
rects1 = ax1.bar(x - width, yolo11, width, label='YOLOv11 (Recall-Optimized)', color='#2563eb')
rects2 = ax1.bar(x, yolo26, width, label='YOLOv26 (Precision-Optimized)', color='#ea580c')
rects3 = ax1.bar(x + width, yolact, width, label='YOLACT++ (Segmentation)', color='#16a34a')

ax1.set_ylabel('mAP@0.50 Score', fontsize=12, fontweight='bold')
ax1.set_title('Detection Models Performance by Site Condition', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(conditions, fontsize=11)
ax1.legend(loc='lower right', framealpha=0.9)
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Add values on top of bars
def autolabel_positive(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel_positive(rects1, ax1)
autolabel_positive(rects2, ax1)
autolabel_positive(rects3, ax1)

# Plot 2: SAMURAI Tracker Stability
bars = ax2.bar(conditions, samurai, color='#dc2626', width=0.4)
ax2.set_ylabel('Temporal Consistency Score (Deviation)', fontsize=12, fontweight='bold')
ax2.set_title('SAMURAI Temporal Tracking Score\n(Values closer to 0 are better)', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_ylim(-0.35, 0)
ax2.set_xticklabels(conditions, fontsize=11)

# Add values below bars
for rect in bars:
    height = rect.get_height()
    ax2.annotate(f'{height:.4f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, -15),
                textcoords="offset points",
                ha='center', va='top', fontsize=10, fontweight='bold', color='white', 
                bbox=dict(boxstyle='round,pad=0.2', fc='#dc2626', ec='none'))

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = r'e:\Company\Green Build AI\Prototypes\BuildSight\models_comparative_graph.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graph successfully saved to: {output_path}")
