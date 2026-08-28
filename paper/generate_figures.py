import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8.0,
    'axes.labelsize': 8.0,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 6.5,
    'text.usetex': False,
    'mathtext.fontset': 'dejavuserif',
})

os.makedirs('paper/figures', exist_ok=True)

# -------------------------------------------------------------
# Figure 1: Isolated Preprocessing Speedup
# -------------------------------------------------------------
resolutions = ['256x256\n(Anomaly)', '640x640\n(Detection)', '1280x720\n(HD Stream)']
eager = [36.9, 110.6, 281.6]
inductor = [8.2, 37.9, 81.8]
triton_1d = [8.2, 46.1, 90.1]
triton_fp16 = [7.2, 29.2, 58.4]

x = np.arange(len(resolutions))
width = 0.2

fig, ax = plt.subplots(figsize=(3.25, 1.95), dpi=300)
ax.bar(x - 1.5*width, eager, width, label='PyTorch Eager', color='#4A5568', edgecolor='black', linewidth=0.5)
ax.bar(x - 0.5*width, inductor, width, label='TorchInductor', color='#718096', edgecolor='black', linewidth=0.5)
ax.bar(x + 0.5*width, triton_1d, width, label='Triton FP32', color='#2B6CB0', edgecolor='black', linewidth=0.5)
ax.bar(x + 1.5*width, triton_fp16, width, label='Triton FP16', color='#319795', edgecolor='black', linewidth=0.5)

ax.set_ylabel('Latency (µs)')
ax.set_title('Isolated Preprocess Latency (Active 64MB L2 Eviction)', pad=6)
ax.set_xticks(x)
ax.set_xticklabels(resolutions)
ax.set_ylim(0, 320)
ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#CBD5E0')
ax.grid(axis='y', linestyle='--', alpha=0.5)

for i in range(len(resolutions)):
    sp = eager[i] / triton_1d[i]
    ax.annotate(f'{sp:.1f}x', xy=(x[i] + 0.5*width, triton_1d[i]),
                xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=6.8, fontweight='bold', color='#1A365D')

plt.savefig('paper/figures/fig_isolated_speedup.pdf', bbox_inches='tight', pad_inches=0.04)
plt.close()
print('[✓] Saved paper/figures/fig_isolated_speedup.pdf')

# -------------------------------------------------------------
# Figure 2: Stage-by-Stage Latency Breakdown (External Bottom Legend)
# -------------------------------------------------------------
labels = ['Eager', 'Triton']
h2d = np.array([30.7, 30.7])
prep = np.array([82.8, 62.5])
inf = np.array([534.5, 534.5])
post = np.array([240.0, 240.0])

fig, ax = plt.subplots(figsize=(3.25, 2.25), dpi=300)
x_pos = np.arange(len(labels))
bar_width = 0.36

ax.bar(x_pos, h2d, bar_width, label='01_H2D (DMA)', color='#CBD5E0', edgecolor='black', linewidth=0.5)
ax.bar(x_pos, prep, bar_width, bottom=h2d, label='02_Preprocess', color='#3182CE', edgecolor='black', linewidth=0.5)
ax.bar(x_pos, inf, bar_width, bottom=h2d + prep, label='03_Model Inf.', color='#4A5568', edgecolor='black', linewidth=0.5)
ax.bar(x_pos, post, bar_width, bottom=h2d + prep + inf, label='04_Postprocess', color='#ED8936', edgecolor='black', linewidth=0.5)

totals = h2d + prep + inf + post
for i in range(len(labels)):
    ax.text(x_pos[i], totals[i] + 25, f'{totals[i]:.1f} µs', ha='center', va='bottom', fontweight='bold', fontsize=7.5, color='#1A202C')

ax.set_ylabel('Time (µs)')
ax.set_title('Stage Breakdown & Amdahl Retention', pad=6)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1050)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=True, facecolor='white', framealpha=0.9, edgecolor='#CBD5E0')

plt.savefig('paper/figures/fig_stage_breakdown.pdf', bbox_inches='tight', pad_inches=0.04)
plt.close()
print(f'[✓] Saved paper/figures/fig_stage_breakdown.pdf (Eager: {totals[0]:.1f} µs, Triton: {totals[1]:.1f} µs)')

# -------------------------------------------------------------
# Figure 3: Warp Stall Breakdown
# -------------------------------------------------------------
stall_types = ['Mem Throttle', 'Long Scoreboard', 'Short Scoreboard', 'Executing', 'Other']
triton_1d_stalls = [34.2, 12.1, 18.4, 25.1, 10.2]
triton_2d_stalls = [6.4, 58.9, 12.0, 8.5, 14.2]

y = np.arange(len(stall_types))
height = 0.35

fig, ax = plt.subplots(figsize=(3.25, 1.95), dpi=300)
ax.barh(y + height/2, triton_1d_stalls, height, label='Triton 1D (47.6 µs)', color='#2B6CB0', edgecolor='black', linewidth=0.5)
ax.barh(y - height/2, triton_2d_stalls, height, label='Triton 2D (239.6 µs)', color='#E53E3E', edgecolor='black', linewidth=0.5)

ax.set_xlabel('Active Warp Cycles (%)')
ax.set_title('Warp Stall Comparison (sm_89)', pad=6)
ax.set_yticks(y)
ax.set_yticklabels(stall_types)
ax.set_xlim(0, 70)
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#CBD5E0')
ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.savefig('paper/figures/fig_stall_breakdown.pdf', bbox_inches='tight', pad_inches=0.04)
plt.close()
print('[✓] Saved paper/figures/fig_stall_breakdown.pdf')
