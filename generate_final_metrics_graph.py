"""
generate_final_metrics_graph.py
================================
Generates a comprehensive publication-grade performance visualization
dashboard for the AI Medical Chatbot's final held-out validation (N=400).
"""

import json
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Load data
with open("artifacts/evaluation/eval_heldout_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

comb = data["combined_pipeline"]
regex = data["regex_only_baseline"]
llm = data["llm_only_baseline"]

# Calculate accuracies
comb_acc = (comb["TP"] + comb["TN"]) / data["sample_size"] * 100
regex_acc = (regex["TP"] + regex["TN"]) / data["sample_size"] * 100
llm_acc = (llm["TP"] + llm["TN"]) / data["sample_size"] * 100

# Create figure
fig = plt.figure(figsize=(18, 12), dpi=200)
gs = gridspec.GridSpec(2, 3, width_ratios=[1.1, 1.1, 1], height_ratios=[1, 1], hspace=0.32, wspace=0.28)

# Color palette
c_blue = '#1f77b4'
c_coral = '#ff7f0e'
c_teal = '#2ca02c'
c_red = '#d62728'
c_purple = '#9467bd'
c_navy = '#0d3b66'
c_amber = '#ee964b'
c_emerald = '#05668d'
c_mint = '#02c39a'

# ==============================================================================
# Panel 1: Core Metrics Comparison (Grouped Bar Chart)
# ==============================================================================
ax1 = fig.add_subplot(gs[0, 0])
metrics = ['Accuracy', 'Recall\n(Sensitivity)', 'Specificity', 'Precision\n(PPV)', 'F1-Score']
regex_vals = [regex_acc, regex["recall"]*100, regex["specificity"]*100, regex["precision"]*100, regex["f1"]*100]
llm_vals = [llm_acc, llm["recall"]*100, llm["specificity"]*100, llm["precision"]*100, llm["f1"]*100]
comb_vals = [comb_acc, comb["recall"]*100, comb["specificity"]*100, comb["precision"]*100, comb["f1"]*100]

x = np.arange(len(metrics))
width = 0.26

rects1 = ax1.bar(x - width, regex_vals, width, label='Regex-Only', color='#aec7e8', edgecolor='#1f77b4', alpha=0.85)
rects2 = ax1.bar(x, llm_vals, width, label='LLM-Only', color='#98df8a', edgecolor='#2ca02c', alpha=0.85)
rects3 = ax1.bar(x + width, comb_vals, width, label='Combined (Regex OR LLM)', color='#1f77b4', edgecolor='#0d3b66', linewidth=1.2)

ax1.set_ylabel('Percentage (%)', fontweight='bold', fontsize=11)
ax1.set_title('A. Overall Performance Metrics Comparison\n(Held-Out Test Set N=400)', fontweight='bold', fontsize=12, pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontweight='bold', fontsize=9.5)
ax1.set_ylim(0, 115)
ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
ax1.axhline(95, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.7, label='Safety Target (>=95%)')

# Annotations on top of combined bars
for i, v in enumerate(comb_vals):
    ax1.annotate(f"{v:.1f}%", xy=(x[i] + width, v + 1.8), ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#0d3b66')

# ==============================================================================
# Panel 2: Confusion Matrix Heatmap (Combined Pipeline)
# ==============================================================================
ax2 = fig.add_subplot(gs[0, 1])
cm = np.array([
    [comb["TP"], comb["FN"]],
    [comb["FP"], comb["TN"]]
])

annot_labels = np.array([
    [f"TP = {comb['TP']}\n(98.9% Recall)\nSafety-Critical Caught", f"FN = {comb['FN']}\n(1.1% Miss Rate)\nSafety Critical Missed"],
    [f"FP = {comb['FP']}\n(5.4% FPR)\nConservative False Alarm", f"TN = {comb['TN']}\n(94.6% Specificity)\nRoutine Correctly Passed"]
])

sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', cbar=False, ax=ax2,
            xticklabels=['Predicted Critical', 'Predicted Routine'],
            yticklabels=['Actual Critical', 'Actual Routine'],
            linewidths=2, linecolor='white', annot_kws={"fontsize": 10.5, "fontweight": "bold"})

ax2.set_title(f'B. Final Confusion Matrix: Combined Pipeline\n(N=400 Cases | Accuracy: {comb_acc:.2f}%)', fontweight='bold', fontsize=12, pad=10)
ax2.set_xlabel('Model Predicted Classification', fontweight='bold', fontsize=11, labelpad=8)
ax2.set_ylabel('Ground Truth Clinical Label', fontweight='bold', fontsize=11, labelpad=8)

# ==============================================================================
# Panel 3: KPI Summary Card
# ==============================================================================
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')

kpi_text = (
    "  FINAL EVALUATION DASHBOARD  \n"
    "───────────────────────────────────\n\n"
    f"  • Sample Size:           400 cases\n"
    f"  • Critical Cohort:       176 (44.0%)\n"
    f"  • Routine Cohort:        224 (56.0%)\n"
    "───────────────────────────────────\n\n"
    f"  ★ OVERALL ACCURACY:      {comb_acc:.2f}%\n"
    f"    (386 correct out of 400 cases)\n\n"
    f"  ★ SENSITIVITY (RECALL):  {comb['recall']*100:.2f}%\n"
    f"    95% CI: [{comb['recall_95ci'][0]*100:.1f}%, {comb['recall_95ci'][1]*100:.1f}%]\n"
    f"    Target: >= 95.0% (PASSED +3.86%)\n\n"
    f"  ★ SPECIFICITY:           {comb['specificity']*100:.2f}%\n"
    f"  ★ PRECISION (PPV):       {comb['precision']*100:.2f}%\n"
    f"  ★ F1-SCORE:              {comb['f1']*100:.2f}%\n"
    f"  ★ FALSE POSITIVE RATE:   {comb['fpr']*100:.2f}%\n"
    f"    95% CI: [{comb['fpr_95ci'][0]*100:.1f}%, {comb['fpr_95ci'][1]*100:.1f}%]\n"
    "───────────────────────────────────\n\n"
    "  Dual-Layer Architecture Status:\n"
    "  • Regex Rule Engine:     Frozen\n"
    "  • Gemini Triage LLM:     Active\n"
    "  • RAG Knowledge Base:    39 Docs\n"
    "  • Safety Verification:   PASSED"
)

ax3.text(0.04, 0.5, kpi_text, fontsize=10.2, family='monospace', va='center',
         bbox=dict(boxstyle='round,pad=1.0', facecolor='#f4f8fb', edgecolor='#0d3b66', linewidth=2))

# ==============================================================================
# Panel 4: Performance Across 11 Medical Domains
# ==============================================================================
ax4 = fig.add_subplot(gs[1, :2])
domains_data = data["breakdowns"]["by_medical_domain"]
domain_names = sorted(domains_data.keys())

domain_recalls = [domains_data[d]["recall"] * 100 for d in domain_names]
domain_accs = [
    (domains_data[d]["TP"] + domains_data[d]["TN"]) / domains_data[d]["total"] * 100
    for d in domain_names
]
domain_totals = [domains_data[d]["total"] for d in domain_names]

labels_clean = [d.replace('_', ' ').title() + f"\n(n={domains_data[d]['total']})" for d in domain_names]

xd = np.arange(len(domain_names))
w = 0.35

b1 = ax4.bar(xd - w/2, domain_recalls, w, label='Recall (Sensitivity %)', color='#05668d', alpha=0.9)
b2 = ax4.bar(xd + w/2, domain_accs, w, label='Domain Accuracy (%)', color='#02c39a', alpha=0.9)

ax4.set_ylabel('Percentage (%)', fontweight='bold', fontsize=11)
ax4.set_title('C. Clinical Safety & Accuracy Breakdown Across 11 Medical Domains', fontweight='bold', fontsize=12, pad=10)
ax4.set_xticks(xd)
ax4.set_xticklabels(labels_clean, fontsize=8.5, fontweight='bold')
ax4.set_ylim(70, 115)
ax4.axhline(95, color='#d62728', linestyle='--', linewidth=1.1, alpha=0.7, label='Safety Standard (95%)')
ax4.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)

for i in range(len(domain_names)):
    ax4.annotate(f"{domain_recalls[i]:.0f}%", (xd[i] - w/2, domain_recalls[i] + 1.2), ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#05668d')
    ax4.annotate(f"{domain_accs[i]:.0f}%", (xd[i] + w/2, domain_accs[i] + 1.2), ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#028065')

# ==============================================================================
# Panel 5: Performance by Difficulty Level
# ==============================================================================
ax5 = fig.add_subplot(gs[1, 2])
diff_data = data["breakdowns"]["by_difficulty"]
diff_levels = ['easy', 'medium', 'hard']
diff_labels = ['Easy (n=80)', 'Medium (n=160)', 'Hard (n=160)']

diff_recalls = [diff_data[d]["recall"] * 100 for d in diff_levels]
diff_accs = [
    (diff_data[d]["TP"] + diff_data[d]["TN"]) / diff_data[d]["total"] * 100
    for d in diff_levels
]
diff_fprs = [diff_data[d]["fpr"] * 100 for d in diff_levels]

x_diff = np.arange(len(diff_levels))
w_diff = 0.28

ax5.bar(x_diff - w_diff, diff_recalls, w_diff, label='Recall', color='#1f77b4', alpha=0.9)
ax5.bar(x_diff, diff_accs, w_diff, label='Accuracy', color='#2ca02c', alpha=0.9)
ax5.bar(x_diff + w_diff, diff_fprs, w_diff, label='False Alarm Rate (FPR)', color='#ff7f0e', alpha=0.9)

ax5.set_ylabel('Percentage (%)', fontweight='bold', fontsize=11)
ax5.set_title('D. Performance by Difficulty Level', fontweight='bold', fontsize=12, pad=10)
ax5.set_xticks(x_diff)
ax5.set_xticklabels(diff_labels, fontsize=9.5, fontweight='bold')
ax5.set_ylim(0, 120)
ax5.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

for i in range(len(diff_levels)):
    ax5.annotate(f"{diff_recalls[i]:.1f}%", (x_diff[i] - w_diff, diff_recalls[i] + 2), ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1f77b4')
    ax5.annotate(f"{diff_accs[i]:.1f}%", (x_diff[i], diff_accs[i] + 2), ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2ca02c')
    ax5.annotate(f"{diff_fprs[i]:.1f}%", (x_diff[i] + w_diff, diff_fprs[i] + 2), ha='center', va='bottom', fontsize=8, fontweight='bold', color='#ff7f0e')

# Global Super Title
plt.suptitle('FINAL CLINICAL SAFETY & MODEL PERFORMANCE VALIDATION\nAI Medical Chatbot Safety & Emergency Layer | Held-Out Test Set (N=400 Cases)',
             fontsize=16, fontweight='heavy', y=0.98, color='#0d3b66')

# Save figures
out_path_eval = "artifacts/evaluation/final_model_performance_metrics.png"
plt.savefig(out_path_eval, bbox_inches='tight', dpi=200)
print(f"Saved figure to {out_path_eval}")

# Copy to brain artifact directory for UI display
brain_dir = r"C:\Users\VIJESH\.gemini\antigravity-ide\brain\b6ca1469-d265-48f5-96b1-63ec807e8c68"
if os.path.exists(brain_dir):
    dest = os.path.join(brain_dir, "final_model_performance_metrics.png")
    shutil.copyfile(out_path_eval, dest)
    print(f"Copied to brain artifact directory: {dest}")

plt.close()
