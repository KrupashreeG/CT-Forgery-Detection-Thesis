
#-------------------------------------------
#Results Visually 
#-------------------------------------------
import os, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "/content/gdrive/MyDrive/thesis_project"
LODO_FILE = f"{BASE}/results/lodo_v3_corrected/summary_full.json"
CATHOLD_FILE = f"{BASE}/results/category_holdout_v2_corrected/summary_full.json"
OUTPUT_DIR = f"{BASE}/results/final_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(LODO_FILE) as f:
    lodo = json.load(f)
with open(CATHOLD_FILE) as f:
    cathold = json.load(f)

sns.set_style("whitegrid")

lodo_folds = sorted(lodo["folds"], key=lambda r: -r["gap"])
lodo_names = [r["held_out"] for r in lodo_folds]

cathold_results = sorted(cathold["results"], key=lambda r: -r["gap"])
cathold_names = [r["held_out_attack"] for r in cathold_results]
cathold_cats = [r["category"] for r in cathold_results]
cathold_seen_auc = cathold["seen_metrics"]["auc"]
cathold_seen_fnr = cathold["seen_metrics"]["fnr"]


# ============================================================
# LODO GRAPHS
# ============================================================

# 1. LODO: Seen vs Zero-day AUC per fold
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(lodo_names))
width = 0.35
seen_aucs = [r["seen_auc"] for r in lodo_folds]
zd_aucs = [r["zeroday_auc"] for r in lodo_folds]
ax.bar(x - width/2, seen_aucs, width, label="Seen (10 trained attacks)", color="#2E7D32")
ax.bar(x + width/2, zd_aucs, width, label="Zero-day (held-out attack)", color="#C62828")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random chance")
ax.set_ylabel("AUC", fontsize=12)
ax.set_title("LODO: Seen vs Zero-Day AUC per Held-Out Attack", fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(lodo_names, rotation=30, ha='right', fontsize=10)
ax.set_ylim(0, 1.05); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_lodo_01_seen_vs_zeroday_auc.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: final_lodo_01_seen_vs_zeroday_auc.png")

# 2. LODO: Generalisation gap bar, sorted
fig, ax = plt.subplots(figsize=(11, 6))
gaps = [r["gap"] for r in lodo_folds]
colors = ["#C62828" if g > lodo["mean_gap"] else "#EF6C00" if g > 0 else "#1565C0" for g in gaps]
bars = ax.bar(lodo_names, gaps, color=colors)
ax.axhline(lodo["mean_gap"], color="black", linestyle="--", linewidth=1.5,
           label=f"Mean gap = {lodo['mean_gap']:.4f}")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_ylabel("Generalisation Gap (Seen AUC − Zero-day AUC)", fontsize=12)
ax.set_title("LODO: Generalisation Gap by Held-Out Attack", fontsize=13, fontweight='bold')
ax.set_xticklabels(lodo_names, rotation=30, ha='right', fontsize=10)
for bar, g in zip(bars, gaps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.005 if g>=0 else -0.015),
            f"{g:.4f}", ha='center', fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_lodo_02_gap_bar.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: final_lodo_02_gap_bar.png")

# 3. LODO: AUC / F1 / Recall comparison
fig, ax = plt.subplots(figsize=(13, 6))
width = 0.25
f1s = [r["zeroday_f1"] for r in lodo_folds]
recs = [r["zeroday_recall"] for r in lodo_folds]
ax.bar(x - width, zd_aucs, width, label="AUC", color="#1565C0")
ax.bar(x, f1s, width, label="F1 Score", color="#6A1B9A")
ax.bar(x + width, recs, width, label="Recall", color="#00838F")
ax.set_ylabel("Score", fontsize=12)
ax.set_title("LODO: Zero-Day AUC vs F1 vs Recall per Attack", fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(lodo_names, rotation=30, ha='right', fontsize=10)
ax.set_ylim(0, 1.05); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_lodo_03_auc_f1_recall.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: final_lodo_03_auc_f1_recall.png")

# 4. LODO: FNR per fold
fig, ax = plt.subplots(figsize=(11, 6))
fnrs = [r["zeroday_fnr"]*100 for r in lodo_folds]
bars = ax.bar(lodo_names, fnrs, color="#B71C1C")
avg_seen_fnr = np.mean([r["seen_fnr"] for r in lodo_folds])*100
ax.axhline(avg_seen_fnr, color="green", linestyle="--", linewidth=1.5,
           label=f"Avg seen-set FNR = {avg_seen_fnr:.2f}%")
ax.set_ylabel("False Negative Rate (%)", fontsize=12)
ax.set_title("LODO: Zero-Day False Negative Rate per Attack", fontsize=13, fontweight='bold')
ax.set_xticklabels(lodo_names, rotation=30, ha='right', fontsize=10)
for bar, f in zip(bars, fnrs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{f:.1f}%", ha='center', fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_lodo_04_fnr.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: final_lodo_04_fnr.png")

# 5. LODO: Confusion matrices grid (2 rows x 6 cols for 11 folds)
fig, axes = plt.subplots(2, 6, figsize=(24, 8))
axes = axes.flatten()
for i, r in enumerate(lodo_folds):
    cm = r["zeroday_confusion"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred:Auth', 'Pred:Tamp'], yticklabels=['True:Auth', 'True:Tamp'],
                ax=axes[i])
    axes[i].set_title(r["held_out"], fontsize=10, fontweight='bold')
for j in range(len(lodo_folds), len(axes)):
    axes[j].axis('off')
plt.suptitle("LODO: Confusion Matrices (Zero-Day)", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_lodo_05_confusion_matrices.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: final_lodo_05_confusion_matrices.png")


# ============================================================
# CATEGORY-HOLDOUT GRAPHS
# ============================================================

labels_ch = [f"{a}\n({c})" for a, c in zip(cathold_names, cathold_cats)]

# 6. Category-holdout: Seen vs Zero-day AUC
fig, ax = plt.subplots(figsize=(11, 6))
x2 = np.arange(len(labels_ch))
zd_aucs_ch = [r["zeroday_auc"] for r in cathold_results]
ax.bar(x2 - width/2, [cathold_seen_auc]*len(labels_ch), width, label="Seen (trained category)", color="#2E7D32")
ax.bar(x2 + width/2, zd_aucs_ch, width, label="Zero-day (unseen category)", color="#C62828")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random chance")
ax.set_ylabel("AUC", fontsize=12)
ax.set_title("Category-Holdout: Seen vs Zero-Day AUC", fontsize=13, fontweight='bold')
ax.set_xticks(x2); ax.set_xticklabels(labels_ch, fontsize=10)
ax.set_ylim(0, 1.05); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_catholdout_01_seen_vs_zeroday_auc.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: final_catholdout_01_seen_vs_zeroday_auc.png")

# 7. Category-holdout: Gap bar
fig, ax = plt.subplots(figsize=(10, 6))
gaps_ch = [r["gap"] for r in cathold_results]
colors_ch = ["#C62828" if g > cathold["mean_gap"] else "#EF6C00" for g in gaps_ch]
bars = ax.bar(labels_ch, gaps_ch, color=colors_ch)
ax.axhline(cathold["mean_gap"], color="black", linestyle="--", linewidth=1.5,
           label=f"Mean gap = {cathold['mean_gap']:.4f}")
ax.set_ylabel("Generalisation Gap", fontsize=12)
ax.set_title("Category-Holdout: Generalisation Gap by Category", fontsize=13, fontweight='bold')
ax.set_xticklabels(labels_ch, fontsize=10)
for bar, g in zip(bars, gaps_ch):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{g:.4f}", ha='center', fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_catholdout_02_gap_bar.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: final_catholdout_02_gap_bar.png")

# 8. Category-holdout: AUC/F1/Recall
fig, ax = plt.subplots(figsize=(11, 6))
f1s_ch = [r["zeroday_f1"] for r in cathold_results]
recs_ch = [r["zeroday_recall"] for r in cathold_results]
ax.bar(x2 - width, zd_aucs_ch, width, label="AUC", color="#1565C0")
ax.bar(x2, f1s_ch, width, label="F1 Score", color="#6A1B9A")
ax.bar(x2 + width, recs_ch, width, label="Recall", color="#00838F")
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Category-Holdout: AUC vs F1 vs Recall", fontsize=13, fontweight='bold')
ax.set_xticks(x2); ax.set_xticklabels(labels_ch, fontsize=10)
ax.set_ylim(0, 1.05); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_catholdout_03_auc_f1_recall.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: final_catholdout_03_auc_f1_recall.png")

# 9. Category-holdout: FNR
fig, ax = plt.subplots(figsize=(10, 6))
fnrs_ch = [r["zeroday_fnr"]*100 for r in cathold_results]
bars = ax.bar(labels_ch, fnrs_ch, color="#B71C1C")
ax.axhline(cathold_seen_fnr*100, color="green", linestyle="--", linewidth=1.5,
           label=f"Seen-set FNR = {cathold_seen_fnr*100:.2f}%")
ax.set_ylabel("False Negative Rate (%)", fontsize=12)
ax.set_title("Category-Holdout: Zero-Day FNR by Category", fontsize=13, fontweight='bold')
ax.set_xticklabels(labels_ch, fontsize=10)
for bar, f in zip(bars, fnrs_ch):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{f:.1f}%", ha='center', fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_catholdout_04_fnr.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: final_catholdout_04_fnr.png")

# 10. Category-holdout: Confusion matrices
fig, axes = plt.subplots(1, len(cathold_results), figsize=(4*len(cathold_results), 4))
for i, r in enumerate(cathold_results):
    cm = r["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred:Auth', 'Pred:Tamp'], yticklabels=['True:Auth', 'True:Tamp'],
                ax=axes[i])
    axes[i].set_title(f"{r['held_out_attack']}\n({r['category']})", fontsize=10, fontweight='bold')
plt.suptitle("Category-Holdout: Confusion Matrices", fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_catholdout_05_confusion_matrices.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: final_catholdout_05_confusion_matrices.png")


# ============================================================
# COMPARISON GRAPHS (LODO vs Category-Holdout)
# ============================================================

# Match the 5 categories present in BOTH experiments
lodo_gap_by_attack = {r["held_out"]: r["gap"] for r in lodo["folds"]}
common_attacks = [r["held_out_attack"] for r in cathold["results"]]
common_cats = {r["held_out_attack"]: r["category"] for r in cathold["results"]}
cathold_gap_by_attack = {r["held_out_attack"]: r["gap"] for r in cathold["results"]}

comp_attacks = sorted(common_attacks, key=lambda a: -cathold_gap_by_attack[a])
comp_labels = [f"{a}\n({common_cats[a]})" for a in comp_attacks]
comp_lodo = [lodo_gap_by_attack[a] for a in comp_attacks]
comp_cathold = [cathold_gap_by_attack[a] for a in comp_attacks]

# 11. Comparison: grouped bar
fig, ax = plt.subplots(figsize=(12, 7))
x3 = np.arange(len(comp_labels))
ax.bar(x3 - width/2, comp_lodo, width, label="LODO (partial exposure)", color="#1976D2")
ax.bar(x3 + width/2, comp_cathold, width, label="Category-holdout (true zero-day)", color="#D32F2F")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_ylabel("Generalisation Gap", fontsize=12)
ax.set_title("LODO vs Category-Holdout: Generalisation Gap Comparison", fontsize=14, fontweight='bold')
ax.set_xticks(x3); ax.set_xticklabels(comp_labels, fontsize=10)
for i, (l, c) in enumerate(zip(comp_lodo, comp_cathold)):
    ax.text(i - width/2, l + (0.01 if l>=0 else -0.02), f"{l:.3f}", ha='center', fontsize=8)
    ax.text(i + width/2, c + (0.01 if c>=0 else -0.02), f"{c:.3f}", ha='center', fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_comparison_01_gap_grouped_bar.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: final_comparison_01_gap_grouped_bar.png")

# 12. Comparison: mean gap overall (single summary bar)
fig, ax = plt.subplots(figsize=(7, 6))
means = [lodo["mean_gap"], cathold["mean_gap"]]
stds = [lodo["std_gap"], cathold["std_gap"]]
bars = ax.bar(["LODO\n(partial exposure)", "Category-holdout\n(true zero-day)"], means,
              yerr=stds, capsize=8, color=["#1976D2", "#D32F2F"])
ax.set_ylabel("Mean Generalisation Gap", fontsize=12)
ax.set_title("Overall Mean Gap: LODO vs Category-Holdout", fontsize=13, fontweight='bold')
for bar, m in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{m:.4f}", ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_comparison_02_mean_gap_summary.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: final_comparison_02_mean_gap_summary.png")

# 13. Comparison: increase per category (delta)
fig, ax = plt.subplots(figsize=(10, 6))
deltas = [c - l for l, c in zip(comp_lodo, comp_cathold)]
colors_delta = ["#D32F2F" if d > 0 else "#1976D2" for d in deltas]
bars = ax.bar(comp_labels, deltas, color=colors_delta)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Increase in Gap (Category-Holdout − LODO)", fontsize=12)
ax.set_title("How Much Worse Does True Zero-Day Get vs Partial Exposure?", fontsize=13, fontweight='bold')
ax.set_xticklabels(comp_labels, fontsize=10)
for bar, d in zip(bars, deltas):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(0.005 if d>=0 else -0.015), f"{d:+.4f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/final_comparison_03_delta.png", dpi=200, bbox_inches='tight')
plt.show()
print("Saved: final_comparison_03_delta.png")

# ============================================================
# CSV EXPORT — everything, ready for thesis tables
# ============================================================
import csv

with open(f"{OUTPUT_DIR}/final_lodo_full_table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Held_Out","Seen_AUC","Seen_CI_low","Seen_CI_high","Zeroday_AUC","Zeroday_CI_low",
                      "Zeroday_CI_high","Gap","Seen_F1","Zeroday_F1","Seen_Precision","Zeroday_Precision",
                      "Seen_Recall","Zeroday_Recall","Seen_FNR","Zeroday_FNR","Seen_FPR","Zeroday_FPR",
                      "TN","FP","FN","TP"])
    for r in lodo_folds:
        cm = r["zeroday_confusion"]
        writer.writerow([r["held_out"], f"{r['seen_auc']:.4f}", f"{r['seen_ci'][0]:.4f}", f"{r['seen_ci'][1]:.4f}",
                          f"{r['zeroday_auc']:.4f}", f"{r['zeroday_ci'][0]:.4f}", f"{r['zeroday_ci'][1]:.4f}",
                          f"{r['gap']:.4f}", f"{r['seen_f1']:.4f}", f"{r['zeroday_f1']:.4f}",
                          f"{r['seen_precision']:.4f}", f"{r['zeroday_precision']:.4f}",
                          f"{r['seen_recall']:.4f}", f"{r['zeroday_recall']:.4f}",
                          f"{r['seen_fnr']:.4f}", f"{r['zeroday_fnr']:.4f}",
                          f"{r['seen_fpr']:.4f}", f"{r['zeroday_fpr']:.4f}",
                          cm["tn"], cm["fp"], cm["fn"], cm["tp"]])
print("Saved: final_lodo_full_table.csv")

with open(f"{OUTPUT_DIR}/final_catholdout_full_table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Attack","Category","Zeroday_AUC","Zeroday_CI_low","Zeroday_CI_high","Gap",
                      "Zeroday_F1","Zeroday_Precision","Zeroday_Recall","Zeroday_FNR","Zeroday_FPR",
                      "TN","FP","FN","TP"])
    for r in cathold_results:
        cm = r["confusion_matrix"]
        writer.writerow([r["held_out_attack"], r["category"], f"{r['zeroday_auc']:.4f}",
                          f"{r['zeroday_ci'][0]:.4f}", f"{r['zeroday_ci'][1]:.4f}", f"{r['gap']:.4f}",
                          f"{r['zeroday_f1']:.4f}", f"{r['zeroday_precision']:.4f}", f"{r['zeroday_recall']:.4f}",
                          f"{r['zeroday_fnr']:.4f}", f"{r['zeroday_fpr']:.4f}",
                          cm["tn"], cm["fp"], cm["fn"], cm["tp"]])
print("Saved: final_catholdout_full_table.csv")

with open(f"{OUTPUT_DIR}/final_comparison_table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Attack","Category","LODO_Gap","Catholdout_Gap","Increase"])
    for a in comp_attacks:
        writer.writerow([a, common_cats[a], f"{lodo_gap_by_attack[a]:.4f}",
                          f"{cathold_gap_by_attack[a]:.4f}",
                          f"{cathold_gap_by_attack[a]-lodo_gap_by_attack[a]:+.4f}"])
print("Saved: final_comparison_table.csv")

print(f"\n{'='*60}")
print(f"ALL 13 GRAPHS + 3 CSV TABLES SAVED TO: {OUTPUT_DIR}")
print(f"{'='*60}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
