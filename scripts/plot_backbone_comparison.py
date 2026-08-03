# -*- coding: utf-8 -*-
"""
多视觉语言大模型 (Backbone Models) 性能对比报表绘制脚本
绘制柱状对比图与折线趋势图，并保存为 figures/backbone_comparison.png
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

# 确保 UTF-8 打印输出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_comparison():
    json_path = os.path.join("saved_models", "benchmark", "backbone_comparison_results.json")
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    names = [item["name"] for item in data]
    baseline_accs = [item["val_acc"] for item in data]
    innov_accs = [item["innov_acc"] for item in data]
    params = [item["params_m"] for item in data]
    mobile_friendly = [item["mobile_friendly"] for item in data]

    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6.5))

    rects1 = ax1.bar(x - width/2, baseline_accs, width, label='基线模型准确率 (%)', color='#94A3B8')
    rects2 = ax1.bar(x + width/2, innov_accs, width, label='融合三大创新准确率 (%)', color='#059669')

    ax1.set_ylabel('验证集准确率 (%)', fontsize=12, fontweight='bold', color='#0F172A')
    ax1.set_title('多视觉语言大模型 (Backbone Models) 农作物生育期诊断性能对比', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right', fontsize=10, fontweight='bold')
    ax1.set_ylim(75, 100)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # 标注数值
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#475569')

    for rect in rects2:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#047857')

    # 标注端侧可部署属性
    for i, is_mobile in enumerate(mobile_friendly):
        tag = "[端侧离线可导出]" if is_mobile else "[需云端 API]"
        color = '#047857' if is_mobile else '#DC2626'
        ax1.text(i, 77.5, tag, ha='center', va='center', fontsize=9, fontweight='bold', color=color,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8FAFC', edgecolor=color, linewidth=1))


    fig.tight_layout()

    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "backbone_comparison.png")
    plt.savefig(out_path, dpi=300)
    print(f"对比图表已成功绘制并保存至: {out_path}")

if __name__ == "__main__":
    plot_comparison()
