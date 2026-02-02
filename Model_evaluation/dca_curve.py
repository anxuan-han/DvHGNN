import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib import font_manager
import os

# 设置字体和样式
font_manager.fontManager.addfont('/scratchdata1/users/a1978372/Anxuan/arial.ttf')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# 文件路径
TEST_PREDICTIONS_FILE = "/scratchdata1/users/a1978372/Anxuan/framework/comparison_results/test/HGNN/DvHGNN_predictions.txt"
TEST_CLINICAL_FILE = "/scratchdata1/users/a1978372/Anxuan/framework/folder/SAH_filtered_recurrence_year_clin.csv"

OUTPUT_DIR = "/scratchdata1/users/a1978372/Anxuan/framework/comparison_results/dca_analysis"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prepare_data():
    """加载预测数据和临床数据"""
    print("="*80)
    print("加载数据...")
    print("="*80)
    
    # 加载预测数据
    df_pred = pd.read_csv(TEST_PREDICTIONS_FILE)
    print(f"\n模型预测数据: {len(df_pred)} 个患者")
    print(df_pred.head())
    
    # 加载临床数据
    df_clin = pd.read_csv(TEST_CLINICAL_FILE)
    print(f"\n临床数据: {len(df_clin)} 个患者")
    print(df_clin.head())
    
    # 合并数据
    df_merged = df_pred.merge(df_clin, on='patient_id', how='inner')
    print(f"\n合并后数据: {len(df_merged)} 个患者")
    
    return df_merged

def normalize_to_probability(values, y_true):
    """将连续变量通过logistic回归转换为概率"""
    X = values.reshape(-1, 1)
    y = y_true
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练logistic回归
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_scaled, y)
    
    # 获取预测概率
    probs = lr_model.predict_proba(X_scaled)[:, 1]
    
    return probs

def calculate_net_benefit(y_true, y_pred_prob, threshold):
    """
    计算净收益 (Net Benefit)
    
    NB(t) = TP/n - (FP/n) * (t/(1-t))
    其中 t 是阈值，TP和FP是在该阈值下的真正例和假正例
    """
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    
    n = len(y_true)
    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    
    # 避免除以零
    if threshold == 0 or threshold == 1:
        return 0
    
    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    
    return net_benefit

def plot_dca_single_features(y_true, model_probs, clinical_dict, save_path):
    """绘制单个临床指标vs模型预测的DCA"""
    
    # 计算阈值范围
    thresholds = np.linspace(0.01, 0.99, 100)
    
    # 计算模型的净收益
    model_nb = [calculate_net_benefit(y_true, model_probs, t) for t in thresholds]
    
    # 绘制
    fig, ax = plt.subplots(figsize=(2.8, 2.5))
    
    # 模型预测曲线 (突出显示)
    ax.plot(thresholds, model_nb, color='#f6796a', linewidth=1.5, 
            label='DvHGNN', alpha=0.9)
    
    # 单个临床指标曲线
    colors_clinical = ['#85ccbf', '#ffa366', '#95b8d1', "#dde41a"]
    
    for idx, (feature_name, feature_probs) in enumerate(clinical_dict.items()):
        feature_nb = [calculate_net_benefit(y_true, feature_probs, t) for t in thresholds]
        linestyle = '-' if 'Combined' in feature_name else '--'
        ax.plot(thresholds, feature_nb, color=colors_clinical[idx], linewidth=1.5, 
                label=feature_name, linestyle=linestyle, alpha=0.8)
    
    # 参考线：Treat All (正确的DCA计算: NB = sensitivity - (1-sensitivity) * t/(1-t))
    positive_rate = np.sum(y_true) / len(y_true)
    treat_all_nb = [positive_rate - (1 - positive_rate) * (t / (1 - t)) if t < 1 else 0 for t in thresholds]
    ax.plot(thresholds, treat_all_nb, color='navy', linewidth=1.5, 
            linestyle=':', label='Treat All', alpha=0.7)
    
    # 参考线：Treat None (始终为0)
    ax.axhline(y=0, color='gray', linewidth=1.5, linestyle=':', 
               label='Treat None', alpha=0.7)
    
    # 美化图表
    ax.set_xlabel('Threshold Probability', fontsize=12, fontweight='bold', color='black')
    ax.set_ylabel('Net Benefit', fontsize=12, fontweight='bold', color='black', labelpad=-5)
    ax.set_title('Decision Curve Analysis', 
                 fontsize=12, fontweight='bold', color='black', pad=5)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([min(np.min(model_nb), -0.05) - 0.05, 
                 max(np.max(treat_all_nb), np.max(model_nb)) + 0.05])
    
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    ax.tick_params(axis='both', colors='black', direction='out', labelsize=10, width=1, length=3)
    
    # 图例 - 参考ROC图格式
    ax.legend(
        loc='best',
        fontsize=7,
        frameon=True,
        shadow=True,
        edgecolor='black',
        facecolor='white',
        labelcolor='black',
        handlelength=1
    )
    
    # 手动控制边距 (left, bottom, right, top)
    plt.subplots_adjust(left=0.08, bottom=0.15, right=0.95, top=0.95)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nDCA图已保存: {save_path}")
    
    # 同时保存为PDF格式
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, format='pdf', bbox_inches='tight')
    print(f"DCA图已保存为PDF: {pdf_path}")
    
    plt.close()

def generate_dca_comparison_statistics(y_true, model_probs, clinical_dict, save_path):
    """生成DCA比较统计报告"""
    
    # 计算特定阈值下的净收益
    thresholds_of_interest = [0.1, 0.2, 0.26, 0.3, 0.5]
    
    with open(save_path, 'w') as f:
        f.write("Decision Curve Analysis - Model vs Clinical Features\n")
        f.write("="*100 + "\n\n")
        
        f.write("样本信息\n")
        f.write("-"*100 + "\n")
        f.write(f"总样本数: {len(y_true)}\n")
        f.write(f"复发患者数: {np.sum(y_true)}\n")
        f.write(f"非复发患者数: {len(y_true) - np.sum(y_true)}\n")
        f.write(f"复发率: {np.sum(y_true)/len(y_true)*100:.2f}%\n\n")
        
        f.write(f"模型预测概率统计\n")
        f.write("-"*100 + "\n")
        f.write(f"最小值: {np.min(model_probs):.4f}\n")
        f.write(f"最大值: {np.max(model_probs):.4f}\n")
        f.write(f"平均值: {np.mean(model_probs):.4f}\n")
        f.write(f"中位数: {np.median(model_probs):.4f}\n\n")
        
        f.write("各临床指标预测概率统计\n")
        f.write("-"*100 + "\n")
        for feature_name, feature_probs in clinical_dict.items():
            f.write(f"\n{feature_name}:\n")
            f.write(f"  最小值: {np.min(feature_probs):.4f}, 最大值: {np.max(feature_probs):.4f}\n")
            f.write(f"  平均值: {np.mean(feature_probs):.4f}, 中位数: {np.median(feature_probs):.4f}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("特定阈值下的净收益比较\n")
        f.write("="*100 + "\n\n")
        
        for threshold in thresholds_of_interest:
            f.write(f"\n阈值: {threshold:.2f}\n")
            f.write("-"*100 + "\n")
            f.write(f"{'模型/特征':<25} {'净收益':<15} {'排名':<10}\n")
            f.write("-"*100 + "\n")
            
            # 计算所有模型的净收益
            nb_dict = {}
            model_nb = calculate_net_benefit(y_true, model_probs, threshold)
            nb_dict['HGNN Model'] = model_nb
            
            for feature_name, feature_probs in clinical_dict.items():
                feature_nb = calculate_net_benefit(y_true, feature_probs, threshold)
                nb_dict[feature_name] = feature_nb
            
            # 排序
            sorted_nb = sorted(nb_dict.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (name, nb) in enumerate(sorted_nb, 1):
                f.write(f"{name:<25} {nb:<15.6f} {rank:<10}\n")
            
            # 确定最优
            best_name, best_nb = sorted_nb[0]
            f.write(f"\n最优: {best_name} (净收益: {best_nb:.6f})\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("结论\n")
        f.write("="*100 + "\n")
        f.write("净收益 (Net Benefit) 表示采用该预测指标相比不做任何治疗的额外临床收益\n")
        f.write("更高的净收益表示该指标具有更好的临床决策价值\n")
        f.write("排名靠前的指标应该在临床决策中优先考虑\n")

def main():
    print(f"\n{'*'*80}")
    print(f"* Decision Curve Analysis (DCA)")
    print(f"* 比较: HGNN模型 vs 单个临床指标 (Gleason Score, PSA, Age) vs 综合临床评分")
    print(f"{'*'*80}\n")
    
    # 加载数据
    df_merged = load_and_prepare_data()
    
    # 获取模型预测和真实标签
    model_probs = df_merged['predicted_probability'].values
    y_true = df_merged['recurrence'].values
    
    # 为单个临床指标生成概率
    print("\n" + "="*80)
    print("为单个临床指标生成预测概率...")
    print("="*80)
    
    clinical_features = {
        'Gleason Score': df_merged['gleason_score'].values,
        'PSA': df_merged['PSA'].values,
        'Age': df_merged['age'].values,
    }
    
    clinical_dict = {}
    for feature_name, feature_values in clinical_features.items():
        print(f"\n处理 {feature_name}...")
        # 填充缺失值
        feature_values_filled = np.nan_to_num(feature_values, nan=np.nanmean(feature_values))
        probs = normalize_to_probability(feature_values_filled, y_true)
        clinical_dict[feature_name] = probs
        print(f"  {feature_name} 概率范围: {np.min(probs):.4f} - {np.max(probs):.4f}")
    
    # 添加综合临床指标评分
    print(f"\n处理 综合临床指标评分...")
    
    # 提取所有临床特征值
    features_for_combined = []
    for feature_name, feature_values in clinical_features.items():
        feature_values_filled = np.nan_to_num(feature_values, nan=np.nanmean(feature_values))
        # 标准化每个特征
        scaler_temp = StandardScaler()
        feature_scaled = scaler_temp.fit_transform(feature_values_filled.reshape(-1, 1)).flatten()
        features_for_combined.append(feature_scaled)
    
    # 计算所有特征的标准化平均值
    combined_score = np.mean(features_for_combined, axis=0)
    
    # 将综合评分转换为概率
    combined_probs = normalize_to_probability(combined_score, y_true)
    clinical_dict['Combined Clinical Score'] = combined_probs
    print(f"  综合临床指标评分 概率范围: {np.min(combined_probs):.4f} - {np.max(combined_probs):.4f}")
    
    # 绘制DCA
    print("\n" + "="*80)
    print("绘制Decision Curve Analysis...")
    print("="*80)
    
    dca_path = os.path.join(OUTPUT_DIR, "decision_curve_analysis.png")
    plot_dca_single_features(y_true, model_probs, clinical_dict, dca_path)
    
    # 生成统计报告
    print("\n" + "="*80)
    print("生成统计报告...")
    print("="*80)
    
    stats_path = os.path.join(OUTPUT_DIR, "dca_statistics.txt")
    generate_dca_comparison_statistics(y_true, model_probs, clinical_dict, stats_path)
    
    # 保存合并数据
    output_csv = os.path.join(OUTPUT_DIR, "dca_data_with_probabilities.csv")
    df_output = df_merged.copy()
    for feature_name, probs in clinical_dict.items():
        df_output[f'{feature_name}_prob'] = probs
    df_output.to_csv(output_csv, index=False)
    print(f"\nDCA数据已保存: {output_csv}")
    
    # 总结
    print(f"\n{'='*80}")
    print("DCA分析完成！")
    print(f"{'='*80}")
    print(f"\n输出文件位置: {OUTPUT_DIR}")
    print("\n生成的文件:")
    print(f"  1. decision_curve_analysis.png - DCA曲线对比图")
    print(f"  2. dca_statistics.txt - 详细的阈值下净收益对比")
    print(f"  3. dca_data_with_probabilities.csv - 所有指标的概率")
    
    print(f"\n比较指标:")
    print(f"  • HGNN Model Prediction (红色，粗线)")
    print(f"  • Gleason Score (绿色，虚线)")
    print(f"  • PSA (橙色，虚线)")
    print(f"  • Age (蓝色，虚线)")
    print(f"  • Combined Clinical Score (黄色，实线 - 所有临床指标的综合评分)")

if __name__ == "__main__":
    main()