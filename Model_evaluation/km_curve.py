import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import font_manager
import os

# 设置字体和样式
font_manager.fontManager.addfont('/scratchdata1/users/a1978372/Anxuan/arial.ttf')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# 文件路径 - 验证集
VALIDATION_PREDICTIONS_FILE = "/scratchdata1/users/a1978372/Anxuan/framework/comparison_results/validation/HGNN/HGNN_predictions.txt"
VALIDATION_SURVIVAL_FILE = "/scratchdata1/users/a1978372/Anxuan/framework/folder/rec_clean_2_final.csv"
VALIDATION_THRESHOLD = 0.8

# 文件路径 - 测试集
TEST_PREDICTIONS_FILE = "/scratchdata1/users/a1978372/Anxuan/framework/comparison_results/test/HGNN/HGNN_predictions.txt"
TEST_SURVIVAL_FILE = "/scratchdata1/users/a1978372/Anxuan/framework/folder/SAH_filtered_recurrence_year.csv"
TEST_THRESHOLD = 0.26

OUTPUT_DIR = "/scratchdata1/users/a1978372/Anxuan/framework/comparison_results/survival_analysis"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prepare_data(pred_file, surv_file, threshold, dataset_name):
    """加载预测数据和生存数据"""
    print("="*80)
    print(f"加载{dataset_name}数据...")
    print("="*80)
    
    # 加载预测数据
    df_pred = pd.read_csv(pred_file)
    print(f"\n{dataset_name}预测数据: {len(df_pred)} 个患者")
    print(df_pred.head())
    
    # 加载生存数据
    df_survival = pd.read_csv(surv_file)
    print(f"\n{dataset_name}生存数据: {len(df_survival)} 个患者")
    print(df_survival.head())
    
    # 合并数据
    df_merged = df_pred.merge(df_survival, on='patient_id', how='inner')
    print(f"\n{dataset_name}合并后数据: {len(df_merged)} 个患者")
    
    # 根据阈值创建分组
    df_merged['model_group'] = (df_merged['predicted_probability'] >= threshold).astype(int)
    df_merged['model_group_label'] = df_merged['model_group'].map({
        0: f'Low Risk',
        1: f'High Risk)'
    })
    
    # 创建真实标签分组
    df_merged['true_group_label'] = df_merged['true_label'].map({
        0: 'Non-Recurrence',
        1: 'Recurrence'
    })
    
    print(f"\n按模型分组:")
    print(df_merged['model_group_label'].value_counts())
    
    print(f"\n按真实标签分组:")
    print(df_merged['true_group_label'].value_counts())
    
    return df_merged

def plot_survival_curve(df_data, duration_col, event_col, group_col, group_labels, 
                       save_path, title, colors=None, dataset_name=""):
    """绘制Kaplan-Meier生存曲线"""
    if colors is None:
        colors = ['#fd83e2', '#944c83']
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    
    kmf = KaplanMeierFitter()
    
    groups = sorted(df_data[group_col].unique())
    
    for idx, group in enumerate(groups):
        mask = df_data[group_col] == group
        T = df_data[mask][duration_col].values
        E = df_data[mask][event_col].values
        
        kmf.fit(T, E, label=group_labels.get(group, str(group)))
        kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=1.2, 
                                   color=colors[idx % len(colors)])
    
    # 执行logrank检验
    mask_0 = df_data[group_col] == groups[0]
    mask_1 = df_data[group_col] == groups[1]
    
    T_0 = df_data[mask_0][duration_col].values
    E_0 = df_data[mask_0][event_col].values
    T_1 = df_data[mask_1][duration_col].values
    E_1 = df_data[mask_1][event_col].values
    
    results = logrank_test(T_0, T_1, E_0, E_1)
    p_value = results.p_value
    
    # 美化图表
    ax.set_xlabel('Time (years)', fontsize=10, fontweight='bold', color='black')
    ax.set_ylabel('Recurrence-Free Probability', fontsize=10, fontweight='bold', color='black')
    #ax.set_title(title, fontsize=10, fontweight='bold', color='black', pad=0)
    
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    ax.tick_params(axis='both', colors='black', direction='out', labelsize=10, width=1, length=3)
    
    # 图例 - 使用坐标自定义位置 (bbox_to_anchor: (x, y) 范围 0-1)
    # 调整这里的坐标 (0.02, 0.98) 来改变legend位置
    ax.legend(loc='upper left', bbox_to_anchor=(0.45, 0.25), fontsize=8, frameon=False, shadow=False,
            edgecolor='black', facecolor='white', labelcolor='black')
    
    # 使用tight_layout自动调整避免重叠
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    
    # 添加p值文本 - 放在左上角避免与图例重叠
    if p_value < 0.001:
        p_text = "p < 0.001***"
    elif p_value < 0.01:
        p_text = f"p = {p_value:.4f}**"
    elif p_value < 0.05:
        p_text = f"p = {p_value:.4f}*"
    else:
        p_text = f"p = {p_value:.4f}"
    
    ax.text(
        0.7, 0.95, p_text,
        transform=ax.transAxes,
        fontsize=8,
        fontweight='bold',
        color='black',
        horizontalalignment='left',
        verticalalignment='top'
    )

    # 使用tight_layout + adjust_subplots避免重叠
    plt.tight_layout()
    # 同时保存PNG和PDF两个版本
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"\n{dataset_name}生存曲线已保存: {save_path}")
    print(f"{dataset_name}矢量图已保存: {pdf_path}")
    print(f"{dataset_name} Log-rank检验 p值: {p_value:.6f}")
    plt.close()
    
    return p_value

def generate_survival_statistics(df_data, duration_col, event_col, group_col, group_labels, save_path, dataset_name=""):
    """生成生存统计信息"""
    kmf = KaplanMeierFitter()
    
    groups = sorted(df_data[group_col].unique())
    
    with open(save_path, 'w') as f:
        f.write(f"{dataset_name} Kaplan-Meier Survival Analysis Statistics\n")
        f.write("="*80 + "\n\n")
        
        survival_data = {}
        
        for group in groups:
            mask = df_data[group_col] == group
            T = df_data[mask][duration_col].values
            E = df_data[mask][event_col].values
            
            kmf.fit(T, E, label=group_labels.get(group, str(group)))
            
            group_label = group_labels.get(group, str(group))
            f.write(f"\n{group_label}\n")
            f.write("-"*80 + "\n")
            f.write(f"样本数: {len(T)}\n")
            f.write(f"事件数 (复发): {np.sum(E)}\n")
            f.write(f"无事件数: {len(E) - np.sum(E)}\n")
            f.write(f"随访时间范围: {T.min():.1f} - {T.max():.1f} 天\n")
            f.write(f"平均随访时间: {T.mean():.1f} 天\n")
            f.write(f"中位随访时间: {np.median(T):.1f} 天\n\n")
            
            # 获取生存概率在特定时间点
            try:
                surv_func = kmf.survival_function_
                f.write(f"生存概率:\n")
                for timepoint in [30, 90, 180, 365]:
                    if timepoint <= T.max():
                        # 找到最接近的时间点
                        idx = np.searchsorted(surv_func.index, timepoint)
                        if idx > 0:
                            prob = surv_func.iloc[idx-1].values[0]
                            f.write(f"  {timepoint}天: {prob:.4f}\n")
            except:
                pass
            
            survival_data[group_label] = {
                'n': len(T),
                'events': np.sum(E),
                'mean_time': T.mean(),
                'median_time': np.median(T)
            }
        
        # Log-rank test
        f.write("\n" + "="*80 + "\n")
        f.write("Log-Rank Test\n")
        f.write("="*80 + "\n")
        
        mask_0 = df_data[group_col] == groups[0]
        mask_1 = df_data[group_col] == groups[1]
        
        T_0 = df_data[mask_0][duration_col].values
        E_0 = df_data[mask_0][event_col].values
        T_1 = df_data[mask_1][duration_col].values
        E_1 = df_data[mask_1][event_col].values
        
        results = logrank_test(T_0, T_1, E_0, E_1)
        
        f.write(f"Test statistic: {results.test_statistic:.4f}\n")
        f.write(f"P-value: {results.p_value:.6f}\n")
        if results.p_value < 0.05:
            f.write("结论: 两组间生存曲线有显著差异 (p < 0.05)\n")
        else:
            f.write("结论: 两组间生存曲线无显著差异 (p >= 0.05)\n")

def process_dataset(pred_file, surv_file, threshold, dataset_name, dataset_short):
    """处理单个数据集"""
    print(f"\n{'#'*80}")
    print(f"# 处理{dataset_name}")
    print(f"{'#'*80}")
    
    # 加载数据
    df_merged = load_and_prepare_data(pred_file, surv_file, threshold, dataset_name)
    
    # 输出数据统计
    print(f"\n{'='*80}")
    print(f"{dataset_name}数据统计")
    print(f"{'='*80}")
    
    print(f"\n按模型预测分组的统计:")
    for group, label in [(0, f'Low Risk'), (1, f'High Risk')]:
        mask = df_merged['model_group'] == group
        n = mask.sum()
        recurrences = df_merged[mask]['recurrence'].sum()
        print(f"\n{label} (Pred Prob {'<' if group==0 else '>='} {threshold}):")
        print(f"  患者数: {n}")
        print(f"  复发数: {recurrences}")
        print(f"  无复发数: {n - recurrences}")
        if n > 0:
            print(f"  复发率: {recurrences/n*100:.1f}%")
    
    # 创建数据集特定的输出目录
    dataset_output_dir = os.path.join(OUTPUT_DIR, dataset_short)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # 1. 基于模型预测的生存曲线
    print(f"\n{'='*80}")
    print(f"绘制{dataset_name}基于模型预测的Kaplan-Meier曲线...")
    print(f"{'='*80}")
    
    model_curve_path = os.path.join(dataset_output_dir, f"survival_curve_model_prediction_{dataset_short}.png")
    model_group_labels = {
        0: f'Low Risk',
        1: f'High Risk'
    }
    model_p_value = plot_survival_curve(
        df_merged, 
        duration_col='time', 
        event_col='recurrence',
        group_col='model_group',
        group_labels=model_group_labels,
        save_path=model_curve_path,
        title=f'Recurrence-Free Survival by Model Prediction',
        colors=['#fd83e2', '#944c83'],
        dataset_name=dataset_name
    )
    
    # 2. 基于真实标签的生存曲线
    print(f"\n{'='*80}")
    print(f"绘制{dataset_name}基于真实标签的Kaplan-Meier曲线...")
    print(f"{'='*80}")
    
    true_curve_path = os.path.join(dataset_output_dir, f"survival_curve_true_label_{dataset_short}.png")
    true_group_labels = {
        0: 'Non-Recurrence',
        1: 'Recurrence'
    }
    true_p_value = plot_survival_curve(
        df_merged,
        duration_col='time',
        event_col='recurrence',
        group_col='true_label',
        group_labels=true_group_labels,
        save_path=true_curve_path,
        title=f'Recurrence-Free Survival by True Label ({dataset_name})',
        colors=['#85ccbf', '#f6796a'],
        dataset_name=dataset_name
    )
    
    # 3. 生成统计报告
    print(f"\n{'='*80}")
    print(f"生成{dataset_name}统计报告...")
    print(f"{'='*80}")
    
    model_stats_path = os.path.join(dataset_output_dir, f"survival_statistics_model_{dataset_short}.txt")
    generate_survival_statistics(
        df_merged,
        duration_col='time',
        event_col='recurrence',
        group_col='model_group',
        group_labels=model_group_labels,
        save_path=model_stats_path,
        dataset_name=dataset_name
    )
    
    true_stats_path = os.path.join(dataset_output_dir, f"survival_statistics_true_label_{dataset_short}.txt")
    generate_survival_statistics(
        df_merged,
        duration_col='time',
        event_col='recurrence',
        group_col='true_label',
        group_labels=true_group_labels,
        save_path=true_stats_path,
        dataset_name=dataset_name
    )
    
    # 4. 保存合并后的数据
    merged_data_path = os.path.join(dataset_output_dir, f"merged_data_with_predictions_{dataset_short}.csv")
    df_merged.to_csv(merged_data_path, index=False)
    print(f"\n{dataset_name}合并数据已保存: {merged_data_path}")
    
    # 返回统计信息
    return {
        'dataset': dataset_name,
        'short': dataset_short,
        'n': len(df_merged),
        'recurrences': df_merged['recurrence'].sum(),
        'model_p': model_p_value,
        'true_p': true_p_value,
        'threshold': threshold,
        'output_dir': dataset_output_dir
    }

def main():
    print(f"\n{'*'*80}")
    print(f"* Kaplan-Meier生存曲线分析（验证集和测试集）")
    print(f"{'*'*80}")
    
    # 处理验证集
    val_results = process_dataset(
        VALIDATION_PREDICTIONS_FILE,
        VALIDATION_SURVIVAL_FILE,
        VALIDATION_THRESHOLD,
        "验证集",
        "validation"
    )
    
    # 处理测试集
    test_results = process_dataset(
        TEST_PREDICTIONS_FILE,
        TEST_SURVIVAL_FILE,
        TEST_THRESHOLD,
        "测试集",
        "test"
    )
    
    # 总结
    print(f"\n{'='*80}")
    print("分析完成！")
    print(f"{'='*80}")
    
    # 创建总结文件
    summary_path = os.path.join(OUTPUT_DIR, "survival_analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Kaplan-Meier生存曲线分析总结\n")
        f.write("="*80 + "\n\n")
        
        f.write("验证集\n")
        f.write("-"*80 + "\n")
        f.write(f"总患者数: {val_results['n']}\n")
        f.write(f"总复发数: {val_results['recurrences']}\n")
        f.write(f"总无复发数: {val_results['n'] - val_results['recurrences']}\n")
        f.write(f"使用的阈值: {val_results['threshold']}\n")
        f.write(f"模型预测生存曲线 Log-rank p值: {val_results['model_p']:.6f}\n")
        f.write(f"真实标签生存曲线 Log-rank p值: {val_results['true_p']:.6f}\n")
        f.write(f"输出目录: {val_results['output_dir']}\n\n")
        
        f.write("测试集\n")
        f.write("-"*80 + "\n")
        f.write(f"总患者数: {test_results['n']}\n")
        f.write(f"总复发数: {test_results['recurrences']}\n")
        f.write(f"总无复发数: {test_results['n'] - test_results['recurrences']}\n")
        f.write(f"使用的阈值: {test_results['threshold']}\n")
        f.write(f"模型预测生存曲线 Log-rank p值: {test_results['model_p']:.6f}\n")
        f.write(f"真实标签生存曲线 Log-rank p值: {test_results['true_p']:.6f}\n")
        f.write(f"输出目录: {test_results['output_dir']}\n")
    
    print(f"\n输出文件位置: {OUTPUT_DIR}")
    print(f"\n验证集 (Threshold = {VALIDATION_THRESHOLD}):")
    print(f"  总患者数: {val_results['n']}")
    print(f"  总复发数: {val_results['recurrences']}")
    print(f"  输出目录: {val_results['output_dir']}")
    
    print(f"\n测试集 (Threshold = {TEST_THRESHOLD}):")
    print(f"  总患者数: {test_results['n']}")
    print(f"  总复发数: {test_results['recurrences']}")
    print(f"  输出目录: {test_results['output_dir']}")
    
    print(f"\n总结文件: {summary_path}")

if __name__ == "__main__":
    main()