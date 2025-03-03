#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from matplotlib import font_manager
from github import Github

# 动态读取Token
token = os.getenv("GITHUB_TOKEN")
if not token:
    st.error("GitHub Token 未设置。请在 Streamlit Cloud 的 Secrets 中添加 GITHUB_TOKEN。")
    st.stop()

# GitHub 配置
repo_name = "xantoxia/neckv4"  # 替换为你的 GitHub 仓库
models_folder = "models/"      # GitHub 仓库中模型文件存储路径
latest_model_file = "latest_model_info.txt"  # 最新模型信息文件
commit_message = "从Streamlit更新模型文件"      # 提交信息

# 定义带时间戳的备份文件名
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_filename = f"MSD-{timestamp}.joblib"

# 上传文件到 GitHub
def upload_file_to_github(file_path, github_path, commit_message):
    try:
        g = Github(token)
        repo = g.get_repo(repo_name)
        # 读取文件内容
        with open(file_path, "rb") as f:
            content = f.read()
        # 检查文件是否存在
        try:
            file = repo.get_contents(github_path)
            repo.update_file(github_path, commit_message, content, file.sha)
            st.success(f"文件已成功更新到 GitHub 仓库：{github_path}")
        except Exception as ex:
            repo.create_file(github_path, commit_message, content)
            st.success(f"文件已成功上传到 GitHub 仓库：{github_path}")
    except Exception as e:
        st.error(f"上传文件到 GitHub 失败：{e}")

# 下载最新模型文件
def download_latest_model_from_github():
    try:
        g = Github(token)
        repo = g.get_repo(repo_name)
        try:
            latest_info = repo.get_contents(models_folder + latest_model_file).decoded_content.decode()
            latest_model_path = models_folder + latest_info.strip()
            st.write(f"最新模型路径：{latest_model_path}")
            file_content = repo.get_contents(latest_model_path)
            with open("/tmp/latest_model.joblib", "wb") as f:
                f.write(file_content.decoded_content)
            st.success("成功下载最新模型！")
            return "/tmp/latest_model.joblib"
        except Exception as ex:
            st.warning("未找到最新模型信息文件，无法下载模型。")
            return None
    except Exception as e:
        st.error(f"从 GitHub 下载模型失败：{e}")
        return None

# 设置中文字体
simhei_font = font_manager.FontProperties(fname="SimHei.ttf")
plt.rcParams['font.family'] = simhei_font.get_name()  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# Streamlit 标题
st.title("肩颈角度分析与异常检测")
st.write("结合AI规则与机器学习模型，实现自动检测异常作业姿势并进行可视化分析。")

# 模板下载
with open("肩颈角度数据模版.csv", "rb") as file:
    st.download_button(
        label="下载 CSV 模板",
        data=file,
        file_name="template.csv",
        mime="text/csv"
    )

# 数据加载与预处理
uploaded_file = st.file_uploader("上传肩颈角度数据文件 (CSV 格式)", type="csv")

if uploaded_file is not None:
    # 提取文件名并去掉扩展名
    csv_file_name = os.path.splitext(uploaded_file.name)[0]
    st.markdown(f"<h3 style='color:blue;'>{csv_file_name} 肩颈作业姿势分析</h3>", unsafe_allow_html=True)
    # 读取数据
    data = pd.read_csv(uploaded_file)
    data.columns = ['工站(w)', '时间(s)', '颈部角度(°)', '肩部前屈角度(°)', '肩部外展角度(°)']
    # 显示数据预览
    st.write("### 1.1  数据预览")
    data_reset = data.copy()
    data_reset.index += 1
    data_reset.index.name = "序号"
    st.write(data_reset.head())

    # 按工站汇总统计
    def summarize_by_station(data):
        st.write("### 1.2  数据统计分析")
        station_summary = data.groupby('工站(w)').agg({
            '时间(s)': ['count'],
            '颈部角度(°)': ['mean', 'min', 'max', 'std'],
            '肩部前屈角度(°)': ['mean', 'min', 'max', 'std'],
            '肩部外展角度(°)': ['mean', 'min', 'max', 'std']
        })
        station_summary.columns = ['_'.join(col).strip() for col in station_summary.columns.values]
        station_summary.reset_index(inplace=True)
        station_summary = station_summary.round(2)
        st.write(station_summary)
    summarize_by_station(data)

    # 数据可视化函数（保持原有可视化展示）
    def generate_visualizations(data):
        st.write("## 各工站数据可视化分析")
        grouped = data.groupby('工站(w)')
        for station, group_data in grouped:
            st.write(f"### 工站 {station} 的数据可视化")
            # 1. 3D 散点图
            st.write("#### 3D 散点图")
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                group_data['时间(s)'], 
                group_data['颈部角度(°)'], 
                group_data['肩部前屈角度(°)'], 
                c=group_data['肩部外展角度(°)'], 
                cmap='viridis'
            )
            ax.set_xlabel('时间(s)', fontproperties=simhei_font)
            ax.set_ylabel('颈部角度(°)', fontproperties=simhei_font)
            ax.set_zlabel('肩部前屈角度(°)', fontproperties=simhei_font)
            plt.title(f'工站 {station} 肩颈角度3D可视化散点图', fontproperties=simhei_font)
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('肩部外展角度(°)', fontproperties=simhei_font)
            st.pyplot(fig)
            
            # 动态分析结论（3D散点图）
            st.write(f"**工站 {station} 的动态分析结论（3D散点图）：**")
            neck_max = group_data['颈部角度(°)'].max()
            if neck_max < 20:
                st.write("- 颈部角度均在20°以内，MSD风险较低。")
            elif 20 <= neck_max <= 40:
                st.write("- 存在部分时间点颈部角度超过20°，存在一定MSD风险。")
            else:
                st.write("- 存在部分时间点颈部角度超过40°，需注意极端低头动作。")
            
            shoulder_max = group_data['肩部前屈角度(°)'].max()
            if shoulder_max < 15:
                st.write("- 肩部前屈角度波动较小，动作一致。")
            elif shoulder_max >= 45:
                st.write("- 部分时间点肩部前屈角度大于45°，需注意作业姿势。")
            if group_data['肩部外展角度(°)'].mean() > 20:
                st.write("- 肩部外展角度整体较大，上臂运动强度可能较高。")
            
            # 2. 时间变化折线图
            st.write("#### 肩颈角度时间变化折线图（带预警线）")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(group_data['时间(s)'], group_data['颈部角度(°)'], label='颈部角度(°)', color='blue', linewidth=2)
            ax2.plot(group_data['时间(s)'], group_data['肩部前屈角度(°)'], label='肩部前屈角度(°)', color='green', linewidth=2)
            ax2.axhline(y=20, color='red', linestyle='--', linewidth=1.5, label='颈部预警线 (20°)')
            ax2.axhline(y=45, color='orange', linestyle='--', linewidth=1.5, label='肩部预警线 (45°)')
            ax2.set_xlabel('时间(s)', fontproperties=simhei_font, fontsize=12)
            ax2.set_ylabel('角度(°)', fontproperties=simhei_font, fontsize=12)
            ax2.set_title(f'工站 {station} 的肩颈角度时间变化折线图', fontproperties=simhei_font, fontsize=12)
            ax2.legend(prop=simhei_font, fontsize=10)
            st.pyplot(fig2)
            
            # 动态分析结论（折线图）
            st.write(f"**工站 {station} 的动态分析结论（折线图）：**")
            neck_exceed = (group_data['颈部角度(°)'] > 20).sum()
            ratio_neck = neck_exceed / len(group_data)
            if neck_exceed > 0:
                if ratio_neck > 0.5:
                    st.markdown(f"<span style='color:red;'>- {neck_exceed}个时间点超过20°（{ratio_neck:.2%}），颈部风险较高。</span>", unsafe_allow_html=True)
                elif ratio_neck >= 0.25:
                    st.markdown(f"<span style='color:orange;'>- {neck_exceed}个时间点超过20°（{ratio_neck:.2%}），颈部风险中等。</span>", unsafe_allow_html=True)
                else:
                    st.write(f"- {neck_exceed}个时间点超过20°，风险轻微。")
            else:
                st.write("- 颈部角度均在20°以内，风险较低。")
            
            shoulder_exceed = (group_data['肩部前屈角度(°)'] > 45).sum()
            ratio_shoulder = shoulder_exceed / len(group_data)
            if shoulder_exceed > 0:
                if ratio_shoulder > 0.5:
                    st.markdown(f"<span style='color:red;'>- {shoulder_exceed}个时间点超过45°（{ratio_shoulder:.2%}），肩部风险较高。</span>", unsafe_allow_html=True)
                elif ratio_shoulder >= 0.25:
                    st.markdown(f"<span style='color:orange;'>- {shoulder_exceed}个时间点超过45°（{ratio_shoulder:.2%}），肩部风险中等。</span>", unsafe_allow_html=True)
                else:
                    st.write(f"- {shoulder_exceed}个时间点超过45°，风险轻微。")
            else:
                st.write("- 肩部前屈角度均在45°以内，风险较低。")
    generate_visualizations(data)
    
    # 综合分析与AI模型融合检测
    def comprehensive_analysis_by_workstation(data, model, scaler):
        st.write("### 3.1  机器学习与规则融合分析结果")
        grouped = data.groupby('工站(w)')
        total_abnormal_indices = []
        
        # 定义一个辅助函数，利用规则和ML预测融合决策
        def get_fusion_decision(row):
            # 规则检测
            rule_decision = "正常"
            if row['颈部角度(°)'] > (row['颈部角度(°)_mean'] + row['颈部角度(°)_std']):
                rule_decision = "异常"
            elif row['肩部前屈角度(°)'] > (row['肩部前屈角度(°)_mean'] + row['肩部前屈角度(°)_std']):
                rule_decision = "异常"
            # ML预测：注意对单条数据进行标准化转换
            row_input = np.array([[row['颈部角度(°)'], row['肩部前屈角度(°)'], row['肩部外展角度(°)']]])
            row_scaled = scaler.transform(row_input)
            ml_pred = model.predict(row_scaled)[0]
            ml_decision = "异常" if ml_pred == 1 else "正常"
            # 融合决策：若两者均异常，则为“确定异常”；任一异常则为“疑似异常”
            if rule_decision == "异常" and ml_decision == "异常":
                final_decision = "确定异常"
            elif rule_decision == "异常" or ml_decision == "异常":
                final_decision = "疑似异常"
            else:
                final_decision = "正常"
            return rule_decision, ml_decision, final_decision

        # 针对每个工站进行分析
        for station, group_data in grouped:
            st.write(f"#### 工站 {station} 的融合检测结果")
            # 动态阈值计算
            group_stats = group_data.agg({
                '颈部角度(°)': ['mean', 'std'],
                '肩部前屈角度(°)': ['mean', 'std']
            })
            neck_mean = group_stats.loc[:, ('颈部角度(°)', 'mean')]
            neck_std  = group_stats.loc[:, ('颈部角度(°)', 'std')]
            shoulder_mean = group_stats.loc[:, ('肩部前屈角度(°)', 'mean')]
            shoulder_std  = group_stats.loc[:, ('肩部前屈角度(°)', 'std')]
            # 将统计结果赋值到数据中，方便规则判断
            group_data = group_data.copy()
            group_data['颈部角度(°)_mean'] = neck_mean.values[0]
            group_data['颈部角度(°)_std'] = neck_std.values[0]
            group_data['肩部前屈角度(°)_mean'] = shoulder_mean.values[0]
            group_data['肩部前屈角度(°)_std'] = shoulder_std.values[0]
            
            st.write(f"- **动态阈值说明**：若颈部角度 > {neck_mean.values[0] + neck_std.values[0]:.2f}° 或肩部前屈角度 > {shoulder_mean.values[0] + shoulder_std.values[0]:.2f}°，则视为异常。")
            
            abnormal_indices = []
            # 对每条数据融合判断
            for i, row in group_data.iterrows():
                rule_decision, ml_decision, fusion_decision = get_fusion_decision(row)
                st.write(f"- 第 {i+1} 条数据：规则检测：{rule_decision}；ML检测：{ml_decision}；最终融合判断：{fusion_decision}")
                if fusion_decision != "正常":
                    abnormal_indices.append(i)
            if abnormal_indices:
                st.write(f"##### 工站 {station} 总结：共检测到 {len(abnormal_indices)} 条异常数据。")
            else:
                st.write(f"##### 工站 {station} 总结：未检测到异常数据。")
            total_abnormal_indices.extend(abnormal_indices)
        return total_abnormal_indices

    # ----- 机器学习部分 -----
    # 提取特征与标签
    X = data[['颈部角度(°)', '肩部前屈角度(°)', '肩部外展角度(°)']]
    # 若数据中未提供标签，使用基于规则的初步标注（请确保后续使用专业标注数据）
    if 'Label' not in data.columns:
        st.warning("数据中未提供标注，采用规则生成初步标签，请尽快引入专业标注数据。")
        data['Label'] = ((data['颈部角度(°)'] > 20) | (data['肩部前屈角度(°)'] > 45)).astype(int)
    y = data['Label']

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # 优化：使用 GridSearchCV 进行超参数调优
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                               param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    st.write("最佳超参数：", grid_search.best_params_)

    # 采用最佳模型进行训练与预测
    model = best_model
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 评估：绘制ROC曲线并选择最佳阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    best_index = (tpr - fpr).argmax()
    best_threshold = thresholds[best_index]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'r--', label="随机模型")
    ax.scatter(fpr[best_index], tpr[best_index], color='red', label=f'最佳阈值: {best_threshold:.2f}')
    ax.annotate(f'({fpr[best_index]:.2f}, {tpr[best_index]:.2f})',
                xy=(fpr[best_index], tpr[best_index]),
                xytext=(fpr[best_index]-0.2, tpr[best_index]-0.1),
                arrowprops=dict(facecolor='red', arrowstyle='->'),
                fontsize=10,
                fontproperties=simhei_font)
    ax.set_xlabel('假阳性率', fontproperties=simhei_font)
    ax.set_ylabel('真阳性率', fontproperties=simhei_font)
    ax.set_title('ROC曲线', fontproperties=simhei_font)
    ax.legend(loc='lower right', prop=simhei_font)
    st.pyplot(fig)
    
    st.write("### 3.4  AI模型质量评估")
    st.write(f"模型AUC为 {roc_auc:.2f}，最佳阈值为 {best_threshold:.2f}。")
    
    # 调用融合分析函数，对每个工站进行逐条数据检测
    total_abnormal_indices = comprehensive_analysis_by_workstation(data, model, scaler)
    
    # 保存新模型到本地临时路径，并上传到 GitHub
    local_model_path = f"/tmp/{model_filename}"
    dump(model, local_model_path)
    st.write("模型已训练并保存到本地临时路径。")
    upload_file_to_github(local_model_path, models_folder + model_filename, commit_message)
    st.write("模型已保存并上传到 GitHub。")
    
    # 更新最新模型信息
    latest_info_path = "/tmp/" + latest_model_file
    with open(latest_info_path, "w") as f:
        f.write(model_filename)
    upload_file_to_github(latest_info_path, models_folder + latest_model_file, "更新最新模型信息")
    st.success("新模型已上传，并更新最新模型记录。")
    
    st.write("#### 页面导出提示")
    st.info("如需将页面导出为 HTML 文件，请在浏览器中按 Ctrl+S 保存。")
