#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
import json
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit  # 使用时间序列拆分
from sklearn.metrics import (classification_report, 
                             roc_curve, auc, 
                             confusion_matrix, 
                             ConfusionMatrixDisplay)
from joblib import dump, load
from datetime import datetime
from github import Github

# ---------------------- 模型管理模块 ----------------------
class ModelManager:
    """封装模型训练、评估、保存的完整流程"""
    
    def __init__(self, data_version="1.0"):
        self.data_version = data_version
        self.best_threshold = 0.5
        self.feature_names = ['颈部角度(°)', 
                              '肩部前屈角度(°)', 
                              '肩部外展角度(°)']
    
    def generate_labels(self, df):
        """基于业务规则生成可靠标签"""
        # 定义各项阈值
        neck_threshold = 20         # 颈部角度阈值
        shoulder_flex_threshold = 45  # 肩部前屈阈值
        shoulder_abduct_threshold = 30  # 肩部外展阈值
        
        # 复合规则生成标签
        df['Label'] = (
            (df['颈部角度(°)'] > neck_threshold) |
            (df['肩部前屈角度(°)'] > shoulder_flex_threshold) |
            (df['肩部外展角度(°)'] > shoulder_abduct_threshold)
        ).astype(int)
        return df
    
    def train_model(self, X_train, y_train):
        """模型训练流程"""
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """综合模型评估"""
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # 自动选择最佳阈值
        self.best_threshold = thresholds[(tpr - fpr).argmax()]
        y_pred = (y_prob >= self.best_threshold).astype(int)
        
        # 显示评估报告
        st.subheader("模型评估报告")
        st.write("#### 分类指标")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
        st.write("#### 混淆矩阵")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)
        
        st.write("#### ROC曲线")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.scatter(fpr[(tpr - fpr).argmax()], 
                   tpr[(tpr - fpr).argmax()], 
                   c='red', label=f'最佳阈值: {self.best_threshold:.2f}')
        ax.set_xlabel("假阳性率")
        ax.set_ylabel("真阳性率")
        ax.legend()
        st.pyplot(fig)
        
        return roc_auc
    
    def save_model(self, model, metrics, train_shape):
        """带版本控制的模型保存"""
        meta = {
            "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_version": self.data_version,
            "features": self.feature_names,
            "train_samples": train_shape[0],
            "best_threshold": float(self.best_threshold),
            "metrics": metrics
        }
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = f"MSD-Model-{timestamp}"
        
        # 将模型保存到临时目录
        dump(model, f"/tmp/{model_name}.joblib")
        # 保存元数据
        with open(f"/tmp/{model_name}.json", "w") as f:
            json.dump(meta, f)
        
        return model_name

# ---------------------- 主流程 ----------------------
def main():
    st.title("改进版-肩颈姿势分析系统")
    
    # 上传数据文件（CSV 格式）
    uploaded_file = st.file_uploader("上传数据文件", type="csv")
    if not uploaded_file:
        st.info("请上传数据文件后继续")
        return
    
    # 数据预处理：读取并重命名列
    data = pd.read_csv(uploaded_file)
    data.columns = ['工站(w)', '时间(s)', '颈部角度(°)', 
                    '肩部前屈角度(°)', '肩部外展角度(°)']
    
    # 初始化模型管理器
    manager = ModelManager(data_version="1.0")
    # 生成标签
    data = manager.generate_labels(data)
    
    # 时间序列拆分
    tscv = TimeSeriesSplit(n_splits=5)
    X = data[manager.feature_names]
    y = data['Label']
    
    # 交叉验证训练与评估
    all_metrics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        st.subheader(f"交叉验证 Fold {fold+1}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 模型训练
        model = manager.train_model(X_train, y_train)
        # 模型评估
        roc_auc = manager.evaluate_model(model, X_test, y_test)
        all_metrics.append(roc_auc)
        
        # SHAP特征解释
        st.write("#### SHAP特征重要性")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(fig)
    
    # 根据交叉验证结果决定是否保存最终模型
    mean_auc = np.mean(all_metrics)
    st.write(f"交叉验证 AUC 均值：{mean_auc:.2f}")
    if mean_auc > 0.7:  # 性能阈值
        final_model = manager.train_model(X, y)
        model_name = manager.save_model(final_model, {"mean_auc": mean_auc}, X.shape)
        st.success(f"模型 {model_name} 已通过验证并保存")
    else:
        st.warning("模型性能未达标准，请检查数据质量")

# ---------------------- GitHub操作模块 ----------------------
def github_ops():
    """改进后的GitHub操作：可在此处实现模型及元数据的上传逻辑"""
    pass

if __name__ == "__main__":
    main()
