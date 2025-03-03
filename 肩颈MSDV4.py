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
import json
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit  # 改为时间序列拆分
from sklearn.metrics import (classification_report, 
                           roc_curve, auc, 
                           confusion_matrix, 
                           ConfusionMatrixDisplay)
from joblib import dump, load
from datetime import datetime
from github import Github

# ---------------------- 新增模块化组件 ----------------------
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
        # 多维度规则定义
        neck_threshold = 20  # 颈部角度阈值
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
        
        # 多维度评估指标
        st.subheader("模型评估报告")
        
        # 1. 分类报告
        st.write("#### 分类指标")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
        # 2. 混淆矩阵
        st.write("#### 混淆矩阵")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)
        
        # 3. ROC曲线
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
        # 生成元数据
        meta = {
            "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_version": self.data_version,
            "features": self.feature_names,
            "train_samples": train_shape[0],
            "best_threshold": float(self.best_threshold),
            "metrics": metrics
        }
        
        # 保存模型和元数据
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = f"MSD-Model-{timestamp}"
        
        # 保存模型文件
        dump(model, f"/tmp/{model_name}.joblib")
        
        # 保存元数据
        with open(f"/tmp/{model_name}.json", "w") as f:
            json.dump(meta, f)
        
        return model_name

# ---------------------- 改进后的主流程 ----------------------
def main():
    st.title("改进版-肩颈姿势分析系统")
    
    # 数据加载
    uploaded_file = st.file_uploader("上传数据文件", type="csv")
    if not uploaded_file:
        return
    
    # 数据预处理
    data = pd.read_csv(uploaded_file)
    data.columns = ['工站(w)', '时间(s)', '颈部角度(°)', 
                   '肩部前屈角度(°)', '肩部外展角度(°)']
    
    # 初始化模型管理器
    manager = ModelManager(data_version="1.0")
    
    # 标签生成（改进点1）
    data = manager.generate_labels(data)
    
    # 时间序列拆分（改进点2）
    tscv = TimeSeriesSplit(n_splits=5)
    X = data[manager.feature_names]
    y = data['Label']
    
    # 交叉验证训练
    all_metrics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        st.subheader(f"交叉验证 Fold {fold+1}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 模型训练
        model = manager.train_model(X_train, y_train)
        
        # 模型评估（改进点3）
        roc_auc = manager.evaluate_model(model, X_test, y_test)
        all_metrics.append(roc_auc)
        
        # 特征解释（改进点5）
        st.write("#### SHAP特征重要性")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        st.pyplot(fig)
    
    # 最终模型保存（改进点4）
    if np.mean(all_metrics) > 0.7:  # 性能阈值验证
        final_model = manager.train_model(X, y)
        model_name = manager.save_model(
            final_model,
            {"mean_auc": np.mean(all_metrics)},
            X.shape
        )
        st.success(f"模型 {model_name} 已通过验证并保存")
    else:
        st.warning("模型性能未达标准，请检查数据质量")

# ---------------------- 版本控制模块 ----------------------
def github_ops():
    """改进后的GitHub操作"""
    # （保持原有文件上传逻辑，增加元数据文件上传）
    pass 

if __name__ == "__main__":
    main()
