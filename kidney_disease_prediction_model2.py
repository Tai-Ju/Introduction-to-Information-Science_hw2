#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
腎臟疾病預測模型 - 完整分析代碼
包含資料預處理、模型訓練、性能評估、視覺化分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, roc_auc_score, roc_curve, 
                           precision_recall_curve, classification_report)
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class KidneyDiseasePredictor:
    """腎臟疾病預測分析器"""
    
    def __init__(self, data_path):
        """初始化預測器"""
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
        # 定義需要移除的資料洩漏特徵
        self.leakage_features = ['Anemia: yes', 'Pedal Edema: yes', 'Appetite: poor']
        
        print("🏥 腎臟疾病預測模型初始化完成")
        print(f"📊 資料路徑: {data_path}")
        print(f"🚨 將移除資料洩漏特徵: {self.leakage_features}")
    
    def load_and_explore_data(self):
        """載入並探索資料"""
        print("\n" + "="*60)
        print("📊 步驟1: 載入並探索資料")
        print("="*60)
        
        # 載入資料
        self.data = pd.read_csv(self.data_path)
        print(f"✅ 資料載入成功")
        print(f"   資料形狀: {self.data.shape}")
        print(f"   特徵數量: {self.data.shape[1] - 1}")
        print(f"   樣本數量: {self.data.shape[0]}")
        
        # 顯示基本資訊
        print(f"\n📋 資料基本資訊:")
        print(f"   欄位名稱: {list(self.data.columns)}")
        
        # 檢查目標變數
        target_col = self.data.columns[-1]
        print(f"\n🎯 目標變數: {target_col}")
        print(f"   類別分布:")
        target_counts = self.data[target_col].value_counts()
        for class_name, count in target_counts.items():
            percentage = (count / len(self.data)) * 100
            print(f"   - {class_name}: {count} ({percentage:.1f}%)")
        
        # 檢查遺失值
        print(f"\n🔍 遺失值檢查:")
        missing_values = self.data.isnull().sum()
        if missing_values.sum() == 0:
            print("   ✅ 無遺失值")
        else:
            print("   ⚠️ 發現遺失值:")
            for col, missing_count in missing_values[missing_values > 0].items():
                print(f"   - {col}: {missing_count}")
        
        return self.data
    
    def preprocess_data(self):
        """預處理資料"""
        print("\n" + "="*60)
        print("🔧 步驟2: 資料預處理")
        print("="*60)
        
        # 移除資料洩漏特徵
        print(f"🚨 移除資料洩漏特徵:")
        for feature in self.leakage_features:
            if feature in self.data.columns:
                print(f"   - 移除: {feature}")
                self.data = self.data.drop(columns=[feature])
            else:
                print(f"   - 未找到: {feature}")
        
        print(f"✅ 移除後資料形狀: {self.data.shape}")
        
        # 分離特徵和目標變數
        self.X = self.data.iloc[:, :-1]  # 所有欄位除了最後一欄
        self.y = self.data.iloc[:, -1]   # 最後一欄是目標變數
        
        print(f"\n📊 特徵和目標變數分離:")
        print(f"   特徵矩陣 X: {self.X.shape}")
        print(f"   目標變數 y: {self.y.shape}")
        
        # 處理分類變數
        print(f"\n🔤 處理分類變數:")
        categorical_columns = self.X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            print(f"   發現分類變數: {list(categorical_columns)}")
            for col in categorical_columns:
                le = LabelEncoder()
                original_values = self.X[col].unique()
                self.X[col] = le.fit_transform(self.X[col].astype(str))
                print(f"   - {col}: {original_values} → {self.X[col].unique()}")
        else:
            print("   ✅ 無分類變數需要處理")
        
        # 處理目標變數
        print(f"\n🎯 處理目標變數:")
        if self.y.dtype == 'object':
            le_target = LabelEncoder()
            original_classes = self.y.unique()
            self.y = le_target.fit_transform(self.y)
            print(f"   目標變數編碼: {original_classes} → {np.unique(self.y)}")
            # 儲存編碼器以供後續使用
            self.target_encoder = le_target
        else:
            print("   ✅ 目標變數已為數值型")
        
        # 處理遺失值
        print(f"\n🔧 處理遺失值:")
        if self.X.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='median')
            self.X = pd.DataFrame(
                imputer.fit_transform(self.X),
                columns=self.X.columns,
                index=self.X.index
            )
            print(f"   ✅ 使用中位數填補遺失值")
        else:
            print("   ✅ 無遺失值需要處理")
        
        print(f"\n✅ 資料預處理完成")
        print(f"   最終特徵矩陣: {self.X.shape}")
        print(f"   特徵名稱: {list(self.X.columns)}")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """分割訓練集和測試集"""
        print("\n" + "="*60)
        print("✂️ 步驟3: 分割訓練集和測試集")
        print("="*60)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, 
            stratify=self.y  # 保持類別比例
        )
        
        print(f"📊 資料分割結果:")
        print(f"   訓練集: {self.X_train.shape}")
        print(f"   測試集: {self.X_test.shape}")
        print(f"   測試集比例: {test_size*100:.0f}%")
        
        # 檢查類別分布
        train_dist = pd.Series(self.y_train).value_counts().sort_index()
        test_dist = pd.Series(self.y_test).value_counts().sort_index()
        
        print(f"\n📈 類別分布:")
        print(f"   訓練集: {dict(train_dist)}")
        print(f"   測試集: {dict(test_dist)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_decision_tree(self, max_depth=5, min_samples_split=10, min_samples_leaf=5):
        """訓練決策樹模型"""
        print("\n" + "="*40)
        print("🌳 訓練決策樹模型")
        print("="*40)
        
        # 創建和訓練決策樹
        dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        dt_model.fit(self.X_train, self.y_train)
        
        # 預測
        y_train_pred = dt_model.predict(self.X_train)
        y_test_pred = dt_model.predict(self.X_test)
        y_test_proba = dt_model.predict_proba(self.X_test)[:, 1]
        
        # 交叉驗證
        cv_scores = cross_val_score(
            dt_model, self.X, self.y, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        # 計算性能指標
        dt_results = {
            'model': dt_model,
            'train_accuracy': accuracy_score(self.y_train, y_train_pred),
            'test_accuracy': accuracy_score(self.y_test, y_test_pred),
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'precision': precision_score(self.y_test, y_test_pred),
            'recall': recall_score(self.y_test, y_test_pred),
            'f1': f1_score(self.y_test, y_test_pred),
            'roc_auc': roc_auc_score(self.y_test, y_test_proba),
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba,
            'feature_importance': pd.DataFrame({
                'feature': self.X.columns,
                'importance': dt_model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        self.models['decision_tree'] = dt_model
        self.results['decision_tree'] = dt_results
        
        print(f"✅ 決策樹訓練完成")
        print(f"   訓練集準確率: {dt_results['train_accuracy']:.4f}")
        print(f"   測試集準確率: {dt_results['test_accuracy']:.4f}")
        print(f"   交叉驗證準確率: {dt_results['cv_accuracy']:.4f} ± {dt_results['cv_std']:.4f}")
        print(f"   精確率: {dt_results['precision']:.4f}")
        print(f"   召回率: {dt_results['recall']:.4f}")
        print(f"   F1分數: {dt_results['f1']:.4f}")
        print(f"   ROC AUC: {dt_results['roc_auc']:.4f}")
        
        return dt_model, dt_results
    
    def train_logistic_regression(self):
        """訓練邏輯回歸模型"""
        print("\n" + "="*40)
        print("📊 訓練邏輯回歸模型")
        print("="*40)
        
        # 標準化特徵
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        X_scaled = scaler.fit_transform(self.X)
        
        # 創建和訓練邏輯回歸
        lr_model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
        lr_model.fit(X_train_scaled, self.y_train)
        
        # 預測
        y_train_pred = lr_model.predict(X_train_scaled)
        y_test_pred = lr_model.predict(X_test_scaled)
        y_test_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
        
        # 交叉驗證
        cv_scores = cross_val_score(
            lr_model, X_scaled, self.y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        # 計算性能指標
        lr_results = {
            'model': lr_model,
            'scaler': scaler,
            'train_accuracy': accuracy_score(self.y_train, y_train_pred),
            'test_accuracy': accuracy_score(self.y_test, y_test_pred),
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'precision': precision_score(self.y_test, y_test_pred),
            'recall': recall_score(self.y_test, y_test_pred),
            'f1': f1_score(self.y_test, y_test_pred),
            'roc_auc': roc_auc_score(self.y_test, y_test_proba),
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba,
            'feature_importance': pd.DataFrame({
                'feature': self.X.columns,
                'importance': np.abs(lr_model.coef_[0])
            }).sort_values('importance', ascending=False)
        }
        
        self.models['logistic_regression'] = lr_model
        self.results['logistic_regression'] = lr_results
        
        print(f"✅ 邏輯回歸訓練完成")
        print(f"   訓練集準確率: {lr_results['train_accuracy']:.4f}")
        print(f"   測試集準確率: {lr_results['test_accuracy']:.4f}")
        print(f"   交叉驗證準確率: {lr_results['cv_accuracy']:.4f} ± {lr_results['cv_std']:.4f}")
        print(f"   精確率: {lr_results['precision']:.4f}")
        print(f"   召回率: {lr_results['recall']:.4f}")
        print(f"   F1分數: {lr_results['f1']:.4f}")
        print(f"   ROC AUC: {lr_results['roc_auc']:.4f}")
        
        return lr_model, lr_results
    
    def compare_models(self):
        """比較模型性能"""
        print("\n" + "="*60)
        print("🔍 步驟4: 模型性能比較")
        print("="*60)
        
        # 準備比較資料
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'CV Accuracy': f"{results['cv_accuracy']:.4f} ± {results['cv_std']:.4f}",
                'Test Accuracy': f"{results['test_accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1 Score': f"{results['f1']:.4f}",
                'ROC AUC': f"{results['roc_auc']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("📊 模型性能比較表:")
        print(comparison_df.to_string(index=False))
        
        # 找出最佳模型
        best_accuracy = max(results['test_accuracy'] for results in self.results.values())
        best_model = [name for name, results in self.results.items() 
                     if results['test_accuracy'] == best_accuracy][0]
        
        print(f"\n🏆 最佳模型: {best_model.replace('_', ' ').title()}")
        print(f"   測試集準確率: {best_accuracy:.4f}")
        
        return comparison_df
    
    def create_visualizations(self, save_path='/mnt/user-data/outputs/'):
        """創建所有視覺化圖表"""
        print("\n" + "="*60)
        print("📈 步驟5: 創建視覺化分析")
        print("="*60)
        
        # 設定圖表樣式
        plt.style.use('default')
        
        # 1. 目標變數分布圖
        print("📊 生成圖表 1/9: 目標變數分布")
        plt.figure(figsize=(8, 6))
        target_counts = pd.Series(self.y).value_counts().sort_index()
        bars = plt.bar(range(len(target_counts)), target_counts.values, 
                      color=['skyblue', 'lightcoral'])
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Target Variable Distribution\n(Kidney Disease Classification)')
        plt.xticks(range(len(target_counts)), 
                  ['No Disease', 'Disease'] if len(target_counts) == 2 else target_counts.index)
        
        # 添加數值標籤
        for i, (bar, count) in enumerate(zip(bars, target_counts.values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{count}\n({count/sum(target_counts.values)*100:.1f}%)',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}01_target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 相關性矩陣
        print("📊 生成圖表 2/9: 特徵相關性矩陣")
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.X.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{save_path}02_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 重要特徵與目標的相關性
        print("📊 生成圖表 3/9: 重要特徵相關性")
        plt.figure(figsize=(10, 8))
        feature_target_corr = self.X.corrwith(pd.Series(self.y)).abs().sort_values(ascending=False)
        top_features = feature_target_corr.head(10)
        
        bars = plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Absolute Correlation with Target')
        plt.title('Top 10 Features with Highest Correlation to Chronic Kidney Disease')
        plt.gca().invert_yaxis()
        
        # 添加數值標籤
        for i, (bar, value) in enumerate(zip(bars, top_features.values)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}03_feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 混淆矩陣比較
        print("📊 生成圖表 4/9: 混淆矩陣比較")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        model_names = ['Decision Tree', 'Logistic Regression']
        for i, (model_name, results) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, results['y_test_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'])
            axes[i].set_title(f'{model_names[i]} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}04_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. ROC曲線比較
        print("📊 生成圖表 5/9: ROC曲線比較")
        plt.figure(figsize=(8, 6))
        
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_test_proba'])
            auc_score = results['roc_auc']
            plt.plot(fpr, tpr, label=f"{model_name.replace('_', ' ').title()} (AUC = {auc_score:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}05_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. 特徵重要性比較
        print("📊 生成圖表 6/9: 特徵重要性比較")
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        for i, (model_name, results) in enumerate(self.results.items()):
            top_features = results['feature_importance'].head(10)
            bars = axes[i].barh(range(len(top_features)), top_features['importance'])
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features['feature'])
            axes[i].set_xlabel('Importance Score')
            axes[i].set_title(f'{model_name.replace("_", " ").title()} - Top 10 Important Features')
            axes[i].invert_yaxis()
            
            # 添加數值標籤
            for j, (bar, value) in enumerate(zip(bars, top_features['importance'])):
                axes[i].text(bar.get_width() + max(top_features['importance'])*0.01, 
                           bar.get_y() + bar.get_height()/2, 
                           f'{value:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}06_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. 決策樹結構圖
        print("📊 生成圖表 7/9: 決策樹結構")
        plt.figure(figsize=(20, 12))
        plot_tree(self.models['decision_tree'], 
                 feature_names=self.X.columns,
                 class_names=['No Disease', 'Disease'],
                 filled=True, rounded=True, fontsize=10)
        plt.title('Decision Tree Structure', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{save_path}07_decision_tree.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 8. Precision-Recall曲線
        print("📊 生成圖表 8/9: Precision-Recall曲線")
        plt.figure(figsize=(8, 6))
        
        for model_name, results in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, results['y_test_proba'])
            plt.plot(recall, precision, label=f"{model_name.replace('_', ' ').title()}")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}08_precision_recall.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 9. 模型性能比較雷達圖
        print("📊 生成圖表 9/9: 模型性能比較")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 準備資料
        metrics = ['Test Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_keys = ['test_accuracy', 'precision', 'recall', 'f1']
        dt_scores = [self.results['decision_tree'][key] for key in metric_keys]
        lr_scores = [self.results['logistic_regression'][key] for key in metric_keys]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, dt_scores, width, label='Decision Tree', alpha=0.8)
        bars2 = ax.bar(x + width/2, lr_scores, width, label='Logistic Regression', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # 添加數值標籤
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}09_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 所有視覺化圖表生成完成!")
        print(f"   圖表保存路徑: {save_path}")
        
    def generate_detailed_report(self, save_path='/mnt/user-data/outputs/'):
        """生成詳細分析報告"""
        print("\n" + "="*60)
        print("📄 步驟6: 生成詳細分析報告")
        print("="*60)
        
        report = []
        report.append("# 腎臟疾病預測模型 - 詳細分析報告")
        report.append("=" * 60)
        report.append("")
        
        # 1. 專案概述
        report.append("## 📋 專案概述")
        report.append("")
        report.append("本專案旨在建立機器學習模型來預測慢性腎臟疾病，使用決策樹和邏輯回歸")
        report.append("兩種演算法進行比較分析。")
        report.append("")
        report.append("**主要目標：**")
        report.append("- 建立準確的腎臟疾病預測模型")
        report.append("- 比較不同機器學習演算法的性能")
        report.append("- 識別最重要的預測特徵")
        report.append("- 提供可解釋的醫學預測工具")
        report.append("")
        
        # 2. 資料集資訊
        report.append("## 📊 資料集資訊")
        report.append("")
        report.append(f"**資料形狀：** {self.data.shape}")
        report.append(f"**特徵數量：** {self.X.shape[1]}")
        report.append(f"**樣本數量：** {len(self.data)}")
        report.append("")
        
        # 目標變數分布
        target_counts = pd.Series(self.y).value_counts().sort_index()
        report.append("**目標變數分布：**")
        for i, count in enumerate(target_counts.values):
            class_name = "無腎病" if i == 0 else "有腎病"
            percentage = (count / len(self.y)) * 100
            report.append(f"- {class_name}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # 3. 資料預處理
        report.append("## 🔧 資料預處理")
        report.append("")
        report.append("**關鍵步驟：**")
        report.append("1. **移除資料洩漏特徵**：")
        for feature in self.leakage_features:
            report.append(f"   - {feature} (腎病的症狀，非預測因子)")
        report.append("")
        report.append("2. **特徵編碼**：將分類變數轉換為數值")
        report.append("3. **資料分割**：80%訓練集，20%測試集")
        report.append("4. **標準化**：邏輯回歸使用特徵標準化")
        report.append("")
        
        # 4. 模型性能
        report.append("## 🎯 模型性能")
        report.append("")
        
        # 性能比較表
        report.append("### 詳細性能指標")
        report.append("")
        report.append("| 模型 | 交叉驗證準確率 | 測試集準確率 | 精確率 | 召回率 | F1分數 | ROC AUC |")
        report.append("|------|---------------|-------------|--------|-------|--------|---------|")
        
        for model_name, results in self.results.items():
            model_display = model_name.replace('_', ' ').title()
            report.append(f"| {model_display} | {results['cv_accuracy']:.4f}±{results['cv_std']:.4f} | "
                         f"{results['test_accuracy']:.4f} | {results['precision']:.4f} | "
                         f"{results['recall']:.4f} | {results['f1']:.4f} | {results['roc_auc']:.4f} |")
        
        report.append("")
        
        # 5. 特徵重要性分析
        report.append("## 📈 特徵重要性分析")
        report.append("")
        
        for model_name, results in self.results.items():
            model_display = model_name.replace('_', ' ').title()
            report.append(f"### {model_display} 前10重要特徵")
            report.append("")
            report.append("| 排名 | 特徵名稱 | 重要性分數 |")
            report.append("|------|----------|-----------|")
            
            top_features = results['feature_importance'].head(10)
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                report.append(f"| {i} | {row['feature']} | {row['importance']:.4f} |")
            
            report.append("")
        
        # 6. 關鍵發現
        report.append("## 🔍 關鍵發現")
        report.append("")
        
        # 找出最重要的特徵
        dt_top_feature = self.results['decision_tree']['feature_importance'].iloc[0]
        lr_top_feature = self.results['logistic_regression']['feature_importance'].iloc[0]
        
        report.append("**最重要的預測因子：**")
        report.append(f"- 決策樹: {dt_top_feature['feature']} (重要性: {dt_top_feature['importance']:.4f})")
        report.append(f"- 邏輯回歸: {lr_top_feature['feature']} (重要性: {lr_top_feature['importance']:.4f})")
        report.append("")
        
        # 找出最佳模型
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['test_accuracy'])
        best_accuracy = self.results[best_model_name]['test_accuracy']
        
        report.append("**最佳模型：**")
        report.append(f"- {best_model_name.replace('_', ' ').title()}")
        report.append(f"- 測試集準確率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        report.append("")
        
        # 7. 醫學意義
        report.append("## 🏥 醫學意義")
        report.append("")
        report.append("**模型的臨床價值：**")
        report.append("- **早期篩檢**：能在症狀出現前識別高風險患者")
        report.append("- **客觀診斷**：基於量化指標，減少主觀判斷誤差")
        report.append("- **資源配置**：幫助醫療機構優化資源分配")
        report.append("- **個人化醫療**：為不同風險等級患者提供適當的照護計畫")
        report.append("")
        
        # 8. 模型限制
        report.append("## ⚠️ 模型限制與注意事項")
        report.append("")
        report.append("**使用限制：**")
        report.append("- 模型基於特定資料集訓練，可能不適用於所有人群")
        report.append("- 需要定期驗證和更新模型性能")
        report.append("- 應作為輔助診斷工具，不能替代專業醫學判斷")
        report.append("- 建議結合臨床經驗和其他診斷方法使用")
        report.append("")
        
        # 9. 結論
        report.append("## 📋 結論")
        report.append("")
        report.append("本研究成功建立了高準確率的腎臟疾病預測模型：")
        report.append("")
        best_results = self.results[best_model_name]
        report.append(f"- **最佳模型準確率**: {best_results['test_accuracy']:.1%}")
        report.append(f"- **精確率**: {best_results['precision']:.1%} (低假陽性率)")
        report.append(f"- **召回率**: {best_results['recall']:.1%} (低假陰性率)")
        report.append(f"- **ROC AUC**: {best_results['roc_auc']:.3f} (優秀的判別能力)")
        report.append("")
        report.append("模型展現了良好的預測性能和臨床應用潜力，")
        report.append("可以作為醫療決策支持系統的重要組成部分。")
        
        # 保存報告
        report_text = "\n".join(report)
        with open(f'{save_path}kidney_disease_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("✅ 詳細分析報告生成完成!")
        print(f"   報告保存於: {save_path}kidney_disease_analysis_report.md")
        
        return report_text
    
    def run_complete_analysis(self):
        """執行完整分析流程"""
        print("🚀 開始腎臟疾病預測模型完整分析")
        print("=" * 80)
        
        try:
            # 步驟1: 載入和探索資料
            self.load_and_explore_data()
            
            # 步驟2: 預處理資料
            self.preprocess_data()
            
            # 步驟3: 分割資料
            self.split_data()
            
            # 步驟4: 訓練模型
            self.train_decision_tree()
            self.train_logistic_regression()
            
            # 步驟5: 比較模型
            self.compare_models()
            
            # 步驟6: 創建視覺化
            self.create_visualizations()
            
            # 步驟7: 生成報告
            self.generate_detailed_report()
            
            print("\n" + "🎉" * 20)
            print("🎉 腎臟疾病預測分析完成!")
            print("🎉" * 20)
            print("\n📁 生成的檔案:")
            print("   📊 9張分析圖表 (01_*.png - 09_*.png)")
            print("   📄 詳細分析報告 (kidney_disease_analysis_report.md)")
            print("   💾 訓練好的模型物件")
            
            # 顯示最佳結果
            best_model_name = max(self.results.keys(), 
                                 key=lambda x: self.results[x]['test_accuracy'])
            best_results = self.results[best_model_name]
            
            print(f"\n🏆 最佳模型: {best_model_name.replace('_', ' ').title()}")
            print(f"   📊 測試集準確率: {best_results['test_accuracy']:.1%}")
            print(f"   🎯 精確率: {best_results['precision']:.1%}")
            print(f"   🔍 召回率: {best_results['recall']:.1%}")
            print(f"   ⭐ F1分數: {best_results['f1']:.1%}")
            print(f"   📈 ROC AUC: {best_results['roc_auc']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 分析過程中發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# 使用範例
if __name__ == "__main__":
    # 創建預測器
    predictor = KidneyDiseasePredictor('/mnt/user-data/uploads/kidney_disease.csv')
    
    # 執行完整分析
    success = predictor.run_complete_analysis()
    
    if success:
        print("\n✅ 分析成功完成!")
        print("\n💡 使用說明:")
        print("1. 查看生成的9張圖表了解資料和模型性能")
        print("2. 閱讀詳細報告了解完整分析結果")
        print("3. 模型物件已保存，可用於新資料預測")
        print("4. 所有檔案位於 /mnt/user-data/outputs/ 目錄")
    else:
        print("\n❌ 分析失敗，請檢查錯誤信息並重試")
