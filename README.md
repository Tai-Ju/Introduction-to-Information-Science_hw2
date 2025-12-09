# 🏥 Introduction to Information Science - Homework 2
## Chronic Kidney Disease Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### 📋 Project Overview

This project implements machine learning models to predict chronic kidney disease using clinical data. The analysis compares Decision Tree and Logistic Regression algorithms, addressing data leakage issues to ensure reliable predictions.

### 🎯 Objectives

- ✅ Build accurate kidney disease prediction models
- ✅ Compare different machine learning algorithms
- ✅ Identify the most important predictive features
- ✅ Provide interpretable medical prediction tools
- ✅ Address data leakage for reliable results

### 📊 Dataset Information

- **Samples**: 400 patients
- **Features**: 21 (after removing data leakage features)
- **Target**: Chronic kidney disease (binary classification)
- **Data Preprocessing**: Removed symptoms-based features (Anemia, Pedal Edema, Appetite)

### 🏆 Model Performance

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|---------------|-----------|--------|----------|---------|
| **Decision Tree** | **98.75%** | **100.0%** | **98.0%** | **99.0%** | **1.000** |
| Logistic Regression | 97.50% | 100.0% | 96.0% | 98.0% | 1.000 |

### 📁 Repository Structure

```
Introduction-to-Information-Science_hw2/
│
├── README.md                                    # Project documentation
├── kidney_disease_prediction_model2.ipynb      # Main analysis notebook
├── kidney_disease_analysis_local.ipynb         # Local analysis version
└── kidney_disease_analysis_presentation.pptx   # Project presentation slides
```

### 📚 Files Description

#### 🔬 **Main Analysis Notebooks**

1. **`kidney_disease_prediction_model2.ipynb`**
   - Complete machine learning pipeline
   - Data preprocessing and feature engineering
   - Model training and evaluation
   - Visualization and interpretation
   - **Recommended for main analysis**

2. **`kidney_disease_analysis_local.ipynb`**
   - Local analysis version
   - Alternative implementation approach
   - Supplementary analysis and experiments

#### 📈 **Presentation**

3. **`kidney_disease_analysis_presentation.pptx`**
   - Executive summary slides
   - Key findings and results
   - Model comparison and insights
   - Clinical implications

### 🛠️ Requirements

```python
# Core Data Science Libraries
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0

# Visualization
matplotlib >= 3.4.0
seaborn >= 0.11.0

# Jupyter Environment
jupyter >= 1.0.0
ipython >= 7.0.0
```

### 🚀 Getting Started

#### 1. Clone the Repository
```bash
git clone https://github.com/Tai-Ju/Introduction-to-Information-Science_hw2.git
cd Introduction-to-Information-Science_hw2
```

#### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

#### 3. Launch Jupyter Notebook
```bash
jupyter notebook
```

#### 4. Open Main Analysis
- Open `kidney_disease_prediction_model2.ipynb`
- Run all cells sequentially
- Explore the interactive analysis

### 📊 Key Features

#### ✨ **Advanced Data Preprocessing**
- **Data Leakage Resolution**: Removed symptom-based features
- **Feature Engineering**: Optimal feature selection
- **Data Validation**: Comprehensive quality checks

#### 🤖 **Machine Learning Pipeline**
- **Dual Algorithm Comparison**: Decision Tree vs Logistic Regression
- **Cross-Validation**: 5-fold stratified validation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Feature Importance Analysis**: Interpretable model insights

#### 📈 **Comprehensive Visualization**
- Target variable distribution
- Feature correlation analysis
- Confusion matrices comparison
- ROC curves and Precision-Recall curves
- Feature importance rankings
- Decision tree structure visualization

### 🔍 Key Findings

#### 🏆 **Model Performance**
- **Best Model**: Decision Tree (98.75% accuracy)
- **Zero False Positives**: 100% precision achieved
- **High Sensitivity**: 98% recall rate
- **Perfect Discrimination**: ROC AUC = 1.000

#### 🩺 **Medical Insights**
- **Most Predictive Features**:
  - Hemoglobin levels (blood indicator)
  - Specific Gravity (kidney function)
  - Serum Creatinine (kidney filtration)
- **Early Detection**: Models can identify high-risk patients before symptoms
- **Clinical Utility**: Suitable for screening and diagnostic support

#### 🔬 **Technical Achievements**
- **Data Leakage Resolution**: Improved model reliability
- **Algorithm Comparison**: Decision Tree outperforms Logistic Regression
- **Feature Interpretation**: Blood-related indicators are most crucial
- **Robust Validation**: Consistent performance across cross-validation

### 🏥 Clinical Applications

#### 💡 **Potential Use Cases**
- **Early Screening**: Identify high-risk patients in routine check-ups
- **Resource Allocation**: Optimize medical resource distribution
- **Decision Support**: Assist healthcare professionals in diagnosis
- **Preventive Care**: Enable proactive intervention strategies

#### ⚠️ **Important Limitations**
- Model trained on specific dataset - may need validation for different populations
- Should be used as diagnostic aid, not replacement for clinical judgment
- Requires regular model performance monitoring and updates
- Recommend combining with clinical expertise and additional tests

### 👨‍🎓 Academic Context

**Course**: Introduction to Information Science  
**Assignment**: Homework 2 - Machine Learning Application  
**Focus**: Healthcare Data Analysis and Predictive Modeling  
**Skills Demonstrated**:
- Data preprocessing and cleaning
- Machine learning model implementation
- Performance evaluation and validation
- Medical data interpretation
- Scientific presentation and documentation

### 📝 Usage Examples

#### Basic Prediction
```python
# Load the trained model
from kidney_disease_predictor import KidneyDiseasePredictor

# Create predictor instance
predictor = KidneyDiseasePredictor('kidney_disease.csv')

# Run complete analysis
predictor.run_complete_analysis()

# Make predictions on new data
results = predictor.predict_kidney_disease(new_patient_data)
print(f"Risk Level: {results['risk_level']}")
```

#### Model Comparison
```python
# Compare model performance
comparison = predictor.compare_models()
print(comparison)

# Generate visualizations
predictor.create_visualizations()
```

### 📖 Documentation

Each notebook contains detailed documentation including:
- **Step-by-step explanations** of the analysis process
- **Code comments** for clarity and understanding
- **Medical context** for feature interpretation
- **Performance analysis** and model validation
- **Clinical implications** of the results

### 🤝 Contributing

This is an academic project for educational purposes. For improvements or suggestions:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### 🙏 Acknowledgments

- Course instructor and teaching assistants
- Healthcare data science community
- Open-source machine learning libraries
- Medical professionals who provided domain expertise

### 📚 References

1. Chronic Kidney Disease Dataset - UCI Machine Learning Repository
2. Scikit-learn Documentation - Machine Learning in Python
3. Medical Literature on Kidney Disease Diagnosis
4. Best Practices in Healthcare Machine Learning

---

**⭐ If this project was helpful for your learning, please give it a star!**

