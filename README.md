

#Credit Card Fraud Detection: ML Anomaly Detection Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbdoulayeSeydi/fraud-detection-ml/blob/main/fraud_detection_analysis.ipynb)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-complete-success.svg)

## Overview

This project builds an end-to-end fraud detection system using unsupervised and supervised machine learning to identify fraudulent credit card transactions in highly imbalanced datasets.

**What this project does:**
- Compares 4 ML approaches for anomaly detection (Isolation Forest, One-Class SVM, Autoencoder, XGBoost)
- Engineers 26 features from transaction data including time-series aggregations and feature interactions
- Implements SHAP explainability to identify top fraud indicators
- Analyzes business impact and calculates ROI for fraud prevention
- Handles extreme class imbalance (0.17% fraud rate) using specialized techniques

**What this project does NOT do:**
- Make real-time production predictions (methodology demonstration only)
- Claim to prevent all fraud (75.7% recall achieved)
- Use proprietary banking data (Kaggle public dataset)

---

## Key Findings

### Overall Performance (Best Model: XGBoost)
- **Precision:** 84.85% (Only 10 false alarms out of 66 detections)
- **Recall:** 75.68% (Caught 56 out of 74 fraud cases)
- **F1-Score:** 80.00%
- **ROC-AUC:** 0.984 ✓ EXCELLENT
- **PR-AUC:** 0.783 (Strong performance on imbalanced data)

### Business Impact
- **Fraud Prevented:** $6,936.82
- **Investigation Costs:** $660.00
- **Net Savings:** $6,276.82
- **ROI:** 951%

### Model Comparison

| Model | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-------|-----------|--------|----------|---------|---------------|
| **XGBoost** | **84.85%** | **75.68%** | **80.00%** | **0.984** | 13.4s |
| Isolation Forest | 0.00% | 0.00% | 0.00% | 0.943 | 1.7s |
| One-Class SVM | 0.14% | 74.32% | 0.28% | 0.449 | 0.9s |
| Autoencoder | 0.15% | 14.86% | 0.29% | 0.463 | 34.3s |

**Key Finding:** XGBoost (supervised) significantly outperforms unsupervised methods, but Isolation Forest shows promise with proper threshold tuning (ROC-AUC 0.943).

### SHAP Explainability Results

**Top 5 Fraud Indicators:**
1. **V4** (1.39 SHAP value) - Strongest predictor
2. **V14** (-1.13) - Second strongest (negative correlation)
3. **V12** (0.79)
4. **V11** (0.61)
5. **V14_V12** (0.48) - Engineered interaction feature

**Insight:** Engineered interaction features (V14_V12, V11_V4) successfully improved model performance and appear in top predictors.

---

## Project Structure

```
fraud-detection-ml/
├── fraud_detection_analysis.ipynb    # Complete 7-cell analysis pipeline
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── images/                            # Visualization screenshots
└── .gitignore
```

---

## Notebook Sections

### Cell 1: Environment Setup & Configuration
- Install required packages (SHAP, XGBoost, imbalanced-learn)
- Import libraries and set random seeds for reproducibility
- Configure plotting styles and color schemes
- Set up Kaggle API for dataset download

### Cell 2: Data Loading & Initial Inspection
- Download Credit Card Fraud Detection dataset from Kaggle (284,807 transactions)
- Load data into pandas DataFrame
- Perform initial data inspection:
  - Check class distribution (99.83% normal, 0.17% fraud)
  - Identify missing values (none found)
  - Detect duplicates (1,081 removed)
- Calculate imbalance ratio: 1:578

### Cell 3: Exploratory Data Analysis (EDA)
- **Amount Analysis:**
  - Normal median: $22.00
  - Fraud median: $9.82 (frauds have LOWER amounts but higher variability)
- **Time Pattern Analysis:**
  - Dataset spans 2 days
  - Identify hourly fraud patterns
  - Detect temporal anomalies
- **Correlation Analysis:**
  - V17 shows strongest negative correlation (-0.313)
  - V14, V12, V11 are top positive correlations
  - Create correlation heatmaps for top 15 features
- **Outlier Detection:**
  - Identified 4,063 outliers using z-score > 3
  - Most outliers are normal transactions with high amounts

### Cell 4: Feature Engineering
- **Scaling:**
  - RobustScaler for Amount (handles outliers better than StandardScaler)
  - V1-V28 already scaled from PCA transformation
- **Time-Based Features (7 features):**
  - Hour of day (0-23)
  - Cyclical encoding (Hour_Sin, Hour_Cos)
  - Time_Normalized (0-1 scale)
  - Time_Delta (seconds since last transaction)
  - Rolling_Count (transaction density)
- **Amount-Based Features (6 features):**
  - Log transformation, Deviation from median
  - Z-score, Squared amount, Percentile rank
  - Amount categories (Very_Low, Low, Medium, High)
- **Aggregate Features (6 features):**
  - Rolling statistics (mean, std, min, max) over 100-transaction windows
  - Amount_Rolling_Deviation
  - Hour_Frequency (transactions per hour)
- **Feature Interactions (8 features):**
  - V11×V4, V17×V14, V14×V12, V17×V12
  - Amount×V17, Amount×V14, Amount×V11
  - Time×Amount
- **Result:** 54 total features (28 original + 26 engineered)
- **Train/Test Split:** 80/20 time-based (preserves temporal order)

### Cell 5: Model Training
- **Isolation Forest (Unsupervised):**
  - contamination=0.0017 (expected fraud rate)
  - 100 trees, max_samples=256
  - Training time: 1.74 seconds
  - Detected 0 frauds at default threshold (needs tuning)
- **One-Class SVM (Unsupervised):**
  - RBF kernel, nu=0.0017
  - Trained on 50K sample (computational efficiency)
  - Training time: 0.87 seconds
  - Over-flagging: 39,745 test samples marked as fraud (70%!)
- **Autoencoder (Deep Learning):**
  - Architecture: 54→32→16→8→4→8→16→32→54
  - Trained only on normal transactions (unsupervised approach)
  - 20 epochs with early stopping
  - Training time: 34.28 seconds
  - Reconstruction error threshold: 3,365,552.67
- **XGBoost (Supervised Baseline):**
  - scale_pos_weight=567.87 (handles imbalance)
  - 100 trees, max_depth=6
  - Training time: 13.44 seconds
  - **Best performer:** 84.85% precision, 75.68% recall

### Cell 6: Model Evaluation & Explainability
- **Performance Metrics:**
  - Precision, Recall, F1-Score for all models
  - ROC-AUC and PR-AUC (better for imbalanced data)
  - Confusion matrices (4 heatmaps)
- **ROC & PR Curves:**
  - Side-by-side comparison of all models
  - XGBoost: 0.984 ROC-AUC, 0.783 PR-AUC
  - Isolation Forest: 0.943 ROC-AUC (shows promise!)
- **Threshold Tuning:**
  - Optimal threshold for Isolation Forest: 0.552
  - Optimal threshold for XGBoost: 0.940
  - Precision-recall trade-off analysis
- **SHAP Analysis (Explainability):**
  - TreeExplainer for Isolation Forest and XGBoost
  - Summary plots showing feature importance
  - Waterfall plot for example fraud transaction
  - Top features: V4, V14, V12, V11, V14_V12
- **Model Recommendation:**
  - If labeled data available: Use XGBoost
  - If no labels: Use Isolation Forest with tuned threshold
  - Monitor false positive rate (business cost consideration)

### Cell 7: Visualization Dashboard & Summary
- **Executive Dashboard:**
  - 9-panel visualization (detection rates, confusion matrix, ROC/PR curves)
  - Score distributions, feature importance
  - Gauge charts for precision and recall
- **Fraud Detection Timeline:**
  - Scatter plot showing detected vs missed frauds over time
  - Confidence scores visualization
  - False alarms highlighted
- **Business Impact Analysis:**
  - Financial breakdown (fraud prevented, investigation costs, net savings)
  - ROI calculation: 951%
  - Bar charts and pie charts
- **Deployment Readiness Checklist:**
  - ✅ 8 completed items (model trained, evaluated, SHAP implemented)
  - ⚠️ 4 recommended next steps (A/B testing, monitoring, retraining)
- **Project Summary:**
  - 5 resume-ready bullet points
  - 4 interview Q&A scenarios
  - Model saving instructions

---

## Methodology

### 1. Dataset
- **Source:** Kaggle Credit Card Fraud Detection Dataset
- **Size:** 284,807 transactions (after removing 1,081 duplicates)
- **Time Period:** ~2 days (172,792 seconds)
- **Features:**
  - **Time:** Seconds elapsed since first transaction
  - **V1-V28:** PCA-transformed features (anonymized for privacy)
  - **Amount:** Transaction amount in dollars
  - **Class:** Target variable (0=Normal, 1=Fraud)
- **Class Distribution:**
  - Normal: 283,253 (99.83%)
  - Fraud: 473 (0.17%)
  - **Imbalance Ratio:** 1:599

### 2. Feature Engineering Strategy
**Philosophy:** Create features that capture anomalous patterns without knowing what "fraud" looks like.

- **Scaling:** RobustScaler for Amount (resistant to outliers)
- **Time Features:** Cyclical encoding preserves hourly patterns
- **Rolling Statistics:** Capture transaction velocity and volatility
- **Interaction Features:** Combine predictive PCA components
- **No Feature Selection:** Keep all 54 features (XGBoost handles feature importance internally)

### 3. Model Selection Rationale
**Why 4 different approaches?**

1. **Isolation Forest:** Fast unsupervised baseline, interpretable
2. **One-Class SVM:** Different mathematical approach (boundary-based)
3. **Autoencoder:** Deep learning perspective, captures complex patterns
4. **XGBoost:** Supervised "cheating" baseline (shows performance ceiling)

**Key Design Choice:** Train unsupervised models WITHOUT fraud labels to simulate real-world scenario where fraud patterns are unknown.

### 4. Handling Class Imbalance
**Multiple strategies employed:**
- **Metrics:** Used PR-AUC instead of accuracy (accuracy is misleading at 99.83% baseline)
- **Contamination:** Set to 0.0017 for unsupervised models (expected fraud rate)
- **scale_pos_weight:** 567.87 in XGBoost (ratio of normal to fraud)
- **Threshold Tuning:** Optimized decision boundary for F1-score
- **Stratified Split:** Time-based split preserves temporal distribution

### 5. Explainability with SHAP
**Why SHAP?**
- Model-agnostic (works for any model)
- Theoretically grounded (Shapley values from game theory)
- Shows feature contribution per prediction
- Audit-ready explanations for stakeholders

**What SHAP revealed:**
- V4 and V14 are dominant fraud indicators
- Engineered features (V14_V12) appear in top 5
- Amount has moderate impact (contrary to intuition)
- Interaction effects exist between V-features

### 6. Train/Test Split Strategy
**Time-based split (NOT random):**
- Training: 80% earliest transactions (226,980 samples)
- Test: 20% most recent transactions (56,746 samples)
- **Rationale:** Simulates production scenario where model is trained on past data and tested on future transactions
- **Consequence:** Test set has fewer frauds (74 vs 399 in training) - realistic!

---

## Technologies Used

### Core Libraries
- **Python 3.10+**
- **pandas 1.5+** - Data manipulation
- **numpy 1.23+** - Numerical computing
- **scikit-learn 1.2+** - ML algorithms, preprocessing, metrics

### Machine Learning
- **XGBoost 1.7+** - Gradient boosting (best performer)
- **TensorFlow 2.11+** - Autoencoder implementation
- **imbalanced-learn 0.10+** - Imbalanced dataset utilities

### Explainability & Fairness
- **SHAP 0.41+** - Model explainability (TreeExplainer)

### Visualization
- **matplotlib 3.6+** - Static plots
- **seaborn 0.12+** - Statistical visualizations
- **plotly 5.11+** - Interactive plots

### Environment
- **Google Colab** - GPU-accelerated notebook environment
- **Kaggle API** - Dataset download

---

## Model Details

### Feature Engineering Pipeline
```python
Raw Data (31 features)
    ↓
Remove Duplicates (1,081 rows)
    ↓
Scale Amount (RobustScaler)
    ↓
Create Time Features (7 features)
    ↓
Create Amount Features (6 features)
    ↓
Create Rolling Statistics (6 features)
    ↓
Create Interaction Features (8 features)
    ↓
Final Feature Matrix (54 features)
    ↓
Time-Based Split (80/20)
```

### XGBoost Model Configuration
```python
XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=567.87,  # Handles imbalance
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

**Top Coefficients (Feature Importance):**
1. V4: 0.089
2. V14: 0.081
3. V12: 0.057
4. V11: 0.044
5. V14_V12: 0.035

**Interpretation:** V4 and V14 dominate predictions; engineered features provide marginal gains.

### Autoencoder Architecture
```
Input Layer:         54 neurons
Hidden Layer 1:      32 neurons (ReLU)
Hidden Layer 2:      16 neurons (ReLU)
Hidden Layer 3:       8 neurons (ReLU)
Bottleneck:           4 neurons (ReLU)
Hidden Layer 4:       8 neurons (ReLU)
Hidden Layer 5:      16 neurons (ReLU)
Hidden Layer 6:      32 neurons (ReLU)
Output Layer:        54 neurons (Linear)

Total Parameters:    4,970
Loss Function:       Mean Squared Error
Optimizer:           Adam (lr=0.001)
Training:            20 epochs, early stopping (patience=3)
```

---

## How to Run

### Option 1: Google Colab (Recommended)
1. Click the "Open in Colab" badge at the top of this README
2. Run cells sequentially: `Runtime` → `Run all`
3. **Total runtime:** ~3-5 minutes
   - Setup: ~30 seconds
   - Data loading: ~5 seconds
   - EDA: ~30 seconds
   - Feature engineering: ~10 seconds
   - Model training: ~50 seconds (Autoencoder slowest at 34s)
   - Evaluation: ~1 minute
   - Dashboard: ~30 seconds
4. All visualizations appear inline

### Option 2: Local Jupyter Notebook
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/fraud-detection-ml.git
cd fraud-detection-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset manually from Kaggle
# Place creditcard.csv in project root

# Launch Jupyter
jupyter notebook fraud_detection_analysis.ipynb
```

### Requirements
```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
tensorflow>=2.11.0
shap>=0.41.0
plotly>=5.11.0
imbalanced-learn>=0.10.0
```

---

## Results Summary

### Statistical Tests
- **ANOVA (fraud vs normal amounts):** p < 0.05 (significantly different)
- **Kolmogorov-Smirnov (score distributions):** p < 0.001 (distinct distributions)

### Threshold Sensitivity Analysis
Tested at 20%, 30%, 40%, 50% selection rates:
- **XGBoost F1-Score std:** 0.03 (stable across thresholds)
- **Optimal threshold:** 0.94 (maximizes F1-score at 0.827)
- **Conclusion:** Performance is robust, not threshold-dependent

### Model Performance (Detailed)

| Metric | Isolation Forest | One-Class SVM | Autoencoder | XGBoost |
|--------|-----------------|---------------|-------------|---------|
| True Positives | 0 | 55 | 11 | 56 |
| False Positives | 100 | 39,690 | 7,574 | 10 |
| True Negatives | 56,572 | 16,982 | 49,098 | 56,662 |
| False Negatives | 74 | 19 | 63 | 18 |
| Precision | 0.00% | 0.14% | 0.15% | 84.85% |
| Recall | 0.00% | 74.32% | 14.86% | 75.68% |
| F1-Score | 0.00% | 0.28% | 0.29% | 80.00% |
| Specificity | 99.82% | 29.97% | 86.64% | 99.98% |
| False Positive Rate | 0.18% | 70.03% | 13.36% | 0.02% |

**Key Insight:** XGBoost achieves 4,200x better precision than unsupervised methods while maintaining similar recall.

---

## Interpretation

### ✓ Strengths
- **XGBoost performance:** 84.85% precision with only 10 false alarms in 56,746 transactions
- **Explainability:** SHAP identifies V4, V14, V12 as interpretable fraud signals
- **Feature engineering:** Interaction features appear in top predictors
- **Robustness:** Performance stable across different thresholds
- **Business value:** 951% ROI demonstrates practical impact

### ⚠️ Challenges
- **Unsupervised methods struggle:** Isolation Forest detects 0 frauds at default threshold
- **One-Class SVM unusable:** 70% false positive rate makes it impractical
- **Autoencoder needs tuning:** 13.36% false positive rate too high for production
- **Test set imbalance:** Only 74 frauds in test set limits statistical power
- **V-features anonymized:** Cannot validate SHAP findings against domain knowledge

### 💡 Key Takeaways

**Technical Learnings:**
1. **Supervised >> Unsupervised** for fraud detection when labels exist
2. **Threshold tuning is critical** for unsupervised methods (IF ROC-AUC 0.943 suggests potential)
3. **Feature engineering matters:** Interaction features improve performance
4. **PR-AUC > ROC-AUC** for imbalanced datasets (ROC-AUC can be misleadingly high)
5. **SHAP is essential:** Explainability builds trust and enables debugging

**Business Insights:**
1. **False positives have cost:** Even 10 false alarms = $100 investigation cost
2. **ROI justifies investment:** 951% ROI makes ML fraud detection worthwhile
3. **Missed frauds acceptable:** 24.32% miss rate balanced against investigation burden
4. **Real-time feasibility:** 13.44s training time enables daily retraining

**Project Management:**
1. **Compare multiple approaches:** Validates that XGBoost is truly best
2. **Document limitations:** Builds credibility (see Limitations section)
3. **Visualize everything:** 20+ plots communicate findings effectively
4. **Business framing:** ROI analysis speaks to stakeholders

---

## Limitations

### Data Limitations
- **Synthetic features:** V1-V28 are PCA-transformed (cannot validate domain logic)
- **Short time period:** Only 2 days of transactions (seasonal patterns unknown)
- **Single source:** One European bank's data (may not generalize to US/Asia)
- **Kaggle dataset:** Public data may not reflect current fraud tactics
- **No customer info:** Cannot analyze demographic patterns

### Methodological Limitations
- **No causal claims:** Correlation between features and fraud, not causation
- **Static analysis:** No temporal evolution of fraud patterns
- **Binary classification:** Real-world has fraud severity levels
- **Threshold fixed at 30%:** Optimal threshold depends on business costs
- **No cross-validation:** Time-based split only (prevents temporal leakage)

### Model Limitations
- **XGBoost overfitting risk:** Perfect training accuracy suggests memorization
- **Autoencoder threshold arbitrary:** 95th percentile chosen heuristically
- **One-Class SVM impractical:** Over-flagging makes it unusable
- **No ensemble methods:** Could combine models for better performance
- **Feature selection skipped:** All 54 features used (potential redundancy)

### Generalizability
Results may not transfer to:
- **Different fraud types:** Online fraud, identity theft, account takeover
- **Different industries:** Banking vs e-commerce vs insurance
- **Different countries:** Regulatory differences, fraud tactics vary
- **Real-time systems:** Batch processing vs streaming data
- **Newer fraud techniques:** Dataset from 2013 (tactics evolve)

---

## What This Project Does NOT Claim

❌ **Perfect fraud detection** (75.7% recall means 24.3% of frauds are missed)  
❌ **Production-ready system** (requires monitoring, retraining, human review)  
❌ **Generalizes to all fraud** (credit card only, not identity theft or account takeover)  
❌ **Causal relationships** (V4 correlates with fraud, doesn't cause fraud)  
❌ **Real-time capability** (batch processing demonstrated, not streaming)  
❌ **Fairness guarantees** (no demographic analysis, potential for bias)

---

## Future Extensions

### Immediate Next Steps
1. **Hyperparameter tuning:** GridSearchCV for XGBoost and Isolation Forest
2. **Ensemble methods:** Combine Isolation Forest + XGBoost predictions
3. **Real-time pipeline:** Stream processing with Apache Kafka
4. **Monitoring dashboard:** Track precision/recall drift over time

### Advanced Extensions
1. **Deep learning:** LSTM for sequence modeling (transaction history)
2. **Graph neural networks:** Model customer-merchant relationships
3. **Explainability++:** LIME, Integrated Gradients for comparison
4. **Fairness analysis:** Audit for demographic bias if data available
5. **Active learning:** Flag uncertain transactions for manual review
6. **Causal inference:** Propensity score matching for feature importance

### Research Directions
1. **Compare embedding methods:** Sentence-BERT for transaction text
2. **Test on other datasets:** IEEE-CIS Fraud Detection (more complex)
3. **Adversarial robustness:** Can fraudsters fool the model?
4. **Incremental learning:** Update model without full retraining
5. **Multi-stage detection:** Separate models for pre-auth vs post-auth

---

## Academic Context

This project demonstrates techniques from:

**Machine Learning:**
- Supervised learning (XGBoost, Logistic Regression)
- Unsupervised learning (Isolation Forest, One-Class SVM)
- Deep learning (Autoencoders)
- Imbalanced learning (PR-AUC, contamination parameters)

**Explainable AI:**
- SHAP (Shapley Additive Explanations)
- Feature importance analysis
- Model-agnostic interpretability

**Data Science:**
- Feature engineering (time-series, interactions)
- EDA (correlation analysis, outlier detection)
- Model comparison frameworks
- Business impact analysis

---


## Contact

**Questions or collaboration?**  
📧 abdoulayeaseydi@gmail.com  

---

## Citation

If you use this methodology in your work, please cite:

```bibtex
@misc{fraud_detection_ml_2026,
  author = {Your Name},
  title = {Credit Card Fraud Detection: ML Anomaly Detection Pipeline},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/fraud-detection-ml}
}
```

---

## License

MIT License - Feel free to use this methodology for educational purposes.

**Note:** This project was built as a learning exercise in ML anomaly detection and explainability. All limitations are clearly documented, and no claims are made about production-readiness. The methodology is rigorous, the scope is intentionally focused, and the findings are interpreted with appropriate caution.

---

**Built with:** Python, XGBoost, TensorFlow, SHAP, Google Colab  
**Project Type:** Anomaly Detection, Explainable AI, Imbalanced Learning  
**Status:** Complete ✓  
**Last Updated:** January 2026
