# Employee Attrition Prediction Using Machine Learning
## Application Link: https://huggingface.co/spaces/Nishal19/Customer_Attrition-END_TO_END_1
## Overview

This project predicts employee attrition ("Yes"/"No") based on HR and demographic data. The goal is to build robust classification models for imbalanced datasets and identify the best performing approach for HR analytics.

---

## Dataset

- **Total Records**: 1,483 employees
- **Class Distribution**: 
  - No Attrition: 1,233 (83.1%)
  - Attrition: 250 (16.9%)
- **Challenge**: Highly imbalanced dataset requiring special handling

---

## Data Preprocessing

### 1. Skewness Correction
- Identified numeric features with high skewness (|skew| > 0.5)
- Applied **PowerTransformer** (Yeo-Johnson method) to reduce skew and normalize feature distributions
- This improves model learning by making distributions more Gaussian

### 2. Outlier Handling
- Used **Interquartile Range (IQR)** method to detect outliers
- Capped values outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` to respective boundaries
- Ensures stable model training without extreme values

### 3. Categorical Encoding
- Encoded all categorical features using **LabelEncoder**
- Made features compatible with ML algorithms

### 4. Class Balancing (Training Data Only)
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance training data
- SMOTE creates synthetic samples of the minority class (attrition cases)
- **Critical**: Applied only to training data to avoid data leakage
- Prevents model bias toward majority class

### 5. Feature Scaling
- Applied **StandardScaler** to normalize numeric features
- Ensures fair comparison among features for distance-based algorithms

---

## Feature Selection

- Used **Random Forest feature importances** to rank predictive power
- Selected **top 15 features** for final models:
  - Reduces noise and overfitting
  - Improves model interpretability
  - Faster training and prediction

---

## Model Building & Evaluation

### Classification Algorithms Tested:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.741 | 0.307 | 0.489 | 0.377 |
| Decision Tree | 0.731 | 0.258 | 0.362 | 0.301 |
| Random Forest | 0.827 | 0.417 | 0.213 | 0.282 |
| SVM | 0.816 | 0.426 | 0.426 | 0.426 |
| KNN | 0.554 | 0.191 | 0.553 | 0.284 |
| Naive Bayes | 0.656 | 0.250 | 0.574 | 0.348 |
| **XGBoost** | **0.854** | **0.583** | 0.298 | **0.394** |

### Hyperparameter Tuning
- Performed **GridSearchCV** with 5-fold cross-validation for each model
- Optimized parameters like:
  - Tree-based: `n_estimators`, `max_depth`, `learning_rate`
  - Regression: `C`, `penalty`, `solver`
  - Distance-based: `n_neighbors`, `weights`
- Evaluated on **precision, recall, and F1-score** (critical for imbalanced data)

---

## Model Selection: XGBoost

### Why XGBoost?

✅ **Highest Accuracy (0.854)**: Best overall performance  
✅ **Highest Precision (0.583)**: When predicting attrition, it's most often correct  
✅ **Best F1-Score (0.394)**: Best balance between precision and recall  
✅ **Robust to Imbalance**: Handles class imbalance well with proper tuning  
✅ **Feature Importance**: Provides interpretable insights for HR decision-making  
✅ **Generalization**: Strong cross-validation performance

### Trade-offs:
- Lower recall (0.298) compared to some models means it misses some attrition cases
- However, high precision means fewer false alarms for HR teams
- Overall, XGBoost provides the best balance for business use

---

## Key Insights

### For Imbalanced Classification Problems:
- **Don't rely on accuracy alone** - it's misleading when classes are imbalanced
- **F1-score and precision/recall** are better metrics for minority class
- **SMOTE should only be applied to training data** to prevent data leakage
- **Feature selection** reduces noise and improves model robustness

### Preprocessing Impact:
- Skewness correction and outlier handling significantly improved model stability
- Feature scaling was essential for distance-based and gradient-based algorithms
- Proper encoding and balancing were critical for capturing attrition patterns

---

## Project Structure

```
Employee_attrition-project-1/
├── Employee-Attrition.csv       # Dataset
├── XGBoost.joblib              # Trained XGBoost model
├── scaler.joblib               # Fitted StandardScaler
├── label_encoders.joblib       # Label encoders for categorical features
├── top_features.joblib         # Selected feature names
├── app.py                      # Gradio web application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/NISHAL2007/Employee_attrition-project-1.git
cd Employee_attrition-project-1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Gradio App
```bash
python app.py
```

### 4. Make Predictions
- Open the Gradio interface in your browser
- Input employee features
- Get attrition prediction (Yes/No)

---

## Technologies Used

- **Python 3.x**
- **scikit-learn**: ML algorithms, preprocessing, evaluation
- **XGBoost**: Gradient boosting classifier
- **imbalanced-learn**: SMOTE for class balancing
- **Gradio**: Web interface for model deployment
- **joblib**: Model serialization
- **pandas & numpy**: Data manipulation

---

## Future Improvements

- Threshold tuning to optimize recall for catching more attrition cases
- Ensemble methods combining multiple models
- SHAP values for better model explainability
- Real-time prediction API
- Dashboard for HR analytics

---

## License

MIT License

---

## Author

**NISHAL2007**

For questions or collaboration, feel free to reach out!
