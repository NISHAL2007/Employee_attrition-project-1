import gradio as gr
import joblib
import numpy as np

# Load saved artifacts
model = joblib.load("XGBoost.joblib")
scaler = joblib.load("scaler.joblib")
encoders = joblib.load("label_encoders.joblib")
top_features = joblib.load("top_features.joblib")

# For UI: Retrieve unique class values for each categorical feature
cat_features = list(encoders.keys())
cat_choices = {col: list(encoders[col].classes_) for col in cat_features}

def predict_func(
    StockOptionLevel, JobLevel, YearsWithCurrManager, JobInvolvement, MonthlyIncome, YearsAtCompany,
    JobSatisfaction, WorkLifeBalance, NumCompaniesWorked, TotalWorkingYears, Age,
    YearsInCurrentRole, DistanceFromHome, MonthlyRate, TrainingTimesLastYear
):
    # Build input in correct order
    inputs = [
        StockOptionLevel, JobLevel, YearsWithCurrManager, JobInvolvement, MonthlyIncome, YearsAtCompany,
        JobSatisfaction, WorkLifeBalance, NumCompaniesWorked, TotalWorkingYears, Age,
        YearsInCurrentRole, DistanceFromHome, MonthlyRate, TrainingTimesLastYear
    ]
    # Encode categorical features
    for i, feat in enumerate(top_features):
        if feat in cat_features:
            inputs[i] = encoders[feat].transform([inputs[i]])[0]
        else:
            inputs[i] = float(inputs[i])
    X = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return "Yes" if pred == 1 else "No"

# Build Gradio UI
inputs_list = []
for feature in top_features:
    if feature in cat_features:
        inputs_list.append(gr.Dropdown(choices=cat_choices[feature], label=feature))
    else:
        inputs_list.append(gr.Number(label=feature))

iface = gr.Interface(
    fn=predict_func,
    inputs=inputs_list,
    outputs=gr.Textbox(label="Attrition Prediction"),
    title="Employee Attrition Prediction"
)

if __name__ == "__main__":
    iface.launch()
