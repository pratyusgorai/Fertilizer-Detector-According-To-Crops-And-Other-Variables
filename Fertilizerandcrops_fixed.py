#!/usr/bin/env python
# coding: utf-8

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ── Load Data ─────────────────────────────────────────────────────────────────
data = pd.read_csv('fertilizer_recommendation_dataset.csv')

# ── EDA ───────────────────────────────────────────────────────────────────────
print(data.head())
print("Shape:", data.shape)
data.info()
print(data.describe())
print("Null values:\n", data.isnull().sum())
print("Duplicate rows:", data.duplicated().sum())
print(data.tail(20))

# ── Outlier Removal (Rainfall) ────────────────────────────────────────────────
# FIX: added plt.show() to both boxplots so they actually render
plt.figure()
sns.boxplot(data['Rainfall'])
plt.title("Rainfall Before Outlier Removal")
plt.show()

min_range = data["Rainfall"].mean() - (3 * data["Rainfall"].std())
max_range = data["Rainfall"].mean() + (3 * data["Rainfall"].std())
print(f"Rainfall range: {min_range:.2f} – {max_range:.2f}")

data = data[(data["Rainfall"] >= min_range) & (data["Rainfall"] <= max_range)]

plt.figure()
sns.boxplot(data["Rainfall"])
plt.title("Rainfall After Outlier Removal")
plt.show()

# ── Correlation Heatmap ───────────────────────────────────────────────────────
corr_matrix = data.select_dtypes(include=['float64']).corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap of Feature Correlations")
plt.show()

# ── Crop vs Fertilizer Line Plot ──────────────────────────────────────────────
# FIX: set figsize BEFORE plotting, not after plt.show()
# NOTE: plotted on string data BEFORE encoding so axis labels are readable
plt.figure(figsize=(20, 15))
plt.xticks(rotation=67, ha="right")
sns.lineplot(y="Fertilizer", x="Crop", data=data)
plt.xlabel("Crops")          # FIX: was "Corps" (typo)
plt.ylabel("Fertilizers")
plt.title("Crop vs Fertilizer")
plt.tight_layout()
plt.show()

# ── Label Encoding ────────────────────────────────────────────────────────────
label_encoders = {}
categorical_columns = ['Soil', 'Crop', 'Fertilizer']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

print("Sample encoded values:")
print(data['Fertilizer'].tail(50))
print(data['Soil'].head(20))
print(data['Crop'].tail(20))

# ── Feature / Target Split ────────────────────────────────────────────────────
# FIX: keep x_dt separate so SVM doesn't overwrite it
x_dt = data.drop(columns=['Fertilizer', 'Remark'])
y_dt = data["Fertilizer"]

# ── Decision Tree ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    x_dt, y_dt, test_size=0.2, random_state=42
)
model1 = DecisionTreeClassifier()
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)

# FIX: removed duplicate accuracy_score line
accuracy = accuracy_score(y_test, y_pred)
print(f'Decision Tree Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree – Confusion Matrix')
plt.show()

# Feature Importances
importances = model1.feature_importances_
print("Feature Importances:")
for feature, importance in zip(x_dt.columns, importances):
    print(f'  {feature}: {importance:.4f}')

# ── SVM ───────────────────────────────────────────────────────────────────────
# FIX: use separate variable names (x_svm) so X_test/y_test remain intact
x_svm = data.drop(columns=['Fertilizer', 'Remark'])
y_svm = data['Fertilizer']
x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(
    x_svm, y_svm, test_size=0.2, random_state=42
)
model2 = SVC(kernel='rbf', C=1.0, gamma='scale')
model2.fit(x_train_s, y_train_s)
y_pred_s = model2.predict(x_test_s)

accuracy_s = accuracy_score(y_test_s, y_pred_s)
print(f'SVM Accuracy: {accuracy_s * 100:.2f}%')
print(classification_report(y_test_s, y_pred_s))

cm_s = confusion_matrix(y_test_s, y_pred_s)
plt.figure()
sns.heatmap(cm_s, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM – Confusion Matrix')
plt.show()

# ── Recommendation Function ───────────────────────────────────────────────────
def recommend_fertilizer(soil, crop, nitrogen, phosphorus, potassium,
                         temperature, carbon, moisture, pH, rainfall):
    soil_encoded = label_encoders['Soil'].transform([soil])[0]
    crop_encoded = label_encoders['Crop'].transform([crop])[0]
    # Feature order matches training columns exactly:
    # [Temperature, Moisture, Rainfall, pH, Nitrogen, Phosphorus, Potassium, Carbon, Soil, Crop]
    input_data = np.array([[temperature, moisture, rainfall, pH,
                            nitrogen, phosphorus, potassium, carbon,
                            soil_encoded, crop_encoded]])
    fertilizer_code = model1.predict(input_data)[0]
    fertilizer_name = label_encoders['Fertilizer'].inverse_transform([fertilizer_code])[0]
    return fertilizer_name

def check_prediction_accuracy():
    # FIX: X_test/y_test now reliably refer to the Decision Tree split
    y_pred_check = model1.predict(X_test)
    acc = accuracy_score(y_test, y_pred_check)
    print(f'Model Accuracy: {acc * 100:.2f}%')

# ── Test Recommendation ───────────────────────────────────────────────────────
# FIX: argument order in the CALL now matches the function SIGNATURE
soil_type   = 'Loamy Soil'
crop_type   = 'wheat'
nitrogen    = 50
phosphorus  = 30
potassium   = 20
temperature = 25
carbon      = 1.5
moisture    = 60
pH          = 6.5
rainfall    = 200

recommended_fertilizer = recommend_fertilizer(
    soil_type, crop_type,
    nitrogen, phosphorus, potassium,
    temperature, carbon, moisture, pH, rainfall   # ← fixed order
)
print(f'Recommended Fertilizer: {recommended_fertilizer}')
check_prediction_accuracy()

# ── Save Model & Encoders ─────────────────────────────────────────────────────
joblib.dump(model1, 'Fertilizer_Prediction_APP')
joblib.dump(label_encoders, 'label_encoders.pkl')   # FIX: encoders were never saved before
print("Model and encoders saved.")

# ── Verify Saved Model ────────────────────────────────────────────────────────
app = joblib.load('Fertilizer_Prediction_APP')
array = [[20.939817, 0.640556, 37.361457, 5.182518,
          42.242562, 36.750004, 39.370552, -0.280560, 15, 2]]
y_pre = app.predict(array)
fertilizer_name = label_encoders['Fertilizer'].inverse_transform(y_pre)
print(f'Verified Prediction: {fertilizer_name[0]}')

# ── Print Valid Label Options ─────────────────────────────────────────────────
soil_encoded_vals = data['Soil'].unique()
soil_names = label_encoders['Soil'].inverse_transform(sorted(soil_encoded_vals))
print("Valid Soil Types:", soil_names)

fertilizer_vals = data['Fertilizer'].unique()
ferti_names = label_encoders['Fertilizer'].inverse_transform(sorted(fertilizer_vals))
print("Valid Fertilizers:", ferti_names)

print(data.head(10))
