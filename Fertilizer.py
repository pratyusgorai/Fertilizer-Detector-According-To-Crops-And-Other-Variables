#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn



# In[2]:


data=pd.read_csv('fertilizer_recommendation_dataset.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data.duplicated()


# In[9]:


data.duplicated().sum()


# In[10]:


data.tail(20)


# In[11]:


sns.boxplot(data['Rainfall'])


# In[13]:


min_range = data["Rainfall"].mean()-(3*data["Rainfall"].std())
max_range = data["Rainfall"].mean()+(3*data["Rainfall"].std())


# In[14]:


min_range,max_range


# In[15]:


data = data[(data["Rainfall"] >= min_range) & (data["Rainfall"] <= max_range)]


# In[16]:


sns.boxplot(data["Rainfall"])


# In[17]:


corr_matrix = data.select_dtypes(include=['float64']).corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap of Feature Correlations")
plt.show()


# In[18]:


plt.xticks(rotation=67, ha="right")
sns.lineplot(y="Fertilizer",x="Crop",data=data)
plt.xlabel("Corps")
plt.ylabel("Fertilizers")
plt.figure(figsize=(20,15)) 
plt.show()


# In[19]:


from sklearn.preprocessing import LabelEncoder
label_encoders={}
categorical_columns = ['Soil','Crop','Fertilizer']
for col in categorical_columns:
    le=LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


# In[20]:


data['Fertilizer'].tail(50)


# In[21]:


data['Soil'].head(20)


# In[22]:


x = data.drop(columns=['Fertilizer','Remark'])
y = data["Fertilizer"]


# In[23]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=42)
model1 = DecisionTreeClassifier()
model1.fit(X_train,y_train)
y_pred = model1.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy:{accuracy * 100:.2f}%')
print(classification_report(y_test,y_pred))


# In[24]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[25]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[26]:


importances = model1.feature_importances_
feature_names = x.columns
print("Feature Importances:")
for feature, importance in zip(feature_names, importances):
    print(f'{feature}: {importance:.4f}')


# In[27]:


from sklearn.svm import SVC
x=data.drop(columns=['Fertilizer', 'Remark'])  # Drop the target variable & remarks
y=data['Fertilizer']
x_train, x_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model2 = SVC(kernel='rbf', C=1.0, gamma='scale')
model2.fit(x_train, Y_train)
Y_pred = model2.predict(x_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(Y_test, Y_pred))
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[28]:


def recommend_fertilizer(soil, crop, nitrogen, phosphorus, potassium, temperature,carbon, moisture, pH, rainfall):
    soil_encoded = label_encoders['Soil'].transform([soil])[0]
    crop_encoded = label_encoders['Crop'].transform([crop])[0]
    input_data = np.array([[temperature, moisture, rainfall, pH, nitrogen, phosphorus, potassium,carbon, soil_encoded, crop_encoded]])
    fertilizer_code = model1.predict(input_data)[0]
    fertilizer_name = label_encoders['Fertilizer'].inverse_transform([fertilizer_code])[0]
    return fertilizer_name
def check_prediction_accuracy():
    y_pred = model1.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
soil_type = 'Loamy Soil'
crop_type = 'wheat'
nitrogen = 50
phosphorus = 30
potassium = 20
carbon=1.5
temperature = 25
moisture = 60
pH = 6.5
rainfall = 200
recommended_fertilizer = recommend_fertilizer(soil_type, crop_type, nitrogen, phosphorus, potassium,carbon,temperature, moisture, pH, rainfall)
print(f'Recommended Fertilizer: {recommended_fertilizer}')
check_prediction_accuracy()


# In[29]:


import joblib
filename='Fertilizer_Prediction_APP'


# In[32]:


joblib.dump(model1,'Fertilizer_Prediction_APP')
app=joblib.load('Fertilizer_Prediction_APP')


# In[33]:


array=[[20.939817,0.640556,37.361457,5.182518,42.242562,36.750004,39.370552,-0.280560,15,2]]
y_pre=app.predict(array)
y_pre


# In[34]:


fertilizer_name = label_encoders['Fertilizer'].inverse_transform(y_pre)
print(f'Recommended Fertilizer: {fertilizer_name[0]}')


# In[35]:


fertilizer = data['Fertilizer'].unique()
ferti=label_encoders['Fertilizer'].inverse_transform(fertilizer)
print(ferti)


# In[36]:


data.head(10)


# In[37]:


import streamlit as st
import numpy as np

# Streamlit App
st.set_page_config(page_title="Fertilizer Recommendation System", page_icon="🌱", layout="centered")

st.title("Fertilizer Recommendation System 🌱")
st.markdown("Enter the soil and crop details to get the best fertilizer recommendation.")

# Layout with columns for better UI
col1, col2 = st.columns(2)

with col1:
    soil_type = st.selectbox("Soil Type", label_encoders['Soil'].classes_)
    crop_type = st.selectbox("Crop Type", label_encoders['Crop'].classes_)
    nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, value=50.0)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, value=30.0)
    potassium = st.number_input("Potassium (K)", min_value=0.0, value=20.0)

with col2:
    temp = st.number_input("Temperature (°C)", min_value=0.0, value=25.0)
    moisture = st.number_input("Moisture/Humidity", min_value=0.0, value=60.0)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=200.0)
    carbon = st.number_input("Organic Carbon", value=1.5)

if st.button("Predict Fertilizer"):
    try:
        # Encode categorical variables
        soil_encoded = label_encoders['Soil'].transform([soil_type])[0]
        crop_encoded = label_encoders['Crop'].transform([crop_type])[0]
        
        # Align features in the exact order model1 was trained on:
        # [Temperature, Moisture, Rainfall, pH, Nitrogen, Phosphorus, Potassium, Carbon, Soil, Crop]
        input_features = np.array([[temp, moisture, rainfall, ph, nitrogen, phosphorus, potassium, carbon, soil_encoded, crop_encoded]])
        
        # Predict using model1
        prediction_code = model1.predict(input_features)[0]
        
        # Decode the prediction
        fertilizer_name = label_encoders['Fertilizer'].inverse_transform([prediction_code])[0]
        
        st.success(f"### Recommended Fertilizer: {fertilizer_name}")
        
        # Displaying input summary for confirmation
        with st.expander("View Input Data"):
            st.write(f"**Soil:** {soil_type}, **Crop:** {crop_type}")
            st.write(f"**NPK:** {nitrogen}-{phosphorus}-{potassium}")
            st.write(f"**Environmental:** Temp: {temp}°C, pH: {ph}, Rainfall: {rainfall}mm")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Ensure all inputs are valid and the model is correctly trained.")


# In[ ]:


S

