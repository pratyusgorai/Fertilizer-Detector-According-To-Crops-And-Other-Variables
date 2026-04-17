#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn


# In[40]:


data=pd.read_csv('fertilizer_recommendation_dataset.csv')


# In[41]:


data.head()


# In[42]:


data.shape


# In[43]:


data.info()


# In[44]:


data.describe()


# In[45]:


data.isnull().sum()


# In[46]:


data.duplicated()


# In[47]:


data.duplicated().sum()


# In[48]:


data.tail(20)


# In[49]:


sns.boxplot(data['Rainfall'])


# In[50]:


min_range = data["Rainfall"].mean()-(3*data["Rainfall"].std())


# In[51]:


max_range = data["Rainfall"].mean()+(3*data["Rainfall"].std())


# In[52]:


min_range,max_range


# In[53]:


data = data[(data["Rainfall"] >= min_range) & (data["Rainfall"] <= max_range)]


# In[54]:


sns.boxplot(data["Rainfall"])


# In[55]:


corr_matrix = data.select_dtypes(include=['float64']).corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap of Feature Correlations")
plt.show()


# In[56]:


plt.xticks(rotation=67, ha="right")
sns.lineplot(y="Fertilizer",x="Crop",data=data)
plt.xlabel("Corps")
plt.ylabel("Fertilizers")
plt.figure(figsize=(20,15)) 
plt.show()


# In[57]:


from sklearn.preprocessing import LabelEncoder
label_encoders={}
categorical_columns = ['Soil','Crop','Fertilizer']
for col in categorical_columns:
    le=LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


# In[58]:


data['Fertilizer'].tail(50)


# In[59]:


data['Soil'].head(20)


# In[60]:


data['Crop'].tail(20)


# In[61]:


x = data.drop(columns=['Fertilizer','Remark'])
y = data["Fertilizer"]


# In[62]:


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


# In[63]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[64]:


importances = model1.feature_importances_
feature_names = x.columns
print("Feature Importances:")
for feature, importance in zip(feature_names, importances):
    print(f'{feature}: {importance:.4f}')


# In[65]:


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


# In[66]:


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


# In[67]:


import joblib
filename='Fertilizer_Prediction_APP'


# In[68]:


joblib.dump(model1,'Fertilizer_Prediction_APP')


# In[69]:


app=joblib.load('Fertilizer_Prediction_APP')


# In[70]:


array=[[20.939817,0.640556,37.361457,5.182518,42.242562,36.750004,39.370552,-0.280560,15,2]]
y_pre=app.predict(array)
y_pre


# In[71]:


fertilizer_name = label_encoders['Fertilizer'].inverse_transform(y_pre)
print(f'Recommended Fertilizer: {fertilizer_name[0]}')


# In[72]:


soil_encoded=data['Soil'].unique()
soil_name = label_encoders['Soil'].inverse_transform(soil_encoded)
print(soil_name) 


# In[73]:


fertilizer = data['Fertilizer'].unique()
ferti=label_encoders['Fertilizer'].inverse_transform(fertilizer)
print(ferti)


# In[74]:


data.head(10)


# In[75]:


import streamlit as st
import pandas as pd

st.title("Fertilizer Recommendation System 🌱")

# example input
nitrogen = st.number_input("Nitrogen")
phosphorus = st.number_input("Phosphorus")

if st.button("Predict"):
    st.write("Recommended Fertilizer: Urea")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




