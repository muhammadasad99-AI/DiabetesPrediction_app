import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib
from openai import OpenAI
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Diabetes Prediction App",
                   page_icon="💉", 
                   layout="wide")
st.title("💉 Diabetes Prediction and Insights App")
st.write("A machine learning web app that predicts the likelihood of diabetes based on patient health data.")

#import datasets of dabetes
df=pd.read_csv("Healthcare-Diabetes.csv")




x=df.drop(columns=["Outcome","Id"])
y=df["Outcome"]
scale=StandardScaler()
x_scale=scale.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x_scale,y,test_size=0.2,random_state=42)

#train the model with the train data(x,y)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)

#predict the outcomes for x more data
y_pred=model.predict(x_test)
acc=accuracy_score(y_test,y_pred)
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Diabetes Dataset Preview")
    st.dataframe(df.head())
with col2:
    
    st.subheader("⚙️ Model Information")
    st.write("Using: Random forest tree")
    st.write(f"Accuracy: **{acc*100:.2f}%**")
if st.checkbox("📈 Show Dataset Insights"):
       st.write("Basic Statistics:", df.describe())
       st.bar_chart(df["Outcome"].value_counts())




# In[60]:


def user_input():
    pregnancies=st.sidebar.number_input("pregnancies is : ",0,17,1)
    glucose=st.sidebar.number_input("gluscos rate is : ",0,199,120)
    bloodpressure=st.sidebar.number_input("blood pressure is : ",0,122,80)
    skinthickness=st.sidebar.number_input("skin thicknes is : ",0,110,23)
    insulin=st.sidebar.number_input("insulin is : ",0,846,80)
    bmi=st.sidebar.number_input("BMI is : ",0,80,30)
    diabetespedigreefunction=st.sidebar.number_input("DiabetesPedigreeFunction is : ",0.0780,2.4200,0.5)
    age=st.sidebar.number_input("age is : ",21,81,30)

    new_data={
       'Pregnancies': pregnancies,
       'Glucose': glucose,
       'BloodPressure' : bloodpressure,
       'SkinThickness' :skinthickness,
       'Insulin' :insulin,
       'BMI' : bmi,
       'DiabetesPedigreeFunction' : diabetespedigreefunction,
       'Age' : age
    }
    return pd.DataFrame(new_data,index=[0])

user_df=user_input()
st.subheader("User Input")
st.write(user_df)


if st.button("Predict Here"):
    scaler_input=scale.transform(user_df)
    prediction=model.predict(scaler_input)[0]

    if prediction == 1:
        st.error("sorry! patient has likely (diabetes)")
        st.info("Tips! : control your sugar level and exercise daily 30 minutes")
    else:
        st.success(" the patient is safe")
        st.info(" your ok!  maintain your daily routine and balanced diet")
# tips section about the diabetes 

 
#to save the file using joblib library
joblib.dump(model,"SVM_project.pkl")

#to load the saved file
#model=joblib.load("SVM_project.pkl")

#optional for others to upload their own datasets

uploaded = st.file_uploader("📂 Upload your diabetes data (CSV and Excel)", type=["csv",'xlsx'])

if uploaded is not None:
    if uploaded.name.endswith('.csv'):
        df_uploaded = pd.read_csv(uploaded)
    elif uploaded.name.endswith('.xlsx'):
        df_uploaded = pd.read_excel(uploaded)
    
    

    st.write("✅ File successfully uploaded!")
    st.dataframe(df_uploaded.head())

    # Scale the input before prediction 
    scaled_uploaded = scale.transform(df_uploaded)

    # Make predictions
    uploaded_pred = model.predict(scaled_uploaded)

    st.subheader("🔮 Uploaded File Predictions")
    df_uploaded["Predicted Outcome"] = uploaded_pred
    st.dataframe(df_uploaded)

    st.success("Prediction completed successfully!")
    
#the feedback option from user
feedback=st.text_input("Give your feedback about the prediction!")
if feedback:
    st.write("Thanks for feedback")
    


# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if api_key is None:
    st.error("API key not found! Make sure it is saved in the .env file.")
else:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

st.title("💬 Diabetes AI Assistant (via OpenRouter)")

prompt = st.text_input("Ask a diabetes-related question:")

if st.button("Ask"):
    if not prompt.strip():
        st.warning("Please enter a question first!")
    else:
        with st.spinner("Thinking... 🧠"):
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "Diabetes AI Assistant",
                },
                model="openai/gpt-oss-20b:free",
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant that answers diabetes-related questions clearly and safely."},
                    {"role": "user", "content": prompt}
                ]
            )

        st.subheader("🩺 AI Response:")
        st.write(response.choices[0].message.content)

