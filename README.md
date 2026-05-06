# 🌍 Financial Inclusion Prediction in East Africa

## 📌 Project Overview
This project predicts the likelihood of an individual having a **bank account** in East Africa (**Kenya, Rwanda, Tanzania, and Uganda**) using demographic and socioeconomic data.  

It was completed as part of the **GOMYCODE Data Science Capstone Project**.

---

## 📊 Dataset
The dataset is provided by Zindi Africa. To respect the competition rules, the data files are **not hosted in this repository**.

🔗 **Data Source:** https://zindi.africa/competitions/financial-inclusion-in-africa/data

---

## 🌐 Live Demo (Streamlit App)
An interactive web application was built using Streamlit to allow users to make real-time predictions.

🚀 **Try the App:**  
https://financial-inclusion-in-east-africa-xju8efl3t6p2p4kn8c6d6a.streamlit.app/  

💡 *Input user details and instantly see the predicted likelihood of having a bank account.*

---

## 📈 Interactive Dashboard
An interactive dashboard was created using Tableau to explore key insights visually.

🔗 **View Dashboard:**  
https://public.tableau.com/views/FinancialInclusionEA/Dashboard2?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link  

💡 *For the best experience, open the dashboard in **fullscreen mode**.*

---

## 🔄 Methodology (CRISP-DM)

- 🧠 **Business Understanding:**  
  Identifying key factors that drive financial inclusion and support outreach strategies.

- 🧹 **Data Preparation:**  
  Data cleaning, handling missing values, One-Hot Encoding, and feature scaling.

- 🤖 **Modeling:**  
  Compared multiple machine learning models:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - XGBoost  
  - Support Vector Machine (SVM)  
  - Naive Bayes  

- 🏆 **Champion Model:**  
  **Random Forest Classifier**
  - Accuracy: **82%**
  - Recall: **75%**  
  - F1-Score: **0.53**  
  - ROC AUC: **0.87**

---

## 🔍 Key Insights

- 🎓 **Education Level** and 📱 **Cellphone Access** were the strongest predictors of financial inclusion.  

- 🎯 The model achieved a **Recall of ~74%**, meaning it successfully identified **3 out of 4 individuals** who have bank accounts.

---

## 💡 Final Takeaway

> Leveraging machine learning can significantly improve financial inclusion efforts by identifying individuals who are most likely to benefit from banking services.
