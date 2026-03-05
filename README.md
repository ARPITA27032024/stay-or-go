🚀 Employee Attrition Prediction Dashboard

A machine learning powered web application that predicts whether an employee is likely to leave the organization based on various workplace and personal factors.

This project uses the IBM HR Analytics Employee Attrition Dataset and applies machine learning techniques to help organizations identify employees at risk of attrition and take preventive actions.

📊 Project Overview

Employee attrition is a major challenge for organizations because losing skilled employees can increase recruitment costs and affect productivity.

This project builds a predictive model that analyzes employee attributes such as age, job satisfaction, work-life balance, years at company, and income to predict the likelihood of attrition.

The model is deployed using Streamlit to create an interactive dashboard where users can input employee details and get instant predictions.

🧠 Problem Statement

Predict whether an employee will Stay or Leave the organization using machine learning.

The goal is to help HR departments:

Identify employees at risk of leaving

Improve employee retention strategies

Make data-driven HR decisions

📂 Dataset

Dataset used:

IBM HR Analytics Employee Attrition & Performance Dataset

Dataset Source: Kaggle

Key features in the dataset include:

Age

Monthly Income

Job Satisfaction

Environment Satisfaction

Work Life Balance

Years at Company

Years Since Promotion

Distance from Home

Job Level

Overtime

Target Variable:

Attrition
Yes = Employee leaves
No = Employee stays
⚙️ Technologies Used

Python

Pandas – Data preprocessing

NumPy – Numerical computations

Scikit-learn – Machine learning model

Streamlit – Web dashboard

CSS – UI styling

Pickle – Model serialization

🧪 Machine Learning Workflow

The project follows a standard ML pipeline:

1️⃣ Data Preprocessing

Handling missing values

Encoding categorical variables

Feature selection

2️⃣ Data Splitting

Training dataset

Testing dataset

3️⃣ Model Training

A classification algorithm was used to predict employee attrition.

4️⃣ Model Evaluation

Performance was evaluated using:

Accuracy

Precision

Recall

F1 Score

5️⃣ Model Deployment

The trained model was integrated into a Streamlit web application for real-time predictions.

🖥️ Dashboard Features

The Streamlit dashboard includes:

✔ Attractive UI
✔ Employee statistics overview
✔ Attrition risk prediction
✔ Easy input form for employee data
✔ Instant prediction results

Users can input employee details and click Predict Attrition Risk to see whether the employee is likely to stay or leave.

📸 Application Preview

Main Dashboard includes:

Total Employees

Attrition Rate

Average Income

Job Satisfaction

Employee Input Form

Attrition Prediction Result

📁 Project Structure
Employee-Attrition-Prediction
│
├── app.py                # Streamlit application
├── page_design.css       # UI styling
├── model.pkl             # Trained machine learning model
├── requirements.txt      # Project dependencies
├── dataset.csv           # Dataset used for training
└── README.md             # Project documentation
▶️ How to Run the Project
Step 1: Clone the repository
git clone https://github.com/yourusername/employee-attrition-prediction.git
Step 2: Install dependencies
pip install -r requirements.txt
Step 3: Run the Streamlit app
streamlit run app.py

The dashboard will open in your browser.

🎯 Future Improvements

Possible improvements for this project:

Add feature importance visualization

Add SHAP explanations for model transparency

Improve prediction accuracy with advanced models

Add department-level attrition insights

Deploy on Streamlit Cloud or AWS

👩‍💻 Author

Arpita Dash
B.Tech – Artificial Intelligence & Data Science
