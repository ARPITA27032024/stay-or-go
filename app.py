import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Stay-Or-Go", layout="wide")

# Load CSS
with open("page_design.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    features = pickle.load(open("features.pkl", "rb"))
    metrics = pickle.load(open("metrics.pkl", "rb"))
    return model, scaler, encoders, features, metrics

model, scaler, encoders, features, metrics = load_models()

# Load dataset for stats
df = pd.read_csv("HR-Employee-Attrition.csv")
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Header
st.markdown(""" <div class="main-header"> <h1>ִ ࣪𖤐⋆ Stay - Or - Go ᯓ✦ </h1>
    <p>This Employee Attrition Predictor will find out chances of an employee leaving the company.</p> </div>""", unsafe_allow_html=True)

# KPI Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""<div class="kpi-card">
        <h3>Total Employees</h3> <div class="value">{len(df)}</div> </div> """, unsafe_allow_html=True)

with col2:
    attrition_rate = round(df["Attrition"].mean() * 100, 1)
    st.markdown(f""" <div class="kpi-card">
        <h3>Attrition Rate</h3> <div class="value">{attrition_rate}%</div> </div> """, unsafe_allow_html=True)

with col3:
    avg_income = round(df["MonthlyIncome"].mean())
    st.markdown(f""" <div class="kpi-card">
        <h3>Avg Income</h3> <div class="value">${avg_income:,.0f}</div> </div> """, unsafe_allow_html=True)

with col4:
    avg_satisfaction = round(df["JobSatisfaction"].mean(), 1)
    st.markdown(f""" <div class="kpi-card">
        <h3>Job Satisfaction</h3> <div class="value">{avg_satisfaction}/4</div> </div> """, unsafe_allow_html=True)

# Input Section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<p class="section-title">📋 Employee Details</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 60, 35)
    monthly_income = st.slider("Monthly Income ($)", 2000, 20000, 5000, step=500)
    years_at_company = st.slider("Years at Company", 0, 40, 5)

with col2:
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    work_life_balance = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
    overtime = st.selectbox("Works Overtime?", ["No", "Yes"])

with col3:
    job_level = st.slider("Job Level", 1, 5, 2)
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

st.markdown('</div>', unsafe_allow_html=True)

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict = st.button("🔮 PREDICT ATTRITION RISK", use_container_width=True)

if predict:
    input_data = pd.DataFrame([{"Age": age, "MonthlyIncome": monthly_income, "YearsAtCompany": years_at_company, "WorkLifeBalance": work_life_balance,
        "OverTime": overtime, "JobLevel": job_level, "Department": department, "MaritalStatus": marital_status,"BusinessTravel": "Travel_Rarely",
        "DailyRate": 800, "DistanceFromHome": 5, "Education": 3, "EducationField": "Life Sciences", "EnvironmentSatisfaction": 3, "Gender": "Male",
        "HourlyRate": 65, "JobInvolvement": 3, "JobRole": "Sales Executive", "MonthlyRate": 14000, "NumCompaniesWorked": 3, "PercentSalaryHike": 15,
        "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 1, "TotalWorkingYears": 10, "TrainingTimesLastYear": 3,
        "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3}])
    
    # Encode categorical variables
    for col, encoder in encoders.items():
        if col in input_data.columns:
            try:
                input_data[col] = encoder.transform(input_data[col])
            except:
                input_data[col] = 0
    
    # Ensure all features are present
    input_data = input_data.reindex(columns=features, fill_value=0)
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    probability = model.predict_proba(input_scaled)[0][1] * 100
    
    # Show result
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    # Create two columns for visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            number={'suffix': "%", 'font': {'size': 50, 'color': '#FFFFFF'}},
            title={'text': "Attrition Risk", 'font': {'size': 24, 'color': '#FFFFFF'}},
            gauge={'axis': {'range': [0, 100], 'tickcolor': '#8AB0FF'},
                'bar': {'color': "#4A7AB0", 'thickness': 0.3},
                'bgcolor': '#0A1A3A',
                'borderwidth': 2,
                'bordercolor': '#2A4A7A',
                'steps': [
                    {'range': [0, 30], 'color': '#1A3A2A'},
                    {'range': [30, 70], 'color': '#4A3A1A'},
                    {'range': [70, 100], 'color': '#4A1A1A'}
                ],
                'threshold': {'line': {'color': '#FFFFFF', 'width': 4},'thickness': 0.75,'value': probability}}))
        fig_gauge.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': '#FFFFFF'})
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with viz_col2:
        risk_levels = pd.DataFrame({
            'Risk Category': ['Low Risk', 'Medium Risk', 'High Risk', 'Current'],
            'Threshold': [15, 50, 85, probability],
            'Color': ['#2A7A4A', '#B07A2A', '#B04A4A', '#4A7AB0']})
        
        fig_line = go.Figure()
        
        fig_line.add_trace(go.Scatter(
            x=['Low', 'Medium', 'High'],
            y=[30, 70, 100],
            mode='lines+markers',
            name='Risk Thresholds',
            line=dict(color='#4A7AB0', width=3, dash='dot'),
            marker=dict(size=10, color=['#2A7A4A', '#B07A2A', '#B04A4A'])))
        
        fig_line.add_trace(go.Scatter(
            x=['Current'],
            y=[probability],
            mode='markers',
            name=f'Current Risk: {probability:.1f}%',
            marker=dict(size=20, color='#FFFFFF', symbol='star',
                       line=dict(color='#4A7AB0', width=2))))
        
        fig_line.update_layout(
            title="Risk Level Analysis",
            xaxis_title="Risk Category",
            yaxis_title="Risk Percentage (%)",
            yaxis_range=[0, 100],
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': '#FFFFFF'},
            xaxis={'gridcolor': '#2A4A7A'},
            yaxis={'gridcolor': '#2A4A7A'})
            
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Risk badge
    if probability < 30:
        st.markdown(f'<div class="risk-low">✅ LOW RISK • {probability:.1f}% chance of leaving</div>', unsafe_allow_html=True)
    elif probability < 70:
        st.markdown(f'<div class="risk-medium">⚠️ MEDIUM RISK • {probability:.1f}% chance of leaving</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="risk-high">🔴 HIGH RISK • {probability:.1f}% chance of leaving</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Model Performance Section
st.markdown('<div class="metrics-section">', unsafe_allow_html=True)
st.markdown('<p class="section-title">📊 Model Performance</p>', unsafe_allow_html=True)

# METRICS
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""<div class="metric-box">  <h4>ACCURACY</h4>
        <div class="number">{metrics['accuracy']*100:.1f}%</div> </div> """, unsafe_allow_html=True)
with col2:
    st.markdown(f""" <div class="metric-box"> <h4>RECALL</h4>
        <div class="number">{metrics['recall']*100:.1f}%</div> </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f""" <div class="metric-box"> <h4>F1 SCORE</h4>
        <div class="number">{metrics['f1']*100:.1f}%</div> </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📈 Classification Report")

report_df = pd.DataFrame(metrics['report']).T
report_df = report_df.round(3)

def color_cells(val):
    if isinstance(val, (int, float)):
        if val > 0.8:
            return 'color: #2A7A4A; font-weight: bold'
        elif val > 0.6:
            return 'color: #B07A2A'
        else:
            return 'color: #B04A4A'
    return 'color: #E0E8FF'

styled_df = report_df.style.applymap(color_cells).set_properties(**{
    'background-color': '#0A1A3A', 'border': '1px solid #2A4A7A','color': '#E0E8FF','text-align': 'center'
}).set_table_styles([
    {'selector': 'th', 'props': [('background', 'linear-gradient(145deg, #1A2F5A 0%, #0A1A3A 100%)'),
                                 ('color', '#FFFFFF'),
                                 ('border', '1px solid #2A4A7A'),
                                 ('padding', '12px')]},
    {'selector': 'td', 'props': [('padding', '10px')]},
    {'selector': 'tr:hover td', 'props': [('background', 'rgba(42, 74, 122, 0.3)')]}
])

st.dataframe(styled_df, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)