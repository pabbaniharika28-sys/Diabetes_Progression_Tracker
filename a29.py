import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
st.set_page_config(page_title="Diabetes Progression Tracker", page_icon="🩺", layout="wide")
if "diabetes_history" not in st.session_state:
    st.session_state.diabetes_history = []
st.title("🩺 Diabetes Progression Tracker")
st.markdown("Predict **1-year diabetes progression score** and simulate longitudinal trends.")
# ─── DATASET ───
data = {
    "PatientID":        [f"PT{str(i).zfill(3)}" for i in range(1, 51)],
    "Gender":           (["Male","Female"] * 25),
    "Age":              [45,52,38,61,None,48,55,35,62,None,
                          46,53,39,62,None,49,56,36,63,44,
                          45,52,38,61,None,48,55,35,62,None,
                          46,53,39,62,None,49,56,36,63,44,
                          45,52,38,61,None,48,55,35,62,None],
    "BMI":              [28.5,32.1,25.0,35.8,None,27.3,31.0,24.5,36.2,None,
                          29.0,33.0,25.5,36.0,None,28.0,31.5,25.0,36.5,26.5,
                          28.5,32.1,25.0,35.8,None,27.3,31.0,24.5,36.2,None,
                          29.0,33.0,25.5,36.0,None,28.0,31.5,25.0,36.5,26.5,
                          28.5,32.1,25.0,35.8,None,27.3,31.0,24.5,36.2,None],
    "BloodPressure":    [80,90,72,95,None,78,88,70,96,None,
                          82,91,73,96,None,79,89,71,97,75,
                          80,90,72,95,None,78,88,70,96,None,
                          82,91,73,96,None,79,89,71,97,75,
                          80,90,72,95,None,78,88,70,96,None],
    "GlucoseLevel":     [120,145,100,165,None,115,140,95,170,None,
                          122,147,102,167,None,117,142,97,172,108,
                          120,145,100,165,None,115,140,95,170,None,
                          122,147,102,167,None,117,142,97,172,108,
                          120,145,100,165,None,115,140,95,170,None],
    "InsulinLevel":     [80,150,60,200,None,75,140,55,210,None,
                          82,152,62,202,None,77,142,57,212,65,
                          80,150,60,200,None,75,140,55,210,None,
                          82,152,62,202,None,77,142,57,212,65,
                          80,150,60,200,None,75,140,55,210,None],
    "HbA1c":            [6.5,7.8,5.8,8.5,None,6.2,7.5,5.5,8.8,None,
                          6.6,7.9,5.9,8.6,None,6.3,7.6,5.6,8.9,5.9,
                          6.5,7.8,5.8,8.5,None,6.2,7.5,5.5,8.8,None,
                          6.6,7.9,5.9,8.6,None,6.3,7.6,5.6,8.9,5.9,
                          6.5,7.8,5.8,8.5,None,6.2,7.5,5.5,8.8,None],
    "PhysicalActivityScore":[6,3,8,2,None,7,4,9,1,None,
                              6,3,8,2,None,7,4,9,1,8,
                              6,3,8,2,None,7,4,9,1,None,
                              6,3,8,2,None,7,4,9,1,8,
                              6,3,8,2,None,7,4,9,1,None],
    "DietScore":        [6,4,8,3,5,7,4,9,2,6,
                          6,4,8,3,5,7,4,9,2,7,
                          6,4,8,3,5,7,4,9,2,6,
                          6,4,8,3,5,7,4,9,2,7,
                          6,4,8,3,5,7,4,9,2,6],
    "FamilyHistory":    [1,1,0,1,0,0,1,0,1,1,
                          1,1,0,1,0,0,1,0,1,0,
                          1,1,0,1,0,0,1,0,1,1,
                          1,1,0,1,0,0,1,0,1,0,
                          1,1,0,1,0,0,1,0,1,1],
    "ProgressionScore": [150,200,80,250,180,120,190,60,260,170,
                          155,205,82,255,185,125,195,62,265,100,
                          152,202,81,252,182,122,192,61,262,172,
                          158,208,83,258,188,128,198,63,268,102,
                          151,201,80,251,181,121,191,60,261,171]
}
df_raw = pd.DataFrame(data)
st.subheader("📋 Patient Dataset")
st.dataframe(df_raw)
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients",    len(df_raw))
col2.metric("Avg Progression",   f"{df_raw['ProgressionScore'].mean():.0f}")
col3.metric("Avg HbA1c",         f"{df_raw['HbA1c'].mean():.2f}%")
# ─── PREPROCESSING ───
st.subheader("🔧 Data Preprocessing")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ Null Check","2️⃣ Fill Nulls","3️⃣ Dedup & Drop",
    "4️⃣ Encoding","5️⃣ HbA1c Validation","6️⃣ Final Data"
])
with tab1:
    null_info = df_raw.isnull().sum().reset_index()
    null_info.columns = ["Column","Nulls"]
    null_info["% Missing"] = (null_info["Nulls"] / len(df_raw) * 100).round(1)
    st.dataframe(null_info)
    fig_null, ax_null = plt.subplots()
    cols_nn = null_info[null_info["Nulls"] > 0]
    ax_null.bar(cols_nn["Column"], cols_nn["% Missing"], color="#e74c3c")
    ax_null.set_title("Missing Values per Column")
    ax_null.set_xticklabels(cols_nn["Column"], rotation=30, ha="right")
    st.pyplot(fig_null)
with tab2:
    df = df_raw.copy()
    for col in ["Age","BMI","BloodPressure","GlucoseLevel","InsulinLevel","HbA1c","PhysicalActivityScore"]:
        df[col] = df[col].fillna(df[col].median())
    st.success("✅ Nulls filled with median!")
    st.dataframe(df.isnull().sum().rename("Remaining"))
with tab3:
    df = df.drop_duplicates().drop(["PatientID"], axis=1)
    st.success(f"Shape: **{df.shape}**")
with tab4:
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    st.write("Gender encoded: Male=0, Female=1")
    st.dataframe(df.head(8))
with tab5:
    valid_hba1c = ((df["HbA1c"] >= 4) & (df["HbA1c"] <= 15)).all()
    valid_gluc  = ((df["GlucoseLevel"] >= 50) & (df["GlucoseLevel"] <= 400)).all()
    st.write(f"{'✅' if valid_hba1c else '❌'} HbA1c in valid range (4–15%)")
    st.write(f"{'✅' if valid_gluc else '❌'} Glucose in valid range (50–400 mg/dL)")
    fig_hba, ax_hba = plt.subplots()
    ax_hba.hist(df["HbA1c"], bins=10, color="#9b59b6", edgecolor="white")
    ax_hba.axvline(x=5.7, color="green",  linestyle="--", label="Pre-diabetes threshold (5.7%)")
    ax_hba.axvline(x=6.5, color="red",    linestyle="--", label="Diabetes threshold (6.5%)")
    ax_hba.legend()
    ax_hba.set_title("HbA1c Distribution")
    st.pyplot(fig_hba)
with tab6:
    st.dataframe(df)
    st.info(f"Final shape: **{df.shape}**")
# ─── FEATURE ENGINEERING ───
st.subheader("⚙️ Feature Engineering")
df["GlucoseInsulinRatio"] = df["GlucoseLevel"] / (df["InsulinLevel"] + 1)
df["MetabolicRisk"]       = df["BMI"] * df["HbA1c"] / 10
df["LifestyleScore"]      = (df["PhysicalActivityScore"] + df["DietScore"]) / 2
df["CardioRisk"]          = df["BloodPressure"] * df["BMI"] / 100
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg BMI",          f"{df['BMI'].mean():.1f}")
col2.metric("Avg Blood Pressure",f"{df['BloodPressure'].mean():.0f} mmHg")
col3.metric("Avg Glucose",      f"{df['GlucoseLevel'].mean():.0f} mg/dL")
col4.metric("High HbA1c (>7)",  f"{(df['HbA1c'] > 7).sum()} patients")
# ─── VISUALIZATION ───
st.subheader("📊 Visualizations")
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["HbA1c"], df["ProgressionScore"], c=df["BMI"], cmap="RdYlGn_r", alpha=0.8)
    ax1.set_xlabel("HbA1c (%)")
    ax1.set_ylabel("Progression Score")
    ax1.set_title("HbA1c vs Progression (colored by BMI)")
    st.pyplot(fig1)
with col2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["GlucoseLevel"], df["InsulinLevel"], c=df["ProgressionScore"], cmap="plasma", alpha=0.8)
    ax2.set_xlabel("Glucose Level (mg/dL)")
    ax2.set_ylabel("Insulin Level")
    ax2.set_title("Glucose vs Insulin (colored by Progression)")
    st.pyplot(fig2)
col1, col2 = st.columns(2)
with col1:
    gender_prog = df.groupby("Gender")["ProgressionScore"].mean()
    fig3, ax3 = plt.subplots()
    ax3.bar(["Male","Female"], gender_prog.values, color=["#3498db","#e91e63"])
    ax3.set_title("Gender vs Avg Progression Score")
    ax3.set_ylabel("Progression Score")
    st.pyplot(fig3)
with col2:
    family_prog = df.groupby("FamilyHistory")["ProgressionScore"].mean()
    fig4, ax4 = plt.subplots()
    ax4.bar(["No Family History","Has Family History"], family_prog.values, color=["#2ecc71","#e74c3c"])
    ax4.set_title("Family History vs Progression")
    ax4.set_ylabel("Avg Progression Score")
    st.pyplot(fig4)
st.subheader("🔗 Correlation Matrix")
st.dataframe(df.corr(numeric_only=True))
# ─── OUTLIER ───
st.subheader("📦 Outlier Detection")
fig5, ax5 = plt.subplots()
ax5.boxplot([df["ProgressionScore"], df["GlucoseLevel"]/10, df["InsulinLevel"]],
            labels=["Progression", "Glucose÷10", "Insulin"])
ax5.set_title("Outlier Check")
st.pyplot(fig5)
# ─── MODEL ───
st.subheader("🤖 ElasticNet Regressor")
X = df.drop("ProgressionScore", axis=1)
y = df["ProgressionScore"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=2000)
model.fit(X_train, y_train)
st.success("✅ ElasticNet Model Trained!")
# ─── EVALUATION ───
st.subheader("📈 Evaluation")
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = math.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)
col1, col2, col3 = st.columns(3)
col1.metric("MAE",      f"{mae:.1f}")
col2.metric("RMSE",     f"{rmse:.1f}")
col3.metric("R² Score", f"{r2:.3f}")
fig6, ax6 = plt.subplots()
ax6.scatter(y_test, y_pred, color="#9b59b6", alpha=0.8)
ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
ax6.set_xlabel("Actual Progression Score")
ax6.set_ylabel("Predicted Progression Score")
ax6.set_title("Actual vs Predicted Progression")
st.pyplot(fig6)
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_}).sort_values("Coefficient")
fig_c, ax_c = plt.subplots()
colors = ["#e74c3c" if c < 0 else "#9b59b6" for c in coef_df["Coefficient"]]
ax_c.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
ax_c.axvline(x=0, color="black", linewidth=0.8)
ax_c.set_title("ElasticNet Coefficients")
st.pyplot(fig_c)
# ─── PREDICTION + TREND SIMULATION ───
st.subheader("🎯 Predict Progression + Trend Simulation")
col1, col2, col3 = st.columns(3)
with col1:
    gender   = st.selectbox("Gender", ["Male(0)","Female(1)"])
    age      = st.slider("Age", 20, 80, 50)
    bmi      = st.slider("BMI", 15.0, 50.0, 28.0)
with col2:
    bp       = st.slider("Blood Pressure (mmHg)", 60, 120, 82)
    glucose  = st.slider("Glucose Level (mg/dL)", 50, 400, 130)
    insulin  = st.slider("Insulin Level", 30, 300, 100)
with col3:
    hba1c    = st.slider("HbA1c (%)", 4.0, 14.0, 6.8)
    activity = st.slider("Physical Activity Score (1-10)", 1, 10, 5)
    diet     = st.slider("Diet Score (1-10)", 1, 10, 5)
    family   = st.selectbox("Family History?", ["No(0)","Yes(1)"])
if st.button("🩺 Predict & Simulate Trend"):
    g_val   = int(gender.split("(")[1].replace(")",""))
    f_val   = int(family.split("(")[1].replace(")",""))
    gi_ratio= glucose / (insulin + 1)
    met_risk= bmi * hba1c / 10
    life_sc = (activity + diet) / 2
    cardio  = bp * bmi / 100
    input_df = pd.DataFrame([[g_val, age, bmi, bp, glucose, insulin, hba1c,
                               activity, diet, f_val,
                               gi_ratio, met_risk, life_sc, cardio]],
                            columns=X.columns)
    pred_score = model.predict(input_df)[0]
    risk_level = "🔴 High Risk" if pred_score > 200 else "🟡 Moderate Risk" if pred_score > 130 else "🟢 Low Risk"
    st.success(f"🩺 **Predicted 1-Year Progression Score: {pred_score:.1f}**")
    st.info(f"📊 Risk Level: **{risk_level}** | Metabolic Risk: **{met_risk:.2f}** | Lifestyle Score: **{life_sc:.1f}**")
    # Trend Simulation (3-year projection)
    st.subheader("📈 3-Year Progression Trend Simulation")
    trend_scores = []
    trend_labels = ["Year 0 (Now)", "Year 1", "Year 2", "Year 3"]
    base_score = pred_score
    for yr in range(4):
        improvement = yr * life_sc * 2
        worsening   = yr * (1 - life_sc / 10) * 5
        trend_score = base_score - improvement + worsening
        trend_scores.append(max(50, trend_score))
    fig_trend, ax_trend = plt.subplots()
    colors_trend = ["#e74c3c" if s > 200 else "#f39c12" if s > 130 else "#2ecc71" for s in trend_scores]
    ax_trend.bar(trend_labels, trend_scores, color=colors_trend)
    ax_trend.axhline(y=200, color="red",    linestyle="--", label="High Risk threshold")
    ax_trend.axhline(y=130, color="orange", linestyle="--", label="Moderate Risk threshold")
    ax_trend.set_title("Projected Progression Score Over 3 Years")
    ax_trend.set_ylabel("Progression Score")
    ax_trend.legend()
    st.pyplot(fig_trend)
    st.write("💡 Trend assumes current lifestyle is maintained. Improving diet & activity score will reduce progression.")
    st.session_state.diabetes_history.append({
        "Age": age, "BMI": bmi, "HbA1c": hba1c,
        "Glucose": glucose, "Activity": activity, "Diet": diet,
        "Progression Score": f"{pred_score:.1f}", "Risk": risk_level
    })
if st.session_state.diabetes_history:
    st.subheader("🕓 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.diabetes_history))